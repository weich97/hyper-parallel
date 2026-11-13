# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Declarative MegaMoe expert replacement for DeepSeek-V4.1 training."""

from __future__ import annotations

from collections.abc import Mapping
import logging
import math
from typing import Any, TYPE_CHECKING

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from transformers.activations import ACT2FN

from hyper_parallel.components.checkpoint.weight_conversion import Transpose, WeightConverter
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.models.replacement import module_replacement

if TYPE_CHECKING:
    from hyper_parallel.core.multicore import MegaMoeExperts


logger = logging.getLogger(__name__)


def _swiglu_limit(value: float) -> float:
    """Require the positive clamp used by the source DeepSeek experts."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("DSV4.1 swiglu_limit must be finite and positive")
    return float(value)


def _validate_execution_layout(fixed_route_tokens: int | None, token_chunk_size: int | None,
                               expert_weight_layout: str) -> None:
    """Validate layout and equal per-rank token chunks before allocating parameters."""
    if fixed_route_tokens is not None and (
            not isinstance(fixed_route_tokens, int) or isinstance(fixed_route_tokens, bool) or fixed_route_tokens < 1):
        raise ValueError("fixed_route_tokens must be a positive integer")
    if expert_weight_layout not in ("EIO", "EOI"):
        raise ValueError("expert_weight_layout must be EIO or EOI")
    if token_chunk_size is None:
        return
    if (not isinstance(token_chunk_size, int) or isinstance(token_chunk_size, bool)
            or token_chunk_size < 128 or token_chunk_size % 128):
        raise ValueError("token_chunk_size must be a positive multiple of 128")
    if fixed_route_tokens is None or fixed_route_tokens % token_chunk_size:
        raise ValueError("token_chunk_size requires evenly divisible fixed_route_tokens")


@module_replacement
class DeepseekV41MegaMoeExperts(nn.Module):
    """Own native-layout parameters while retaining the HF checkpoint names."""

    _native_forward_reported = False

    def __init__(self, *, module: nn.Module, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None,
                 dispatch_mode: str = "push", initial_capacity_factor: float | None = None,
                 capacity_growth_factor: float | None = None, initializer_range: float = 0.02,
                 fixed_route_tokens: int | None = None, combine_split: int = 128,
                 use_matmul_weight_grad: bool = False, expert_weight_layout: str = "EIO",
                 token_chunk_size: int | None = None) -> None:
        """Convert a full HF holder before EP partitioning and FSDP construction.

        Args:
            module: HF DeepSeek routed experts with a supported SwiGLU limit.
            module_fqn: Checkpoint scope assigned by the replacement executor.
            context: Active parallel axes and sequence_parallel; TP requires SP and EP>1, and PP must be 1.
            dispatch_mode: MegaMoe dispatch transport, either push or pull.
            initial_capacity_factor: Push initial receive capacity; None uses the executor default.
            capacity_growth_factor: Push capacity growth multiplier; omit for pull.
            initializer_range: Standard deviation used for meta-model random initialization.
            fixed_route_tokens: Reuse route metadata for this fixed benchmark shape.
            combine_split: Number of rows in each native token combine tile.
            use_matmul_weight_grad: Use separate device GEMMs for expert weight gradients.
            expert_weight_layout: EIO native layout or EOI to retain the source parameter layout.
            token_chunk_size: Bound native workspace with equal fixed-routing token chunks.
                Requires fixed_route_tokens divisible by this positive multiple of 128.
        """
        super().__init__()
        del module_fqn
        context = context or {}
        if context.get("pp", False):
            raise ValueError("DSV4.1 MegaMoe requires PP=1")
        if context.get("tp", False) and not context.get("ep", False):
            raise ValueError("DSV4.1 MegaMoe with TP>1 requires EP>1; expert tensor parallelism is unsupported")
        self.sequence_parallel = bool(context.get("sequence_parallel", False))
        if context.get("tp", False) and not self.sequence_parallel:
            raise ValueError("DSV4.1 MegaMoe with TP>1 requires sequence_parallel=true")
        self.dispatch_mode = dispatch_mode
        self.initial_capacity_factor = initial_capacity_factor
        self.capacity_growth_factor = capacity_growth_factor
        _validate_execution_layout(fixed_route_tokens, token_chunk_size, expert_weight_layout)
        self.fixed_route_tokens = fixed_route_tokens
        self.token_chunk_size = token_chunk_size
        self.expert_weight_layout = expert_weight_layout
        self.combine_split = combine_split
        self.use_matmul_weight_grad = use_matmul_weight_grad
        source_limit = _swiglu_limit(module.limit)
        if not isinstance(module.act_fn, (nn.SiLU, type(ACT2FN["silu"]))):
            raise ValueError("MegaMoe Trainer replacement requires SiLU")
        self.global_experts = self.num_experts = module.num_experts
        self.hidden_size = module.hidden_dim
        self.intermediate_size = module.intermediate_dim
        self.is_transposed = expert_weight_layout == "EIO"
        self.initializer_range = initializer_range
        self.swiglu_limit = source_limit
        if not self.is_transposed:
            # Identity-layout replacements retain the registered source contract.
            self.act_fn = module.act_fn
        for name in ("gate_up_proj", "down_proj"):
            source = getattr(module, name)
            parameter = (nn.Parameter(source.detach().transpose(1, 2).contiguous(),
                                      requires_grad=source.requires_grad)
                         if self.is_transposed else source)
            setattr(self, name, parameter)
        self._executor: MegaMoeExperts | None = None
        self.train(module.training)

    def reset_parameters(self) -> None:
        """Initialize native-layout weights after meta materialization."""
        nn.init.normal_(self.gate_up_proj, std=self.initializer_range)
        nn.init.normal_(self.down_proj, std=self.initializer_range)

    def make_transforms(self) -> list[WeightConverter]:
        """Describe the reversible HF-to-native checkpoint layout conversion."""
        if not self.is_transposed:
            return []
        return [WeightConverter(source_patterns=name, target_patterns=name,
                                operations=[Transpose(dim0=-2, dim1=-1)])
                for name in ("gate_up_proj", "down_proj")]

    def configure(self, ep_group: Any, ep_size: int, top_k: int, *, fix_router: bool = False) -> None:
        """Bind the EP mesh and routing width after model replacement.

        Args:
            ep_group: Ordered EP process group, including a subgroup of the training world.
            ep_size: Number of EP ranks.
            top_k: Number of selected experts per token.
            fix_router: Indicates the explicit benchmark-only balanced destinations.
        """
        if self._executor is not None:
            raise RuntimeError("Cannot reconfigure an active MegaMoe expert executor")
        if self.fixed_route_tokens is not None and not fix_router:
            raise ValueError("Static MegaMoe routes require explicit fixed expert destinations")
        if self.token_chunk_size is not None and self.token_chunk_size * top_k % self.global_experts:
            raise ValueError("Every fixed-routing token chunk must contain complete expert-assignment cycles")
        # Multicore is optional for the default native-EP replacement provider.
        from hyper_parallel.core.multicore import MegaMoeExperts  # pylint: disable=C0415

        self._executor = MegaMoeExperts(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size, num_experts=self.global_experts,
            top_k=top_k, ep_group=ep_group, ep_size=ep_size,
            dispatch_mode=self.dispatch_mode, initial_capacity_factor=self.initial_capacity_factor,
            capacity_growth_factor=self.capacity_growth_factor, swiglu_limit=self.swiglu_limit,
            create_parameters=False, combine_split=self.combine_split,
            use_matmul_weight_grad=self.use_matmul_weight_grad,
            expert_weight_layout=self.expert_weight_layout,
        )

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        """Execute with the current parameters supplied by FSDP unshard hooks.

        Args:
            hidden_states: Token states before any EP dispatch.
            top_k_index: Global expert IDs selected by the unchanged router.
            top_k_weights: Differentiable router weights, including source scaling.

        Returns:
            Routed output with the same shape as hidden_states.
        """
        if self._executor is None:
            raise RuntimeError("Configure MegaMoe EP execution before forward")
        weights = tuple(parameter.to_local() if isinstance(parameter, DTensor) else parameter
                        for parameter in (self.gate_up_proj, self.down_proj))
        if self.token_chunk_size is None:
            output = self._executor(
                hidden_states, top_k_index, top_k_weights,
                expert_weights=weights, fixed_route_tokens=self.fixed_route_tokens,
            )
        else:
            output = self._forward_chunks(hidden_states, top_k_index, top_k_weights, weights)
        if not DeepseekV41MegaMoeExperts._native_forward_reported:
            logger.info_rank0("DeepSeek-V4.1 native MegaMoE engaged")
            DeepseekV41MegaMoeExperts._native_forward_reported = True
        return output

    def _forward_chunks(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                        top_k_weights: torch.Tensor, weights: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Bound each fused dispatch/compute/combine call inside one FSDP expert boundary."""
        states = hidden_states.reshape(-1, self.hidden_size)
        if states.shape[0] != self.fixed_route_tokens:
            raise ValueError("Chunked MegaMoe requires the declared fixed total token count on every EP rank")
        expected = (self.fixed_route_tokens, self._executor.top_k)
        if tuple(top_k_index.shape) != expected or tuple(top_k_weights.shape) != expected:
            raise ValueError("Chunked MegaMoe requires one routing row per flattened input token")
        outputs = []
        for start in range(0, self.fixed_route_tokens, self.token_chunk_size):
            end = start + self.token_chunk_size
            outputs.append(self._executor(
                states[start:end], top_k_index[start:end], top_k_weights[start:end],
                expert_weights=weights, fixed_route_tokens=self.token_chunk_size,
            ))
        return torch.cat(outputs, dim=0).reshape_as(hidden_states)

    @staticmethod
    def prepare_runtime_group(experts: list[DeepseekV41MegaMoeExperts]) -> None:
        """Share transient storage across configured, serial decoder layers.

        Args:
            experts: Experts with compatible shapes and the same EP process group.
        """
        if not experts:
            return
        # Sharing is internal to these peers; their executors remain encapsulated.
        executors = [module._executor for module in experts]  # pylint: disable=protected-access
        if any(executor is None for executor in executors):
            raise RuntimeError("Configure every expert before sharing execution resources")
        # Import only when explicitly sharing configured MegaMoe executors.
        from hyper_parallel.core.multicore import MegaMoeExperts  # pylint: disable=C0415

        MegaMoeExperts.share_execution_resources(executors)

    def close_runtime(self) -> None:
        """Release native resources after the final backward has completed."""
        if self._executor is not None:
            self._executor.close()
            self._executor = None


__all__ = ["DeepseekV41MegaMoeExperts"]
