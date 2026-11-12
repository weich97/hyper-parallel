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
"""Lossless staging of owned expert gradients between microbatches."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.core.optimizer.dtensor_compat import to_local_if_dtensor
from hyper_parallel.trainer.runtime.logging import create_logger


logger = create_logger(__name__)


class ExpertGradientStaging:
    """Keep the first exact BF16 partial and later accumulated partials in FP32."""

    def __init__(self, parameters: tuple[torch.nn.Parameter, ...], host_count: int = 0) -> None:
        """Own the gradient partials for a set of local expert parameters.

        Args:
            parameters: Parameters whose local FP32 gradients are staged.
            host_count: Number of first-partial tensors to store on the host.
        """
        self.parameters = parameters
        self.host_count = host_count
        self.active = False

    def _local_gradients(self) -> tuple[torch.Tensor, ...]:
        """Fetch the sharded FP32 accumulator for each participating expert weight."""
        gradients = tuple(to_local_if_dtensor(getattr(parameter, "main_grad", None))
                          for parameter in self.parameters)
        if any(gradient is None or gradient.dtype != torch.float32 for gradient in gradients):
            raise RuntimeError("Expert staging requires every reduced gradient shard in FP32")
        return gradients

    @torch.no_grad()
    def pack(self, *, first_partial: bool = True) -> None:
        """Stage a completed partial before the next backward.

        Args:
            first_partial: Check and compact the initial BF16-representable gradient.
                Later accumulated gradients remain FP32 and are staged entirely on the host.
        """
        if self.active:
            raise RuntimeError("Expert gradient staging already owns an unconsumed partial")
        gradients = self._local_gradients()
        if first_partial:
            packed = tuple(gradient.to(dtype=torch.bfloat16) for gradient in gradients)
            lossless = torch.stack([
                (original == compact).all() for original, compact in zip(gradients, packed, strict=True)
            ]).all().to(torch.int32)
            if dist.is_initialized():
                dist.all_reduce(lossless, op=dist.ReduceOp.MIN)
            if not lossless.item():
                raise RuntimeError("First expert gradient partial has values outside the BF16 format")
        else:
            # A sum of BF16 gradients is generally not BF16-representable. Preserve
            # the FP32 accumulator and its addition order across subsequent microbatches.
            packed = gradients
        staged = list(packed)
        host_count = self.host_count if first_partial else len(packed)
        if packed[0].device.type == "cpu":
            host_count = 0
        ready = None
        if host_count:
            for index, compact in enumerate(packed[:host_count]):
                host = torch.empty_like(compact, device="cpu", pin_memory=True)
                host.copy_(compact, non_blocking=True)
                # Later FP32 partials may have been allocated on the reduction
                # stream. Keep their storage alive until this D2H copy finishes.
                compact.record_stream(torch.npu.current_stream(compact.device))
                staged[index] = host
            ready = torch.npu.Event()
            ready.record(torch.npu.current_stream(packed[0].device))
        for parameter, compact in zip(self.parameters, staged, strict=True):
            parameter._hp_staged_main_grad = (compact, ready if compact.device.type == "cpu" else None)
            parameter.main_grad = None
        self.active = True

    def finish(self) -> None:
        """Require the FSDP reduction path to have consumed all staged partials."""
        if not self.active:
            raise RuntimeError("Expert gradient staging has no pending partial")
        if any(hasattr(parameter, "_hp_staged_main_grad") for parameter in self.parameters):
            raise RuntimeError("Expert gradients were not consumed during the next backward")
        self._local_gradients()
        self.active = False


def _supports_owned_expert_staging(base: Any) -> bool:
    """Require a reduction boundary exposing each staged local expert partial."""
    config = base.config
    model_type = (getattr(base.model_config, "v41_source_model_type", None)
                  or getattr(base.model_config, "model_type", ""))
    return all((
        base.num_micro_batches >= 2,
        model_type == "deepseek_v41",
        config.accelerator.ep_size == 128,
        config.accelerator.cp_size == 1,
        config.accelerator.tp_size == 1,
        config.fsdp_config.edp_shard_size == 1,
        not config.fsdp_config.comm_fusion,
        config.fsdp_config.dp_shard_size in (32, 64, 128),
        config.fsdp_config.requires_grad_sync,
        config.optimizer.fp32_main_params,
        # Full sharding does not expose intermediate local expert reductions.
        base.num_micro_batches == 2 or config.fsdp_config.dp_shard_size != 128,
    ))


def build_expert_gradient_staging(base: Any) -> ExpertGradientStaging | None:
    """Select owned MegaMoE weights with supported microbatch reduction boundaries.

    Args:
        base: Trainer state containing the model and distributed configuration.

    Returns:
        A staging owner, or None when the layout cannot expose every required partial.
    """
    config = base.config
    if not _supports_owned_expert_staging(base):
        return None
    # Import the model adapter only for the optional V4.1 execution path.
    from hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe import (  # pylint: disable=C0415
        DeepseekV41MegaMoeExperts,
    )

    experts = tuple(layer.mlp.experts for layer in base.model.model.layers)
    if not experts or not all(isinstance(module, DeepseekV41MegaMoeExperts) for module in experts):
        return None
    parameters = tuple(weight for module in experts for weight in (module.gate_up_proj, module.down_proj))
    if config.fsdp_config.dp_shard_size != 128:
        # Dense HSDP delays its replica all-reduce. EDP=1 expert weights have one owner and can stage now.
        for parameter in parameters:
            parameter._hp_stage_local_main_grad = True
    host_count = 2 * min(16, len(experts)) if base.device.type == "npu" else 0
    if base.global_rank == 0:
        elements = sum(to_local_if_dtensor(parameter).numel() for parameter in parameters)
        logger.info("V4.1 lossless streaming gradient staging weights=%d host_weights=%d compact_bytes=%d",
                    len(parameters), host_count, 2 * elements)
        logger.info("V4.1 expert accumulation microbatches=%d later_partial_dtype=float32 later_storage=host",
                    base.num_micro_batches)
    return ExpertGradientStaging(parameters, host_count)
