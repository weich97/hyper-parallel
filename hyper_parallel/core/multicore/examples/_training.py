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

"""Model-neutral optimizer construction and common/Mega tensor comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.components.optim import Float16OptimizerWithFloat16Params
from hyper_parallel.core.optimizer import get_hyper_optimizer


@dataclass
class _TensorComparison:
    """Device-side pass state and per-tensor comparison diagnostics."""

    passed: torch.Tensor
    tensor_names: list[str] = field(default_factory=list)
    absolute_errors: list[torch.Tensor] = field(default_factory=list)
    normalized_errors: list[torch.Tensor] = field(default_factory=list)


def _compare_tensor_maps(
    name: str,
    common_tensors: dict[str, torch.Tensor],
    mega_tensors: dict[str, torch.Tensor],
    device: torch.device,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Compare tensor maps on device and reduce diagnostics across ranks."""
    if common_tensors.keys() != mega_tensors.keys():
        raise RuntimeError(
            f"{name} tensor names differ: common={sorted(common_tensors)}, "
            f"mega={sorted(mega_tensors)}."
        )
    comparison = _TensorComparison(
        passed=torch.ones((), dtype=torch.int32, device=device)
    )
    for tensor_name, common_tensor in common_tensors.items():
        _compare_tensor_pair(
            name,
            tensor_name,
            common_tensor,
            mega_tensors[tensor_name],
            comparison,
            rtol=rtol,
            atol=atol,
        )
    return _finalize_tensor_comparison(name, comparison, rtol=rtol, atol=atol)


def _compare_tensor_pair(
    comparison_name: str,
    tensor_name: str,
    common_tensor: torch.Tensor,
    mega_tensor: torch.Tensor,
    comparison: _TensorComparison,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Accumulate device-side diagnostics for one corresponding tensor pair."""
    comparison.tensor_names.append(tensor_name)
    if mega_tensor.shape != common_tensor.shape:
        raise RuntimeError(
            f"{comparison_name}.{tensor_name} shape differs: common={tuple(common_tensor.shape)}, "
            f"mega={tuple(mega_tensor.shape)}."
        )
    common_float = common_tensor.detach().float()
    mega_float = mega_tensor.detach().float()
    difference = (mega_float - common_float).abs()
    comparison.passed = torch.minimum(
        comparison.passed,
        torch.isclose(mega_float, common_float, rtol=rtol, atol=atol)
        .all()
        .to(torch.int32),
    )
    comparison.absolute_errors.append(difference.max())
    tolerance = atol + rtol * common_float.abs()
    normalized_error = difference / tolerance.clamp_min(torch.finfo(torch.float32).tiny)
    comparison.normalized_errors.append(normalized_error.max())


def _finalize_tensor_comparison(
    name: str,
    comparison: _TensorComparison,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Reduce accumulated diagnostics and return the public result mapping."""
    absolute_by_tensor = torch.stack(comparison.absolute_errors)
    normalized_by_tensor = torch.stack(comparison.normalized_errors)
    if dist.is_initialized():
        dist.all_reduce(comparison.passed, op=dist.ReduceOp.MIN)
        dist.all_reduce(absolute_by_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(normalized_by_tensor, op=dist.ReduceOp.MAX)
    maximum_absolute, absolute_index = absolute_by_tensor.max(dim=0)
    maximum_normalized, normalized_index = normalized_by_tensor.max(dim=0)
    result = {
        "passed": bool(comparison.passed.cpu().item()),
        "tensor_count": len(comparison.tensor_names),
        "rtol": rtol,
        "atol": atol,
        "max_absolute_error": float(maximum_absolute.cpu().item()),
        "max_absolute_error_tensor": comparison.tensor_names[int(absolute_index.cpu().item())],
        "max_normalized_error": float(maximum_normalized.cpu().item()),
        "max_normalized_error_tensor": comparison.tensor_names[int(normalized_index.cpu().item())],
    }
    if not result["passed"]:
        raise RuntimeError(f"common/Mega {name} comparison failed: {result}.")
    return result


def build_optimizer(
    model: torch.nn.Module, *, learning_rate: float, weight_decay: float,
) -> Float16OptimizerWithFloat16Params:
    """Create independent AdamW state with FP32 main parameters for an example model.

    Args:
        model: Final model after choosing its execution backend.
        learning_rate: Positive optimizer learning rate.
        weight_decay: Nonnegative weight decay.
    """
    optimizer = get_hyper_optimizer(
        model=model, muon_params=[],
        adamw_params=[{"params": list(model.parameters()), "weight_decay": weight_decay}],
        adamw_kwargs={"lr": learning_rate, "betas": (0.9, 0.999), "eps": 1e-8},
    )
    return Float16OptimizerWithFloat16Params(optimizer, model)
