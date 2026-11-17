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
"""Runtime gradient types and forward cache for HyperMegaMhcGrad."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..mega_mhc_common import MHC_MAPPINGS, MHC_SINKHORN_ITERS, MHC_STREAMS, validate_mega_mhc_tensor


MegaMhcGradOutputs = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


@dataclass(frozen=True)
class MegaMhcGradCache:
    """Forward tensors required by the native backward kernels."""

    new_residual: torch.Tensor
    hc_before_norm: torch.Tensor
    inv_rms: torch.Tensor
    sum_out: torch.Tensor
    norm_out: torch.Tensor
    mixed_input: torch.Tensor
    rms_rstd: torch.Tensor


def validate_mega_mhc_grad_tensors(
    gradients: tuple[torch.Tensor, ...], residual: torch.Tensor, cache: MegaMhcGradCache,
) -> None:
    """Validate upstream gradients and every native forward-cache tensor.

    Args:
        gradients: Gradients of the five public forward outputs, in output order.
        residual: Original residual streams defining the shape and device.
        cache: Forward tensors retained for native backward.
    """
    if not isinstance(cache, MegaMhcGradCache):
        raise TypeError("cache must be a MegaMhcGradCache.")
    leading_shape = tuple(residual.shape[:-2])
    hidden_size = residual.shape[-1]
    tokens = residual.numel() // (MHC_STREAMS * hidden_size)
    gradient_specs = (
        ("grad_new_residual", (*leading_shape, MHC_STREAMS, hidden_size), torch.bfloat16),
        ("grad_next_pre_mix", (*leading_shape, MHC_STREAMS), torch.float32),
        ("grad_next_post_mix", (*leading_shape, MHC_STREAMS), torch.float32),
        ("grad_next_residual_mix", (*leading_shape, MHC_STREAMS, MHC_STREAMS), torch.float32),
        ("grad_block_input", (*leading_shape, hidden_size), torch.bfloat16),
    )
    if len(gradients) != len(gradient_specs):
        raise ValueError(f"Expected five output gradients, got {len(gradients)}.")
    for value, (name, shape, dtype) in zip(gradients, gradient_specs):
        validate_mega_mhc_tensor(name, value, shape, dtype, residual.device)
    cache_specs = (
        ("new_residual", cache.new_residual, (1, tokens, MHC_STREAMS, hidden_size), torch.bfloat16),
        ("hc_before_norm", cache.hc_before_norm, (1, tokens, MHC_MAPPINGS), torch.float32),
        ("inv_rms", cache.inv_rms, (1, tokens, 1), torch.float32),
        ("sum_out", cache.sum_out, (2 * MHC_SINKHORN_ITERS, 1, tokens, MHC_STREAMS), torch.float32),
        ("norm_out", cache.norm_out, (2 * MHC_SINKHORN_ITERS, 1, tokens, MHC_STREAMS, MHC_STREAMS), torch.float32),
        ("mixed_input", cache.mixed_input, (1, tokens, hidden_size), torch.bfloat16),
        ("rms_rstd", cache.rms_rstd, (1, tokens, 1), torch.float32),
    )
    for name, value, shape, dtype in cache_specs:
        validate_mega_mhc_tensor(f"cache.{name}", value, shape, dtype, residual.device)
