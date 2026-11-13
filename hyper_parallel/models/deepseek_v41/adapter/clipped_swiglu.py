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
"""Packed V4.1 SwiGLU with native forward and bounded first-order backward."""

import math
from typing import Any

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional


def _forward(packed: torch.Tensor, limit: float) -> torch.Tensor:
    """Dispatch the native forward, retaining an FP32 CPU reference implementation.

    Args:
        packed: Gate/up projection in contiguous packed layout.
        limit: Clipping limit; nonpositive values disable clipping.
    """
    if packed.device.type == "npu":
        # The optional NPU dependency must not be required by model imports on CPU.
        import torch_npu  # pylint: disable=import-outside-toplevel

        if limit > 0:
            return torch_npu.npu_clipped_swiglu(
                packed, dim=-1, alpha=1.0, limit=limit, bias=0.0, interleaved=False,
            )
        return torch_npu.npu_swiglu(packed, dim=-1)
    gate, up = packed.float().chunk(2, dim=-1)
    if limit > 0:
        gate, up = gate.clamp(max=limit), up.clamp(min=-limit, max=limit)
    return (functional.silu(gate) * up).to(packed.dtype)


def _backward_chunk(gradient: torch.Tensor, packed: torch.Tensor, limit: float) -> torch.Tensor:
    """Apply SwiGLU's chain rule in FP32, including PyTorch's clamp boundary rule.

    Args:
        gradient: Activation derivative for the current token tile.
        packed: Original packed projection for the tile.
        limit: Clipping limit used by the corresponding forward.
    """
    gate, up = packed.float().chunk(2, dim=-1)
    clipped_gate = gate.clamp(max=limit) if limit > 0 else gate
    clipped_up = up.clamp(min=-limit, max=limit) if limit > 0 else up
    if packed.device.type == "npu":
        # ClippedSwiglu has no registered backward on the supported runtime.
        import torch_npu  # pylint: disable=import-outside-toplevel

        clipped = torch.cat((clipped_gate, clipped_up), dim=-1)
        derivative = torch_npu.npu_swiglu_backward(gradient.float().contiguous(), clipped, dim=-1)
        gate_grad, up_grad = derivative.chunk(2, dim=-1)
    else:
        gate_grad = torch.ops.aten.silu_backward(gradient.float() * clipped_up, clipped_gate)
        up_grad = gradient.float() * functional.silu(clipped_gate)
    if limit > 0:
        gate_grad = gate_grad.masked_fill(gate > limit, 0)
        up_grad = up_grad.masked_fill((up < -limit) | (up > limit), 0)
    return torch.cat((gate_grad, up_grad), dim=-1).to(packed.dtype)


class _PackedSwiGLU(torch.autograd.Function):
    """Keep only the packed projection and bound transient FP32 backward storage."""

    @staticmethod
    def forward(ctx: Any, packed: torch.Tensor, limit: float, chunk_size: int) -> torch.Tensor:
        """Save the original projection for the clamp mask and SwiGLU derivative.

        Args:
            ctx: Autograd state for the first-order backward.
            packed: Original gate/up projection.
            limit: Model activation clipping limit.
            chunk_size: Maximum rows in a backward tile.
        """
        ctx.save_for_backward(packed)
        ctx.limit = limit
        ctx.chunk_size = chunk_size
        if packed.numel() == 0:
            return packed.new_empty((packed.shape[0], packed.shape[1] // 2))
        return _forward(packed, limit)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Recompute bounded activation tiles; no expert communication is replayed.

        Args:
            ctx: Saved projection and activation configuration.
            gradient: Derivative with respect to the activation output.
        """
        (packed,) = ctx.saved_tensors
        derivative = torch.empty_like(packed)
        for start in range(0, packed.shape[0], ctx.chunk_size):
            end = min(start + ctx.chunk_size, packed.shape[0])
            derivative[start:end].copy_(_backward_chunk(gradient[start:end], packed[start:end], ctx.limit))
        return derivative, None, None


def deepseek_v41_packed_swiglu(
        packed: torch.Tensor, limit: float, backward_chunk_size: int = 4096,
) -> torch.Tensor:
    """Compute model-preserving packed SwiGLU without full-size FP32 activations.

    Args:
        packed: Contiguous two-dimensional gate/up projection in FP32 or BF16.
            Gate occupies the first half of the final dimension.
        limit: Finite gate upper bound and symmetric up bound. Nonpositive values
            disable clipping, matching the published model's reference path.
        backward_chunk_size: Maximum rows of temporary FP32 derivative storage.

    Returns:
        The activation in the projection dtype. Native arithmetic accumulates in
        FP32 and rounds after the activation. CPU execution is the reference path.

    Raises:
        ValueError: If the projection or chunk configuration is unsupported.

    Note:
        This adapter supports first-order training gradients. It deliberately
        rejects double backward through ``once_differentiable``.
    """
    if packed.ndim != 2 or packed.shape[-1] % 2 or packed.shape[-1] == 0 or not packed.is_contiguous():
        raise ValueError("Packed SwiGLU requires contiguous [rows, 2 * intermediate] projections")
    if packed.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Packed SwiGLU supports BF16 and FP32 projections")
    if isinstance(backward_chunk_size, bool) or not isinstance(backward_chunk_size, int) or backward_chunk_size <= 0:
        raise ValueError("backward_chunk_size must be a positive integer")
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit):
        raise ValueError("SwiGLU limit must be a finite real number")
    return _PackedSwiGLU.apply(packed, float(limit), backward_chunk_size)
