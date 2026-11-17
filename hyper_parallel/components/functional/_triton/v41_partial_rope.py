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
"""Fused interleaved RoPE for contiguous BF16 V4.1 attention tensors on Ascend."""

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


@triton.jit
def _partial_rope_kernel(
        source, destination, cosine, sine, tokens: tl.constexpr, backward: tl.constexpr,
):
    head = tl.arange(0, 64)
    lane = tl.arange(0, 64)
    odd_lane = lane[None, None, :] % 2 == 1
    prefix = tl.arange(0, 512)
    partner_index = tl.broadcast_to((lane ^ 1)[None, None, :], (1, 64, 64))
    # Each token's 64 heads share one table row. Load that row once per tile.
    for tile in range(tl.program_id(0), tokens, tl.num_programs(0)):
        token = tile + tl.arange(0, 1)
        valid = token < tokens
        table_offset = token[:, None] * 64 + lane[None, :]
        cos_value = tl.load(cosine + table_offset, valid[:, None], other=0).to(tl.float32)
        sin_value = tl.load(sine + table_offset, valid[:, None], other=0).to(tl.float32)
        row_base = (token[:, None, None] * 64 + head[None, :, None]) * 512
        rotary_offset = row_base + 448 + lane[None, None, :]
        value = tl.load(source + rotary_offset, valid[:, None, None], other=0).to(tl.float32)
        partner = tl.gather(value, partner_index, axis=2)
        if backward:
            # Match the two BF16 roundings before adding the two gradient contributions.
            direct = (value * cos_value[:, None, :]).to(tl.bfloat16).to(tl.float32)
            crossed = (partner * sin_value[:, None, :]).to(tl.bfloat16).to(tl.float32)
            rotary = direct + tl.where(odd_lane, -crossed, crossed)
        else:
            rotary = value * cos_value[:, None, :] + tl.where(odd_lane, partner, -partner) * sin_value[:, None, :]
        tl.store(destination + rotary_offset, rotary, valid[:, None, None])
        prefix_offset = row_base + prefix[None, None, :]
        prefix_valid = valid[:, None, None] & (prefix[None, None, :] < 448)
        prefix_value = tl.load(source + prefix_offset, prefix_valid, other=0)
        tl.store(destination + prefix_offset, prefix_value, prefix_valid)


def _run_partial_rope(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, backward: bool) -> torch.Tensor:
    output = torch.empty_like(value)
    _partial_rope_kernel[(48,)](
        value, output, cos, sin, value.shape[1], backward, enable_fp_fusion=False,
    )
    return output


class _V41PartialRope(torch.autograd.Function):
    """Apply RoPE with a separate output buffer and first-order input gradients."""

    @staticmethod
    def forward(ctx, value, cos, sin):
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
        ctx.save_for_backward(cos, sin)
        return _run_partial_rope(value, cos, sin, backward=False)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors
        return _run_partial_rope(grad_output.contiguous(), cos, sin, backward=True), None, None


def fused_v41_partial_rope(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the trailing 64 channels of a contiguous [1, S, 64, 512] BF16 tensor."""
    return _V41PartialRope.apply(value, cos, sin)
