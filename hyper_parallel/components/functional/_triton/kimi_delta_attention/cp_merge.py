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
# Portions copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li.
# FLA's MIT-licensed CP merge recurrence is adapted to reuse local M when gathering G.
# See FLA_LICENSE in this directory.
"""FLA-style one-kernel prefix/suffix with separate transfer storage and fused transpose."""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["count", "rank"])
def kda_cached_affine_merge_kernel(
    output: tl.tensor, states: tl.tensor, matrices: tl.tensor, count: tl.tensor, rank: tl.tensor,
    state_rank_stride: tl.constexpr, state_head_stride: tl.constexpr, state_row_stride: tl.constexpr,
    matrix_rank_stride: tl.constexpr, matrix_head_stride: tl.constexpr, matrix_row_stride: tl.constexpr,
    forward: tl.constexpr, block_v: tl.constexpr,
) -> None:
    """Evaluate a zero-entry prefix/suffix state, without materializing a composed M.

    Local transfer matrices are produced before this kernel. Each iteration
    evaluates M @ state + S (or M.T @ gradient + G) in chronological order.

    Args:
        output: Contiguous FP32 recurrent-state output pointer.
        states: Chronological affine offsets [P,B,H,128,128].
        matrices: Matching chronological FP32 transfer matrices.
        count: Number of chronological offsets to merge.
        rank: Chronological rank within the CP group.
        state_rank_stride: Element stride between source-rank state slabs.
        state_head_stride: Element stride between flattened batch/head state slabs.
        state_row_stride: Element stride between key rows in state storage.
        matrix_rank_stride: Element stride between source-rank transfer slabs.
        matrix_head_stride: Element stride between flattened batch/head transfer slabs.
        matrix_row_stride: Element stride between rows in M.
        forward: Compile-time selection of prefix or transposed suffix recurrence.
        block_v: Value-column tile width.
    """
    column, head = tl.program_id(0), tl.program_id(1)
    rows = tl.arange(0, 128)
    cols = column*block_v + tl.arange(0, block_v)
    value = tl.zeros((128, block_v), tl.float32)
    for index in range(count):
        source = index if forward else rank+count-index
        offset = states + source*state_rank_stride + head*state_head_stride
        state = tl.load(offset + rows[:, None]*state_row_stride + cols[None, :])
        offset_m = matrices + source*matrix_rank_stride + head*matrix_head_stride
        if forward:
            offsets_m = rows[:, None]*matrix_row_stride + rows[None, :]
        else:
            offsets_m = rows[None, :]*matrix_row_stride + rows[:, None]
        matrix = tl.load(offset_m+offsets_m)
        value = tl.dot(matrix.to(tl.float32), value, input_precision="ieee") + state.to(tl.float32)
    tl.store(output+head*16384+rows[:, None]*128+cols[None, :], value)


@triton.jit(do_not_specialize=["count", "start", "prefix_count"])
def kda_affine_scan_kernel(
    output: tl.tensor, prefix_output: tl.tensor, maps: tl.tensor, initial: tl.tensor,
    count: tl.tensor, start: tl.tensor, prefix_count: tl.tensor,
    rank_stride: tl.constexpr, head_stride: tl.constexpr, row_stride: tl.constexpr,
    forward: tl.constexpr, save_prefix: tl.constexpr, block_v: tl.constexpr,
) -> None:
    """Scan packed summaries from a nonzero state, optionally saving an owner prefix.

    Args:
        output: Contiguous FP32 final state pointer.
        prefix_output: Optional contiguous FP32 intermediate-state pointer.
        maps: Packed offsets and M, shaped [P,B,H,128,256].
        initial: Contiguous incoming state shaped [B,H,128,128].
        count: Number of ordered affine applications.
        start: First source index in the packed summaries.
        prefix_count: Number of applications before saving the intermediate state.
        rank_stride: Element stride between source ranks.
        head_stride: Element stride between flattened batch/head slabs.
        row_stride: Element stride between key rows.
        forward: Select forward order and M instead of reverse order and M.T.
        save_prefix: Write an intermediate state during the scan.
        block_v: Value-column tile width.
    """
    column, head = tl.program_id(0), tl.program_id(1)
    rows = tl.arange(0, 128)
    cols = column * block_v + tl.arange(0, block_v)
    output_offsets = head * 16384 + rows[:, None] * 128 + cols[None, :]
    value = tl.load(initial + output_offsets).to(tl.float32)
    for index in range(count):
        source = start + index if forward else start - index
        base = maps + source * rank_stride + head * head_stride
        offset = tl.load(base + rows[:, None] * row_stride + cols[None, :])
        if forward:
            matrix_offsets = rows[:, None] * row_stride + rows[None, :]
        else:
            matrix_offsets = rows[None, :] * row_stride + rows[:, None]
        matrix = tl.load(base + 128 + matrix_offsets).to(tl.float32)
        value = tl.dot(matrix, value, input_precision="ieee") + offset.to(tl.float32)
        if save_prefix:
            if index + 1 == prefix_count:
                tl.store(prefix_output + output_offsets, value)
    tl.store(output + output_offsets, value)
