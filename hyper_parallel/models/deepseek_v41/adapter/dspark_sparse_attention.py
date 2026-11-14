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
"""Selected-KV DSpark attention using an explicitly provisioned native backward.

The A3 backward package must support a valid index prefix followed by -1
padding. Older implementations infer the selected count from KV capacity,
which is incorrect for DSpark windows. The experiment/runtime owner must
validate and provision the repaired operator before selecting this backend.
"""

from functools import lru_cache

import torch


@lru_cache(maxsize=8)
def _sparse_draft_metadata(batch, main_length, block, window, start, document_positions, device):
    """Keep each query's valid main and draft indices before its padding suffix."""
    anchors = len(document_positions)
    if (batch <= 0 or anchors == 0 or block <= 0 or window <= 0 or window + block > 1024
            or start < 0 or start + anchors > main_length
            or any(position < 0 or position > start + index
                   for index, position in enumerate(document_positions))):
        raise ValueError("Invalid DSpark sparse window geometry")
    # Vectorized Host metadata avoids constructing millions of Python integers
    # for each query chunk. It never reads or synchronizes a device tensor.
    columns = torch.arange(1024, dtype=torch.int32).view(1, 1, -1)
    anchor_indices = torch.arange(anchors, dtype=torch.int32).view(1, -1, 1)
    batch_indices = torch.arange(batch, dtype=torch.int32).view(-1, 1, 1)
    counts = (torch.tensor(document_positions, dtype=torch.int32) + 1).clamp_max(window).view(1, -1, 1)
    main_indices = batch_indices * main_length + start + anchor_indices - counts + 1 + columns
    draft_indices = batch * main_length + (batch_indices * anchors + anchor_indices) * block + columns - counts
    selected = torch.where(columns < counts, main_indices, draft_indices)
    selected = selected.masked_fill(columns >= counts + block, -1)
    return selected.repeat_interleave(block, dim=1).reshape(-1, 1, 1024).to(device)


class _SparseSinkAttention(torch.autograd.Function):
    """Preserve the learned sink and combine shared K/V gradients before casting."""

    @staticmethod
    def forward(ctx, query, bank, indices, sink):
        # These runtime dependencies must stay optional on CPU-only hosts.
        import torch_npu  # pylint: disable=import-outside-toplevel

        qlen = torch.tensor([query.shape[0]], dtype=torch.int32, device=query.device)
        klen = torch.tensor([bank.shape[0]], dtype=torch.int32, device=query.device)
        # A3's selected-KV kernel requires the separate rotary operand. The
        # real DSpark rotary coordinates are already inside query and bank.
        qrope = query.new_zeros((*query.shape[:-1], 64))
        krope = bank.new_zeros((*bank.shape[:-1], 64))
        value, maximum, denominator = torch_npu.npu_sparse_flash_attention(
            query, bank, bank, indices, query.shape[-1] ** -0.5,
            actual_seq_lengths_query=qlen, actual_seq_lengths_kv=klen,
            sparse_block_size=1, layout_query="TND", layout_kv="TND", sparse_mode=0,
            attention_mode=2, return_softmax_lse=True, sinks=None, query_rope=qrope, key_rope=krope,
        )
        row_max, row_sum = maximum.reshape(query.shape[:2]), denominator.reshape(query.shape[:2])
        total_max = torch.maximum(row_max, sink[None, :])
        selected_sum = row_sum * torch.exp(row_max - total_max)
        total_sum = selected_sum + torch.exp(sink[None, :] - total_max)
        alpha = selected_sum / total_sum
        output = (value.float() * alpha.unsqueeze(-1)).to(query.dtype)
        ctx.save_for_backward(query, bank, indices, output, total_max.reshape_as(maximum).contiguous(),
                              total_sum.reshape_as(denominator).contiguous(), alpha, qlen, klen, qrope, krope)
        return output

    @staticmethod
    def backward(ctx, gradient):
        # The repaired custom operator is required; no fallback changes its semantics.
        import cann_ops_transformer as ops  # pylint: disable=import-outside-toplevel

        query, bank, indices, output, maximum, denominator, alpha, qlen, klen, qrope, krope = ctx.saved_tensors
        dq, dkv, dv, _, _, _ = ops.sparse_flash_attention_grad(
            query, bank, None, indices, gradient.contiguous(), output, maximum, denominator,
            sinks=None, scale_value=query.shape[-1] ** -0.5, sparse_block_size=1,
            query_rope=qrope, key_rope=krope, actual_seq_qlen=qlen, actual_seq_kvlen=klen,
            layout="TND", sparse_mode=0, attention_mode=2,
        )
        if dv.numel() != 0:
            raise RuntimeError("DSpark shared-KV backward must return one combined KV derivative")
        dsink = -((gradient.float() * output.float()).sum(-1) * (1 - alpha)).sum(0)
        return dq, dkv, None, dsink


def dspark_sparse_attention(query: torch.Tensor, main_kv: torch.Tensor, draft_kv: torch.Tensor,
                            sink: torch.Tensor, window: int, start: int,
                            document_positions: tuple[int, ...]) -> torch.Tensor:
    """Attend to the exact DSpark windows without duplicating overlapping KV.

    Args:
        query: Rotated BF16 ``[batch, anchors, block, heads, 512]`` queries.
        main_kv: Rotated ``[batch, main_sequence, 512]`` main KV.
        draft_kv: Rotated ``[batch, anchors, block, 512]`` draft KV.
        sink: Per-head FP32 sink logits, following the reference's dtype conversion.
        window: Main-context window size, excluding the draft block.
        start: First absolute anchor position of this contiguous chunk.
        document_positions: Document-relative positions of the chunk's anchors.

    Returns:
        Attention outputs with the same shape and dtype as query.
    """
    batch, anchors, block, heads, width = query.shape
    if query.device.type != "npu" or query.dtype != torch.bfloat16 or width != 512 or heads != 64:
        raise ValueError("DSpark sparse_v2 requires BF16 NPU inputs with 64 heads and width 512")
    if (main_kv.shape[0] != batch or main_kv.shape[-1] != width
            or draft_kv.shape != (batch, anchors, block, width)
            or len(document_positions) != anchors):
        raise ValueError("DSpark sparse KV and document metadata must match the query chunk")
    indices = _sparse_draft_metadata(batch, main_kv.shape[1], block, window, start,
                                     document_positions, query.device)
    bank = torch.cat((main_kv.reshape(-1, width), draft_kv.reshape(-1, width)), dim=0)
    bank = bank.unsqueeze(1).to(query.dtype).contiguous()
    output = _SparseSinkAttention.apply(query.reshape(-1, heads, width).contiguous(), bank, indices, sink)
    return output.view_as(query)
