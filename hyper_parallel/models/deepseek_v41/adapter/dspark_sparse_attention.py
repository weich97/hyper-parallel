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

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch


@lru_cache(maxsize=8)
def _sparse_draft_metadata(batch, main_length, block, window, start, document_positions, device,
                           anchor_positions=None, include_sink=False):
    """Keep each query's valid main and draft indices before its padding suffix."""
    anchors = len(document_positions)
    absolute = tuple(range(start, start + anchors)) if anchor_positions is None else anchor_positions
    if (batch <= 0 or anchors == 0 or block <= 0 or window <= 0 or window + block + int(include_sink) > 1024
            or len(absolute) != anchors or any(index < 0 or index >= main_length for index in absolute)
            or any(position < 0 or position > index
                   for index, position in zip(absolute, document_positions, strict=True))):
        raise ValueError("Invalid DSpark sparse window geometry")
    # Vectorized Host metadata avoids constructing millions of Python integers
    # for each query chunk. It never reads or synchronizes a device tensor.
    columns = torch.arange(1024, dtype=torch.int32).view(1, 1, -1)
    anchor_indices = torch.arange(anchors, dtype=torch.int32).view(1, -1, 1)
    main_positions = torch.tensor(absolute, dtype=torch.int32).view(1, -1, 1)
    batch_indices = torch.arange(batch, dtype=torch.int32).view(-1, 1, 1)
    counts = (torch.tensor(document_positions, dtype=torch.int32) + 1).clamp_max(window).view(1, -1, 1)
    main_indices = batch_indices * main_length + main_positions - counts + 1 + columns
    draft_indices = batch * main_length + (batch_indices * anchors + anchor_indices) * block + columns - counts
    selected = torch.where(columns < counts, main_indices, draft_indices)
    selected = selected.masked_fill(columns >= counts + block, -1)
    if include_sink:
        sink_index = batch * main_length + batch * anchors * block
        selected = torch.where(columns == counts + block, sink_index, selected)
    return selected.repeat_interleave(block, dim=1).reshape(-1, 1, 1024).to(device)


@lru_cache(maxsize=8)
def _sink_key_components(width, dtype):
    """Represent the inverse scale with three mantissas, computed once on Host."""
    inverse_scale = width ** 0.5
    residual, components = inverse_scale, []
    for _ in range(3):
        component = torch.tensor(residual, dtype=dtype, device="cpu").item()
        components.append(component)
        residual -= component
    return tuple(components), sum(components) / inverse_scale


def _sink_rotary_operands(query, bank, sink):
    """Encode a zero-valued virtual key whose scaled logit is the learned sink."""
    components, factor = _sink_key_components(query.shape[-1], query.dtype)
    query_rope = query.new_zeros((*query.shape[:-1], 64))
    key_rope = bank.new_zeros((*bank.shape[:-1], 64))
    query_rope[..., :len(components)] = sink.to(query.dtype)[None, :, None]
    for index, component in enumerate(components):
        key_rope[-1, :, index] = component
    return query_rope, key_rope, factor


class _SparseSinkAttention(torch.autograd.Function):
    """Preserve the learned sink and combine shared K/V gradients before casting."""

    @staticmethod
    def forward(ctx: Any, query: torch.Tensor, bank: torch.Tensor,
                indices: torch.Tensor, sink: torch.Tensor) -> torch.Tensor:
        """Include the learned per-head sink in the native softmax.

        Args:
            ctx: Autograd context retaining native backward inputs.
            query: Rotated BF16 queries in TND layout.
            bank: Shared main and draft KV rows.
            indices: Valid KV indices, one virtual sink index, then negative padding.
            sink: Per-head FP32 sink logits.

        Returns:
            BF16 attention outputs in the query layout.
        """
        # These runtime dependencies must stay optional on CPU-only hosts.
        import torch_npu  # pylint: disable=import-outside-toplevel

        bank = torch.cat((bank, bank.new_zeros(1, *bank.shape[1:])), dim=0)
        qlen = torch.tensor([query.shape[0]], dtype=torch.int32, device=query.device)
        klen = torch.tensor([bank.shape[0]], dtype=torch.int32, device=query.device)
        # A3 does not support the native sinks argument. An auxiliary rotary
        # operand puts the scalar sink inside softmax without another BF16
        # output cast. The real DSpark rotary coordinates remain unchanged.
        qrope, krope, factor = _sink_rotary_operands(query, bank, sink)
        output, maximum, denominator = torch_npu.npu_sparse_flash_attention(
            query, bank, bank, indices, query.shape[-1] ** -0.5,
            actual_seq_lengths_query=qlen, actual_seq_lengths_kv=klen,
            sparse_block_size=1, layout_query="TND", layout_kv="TND", sparse_mode=0,
            attention_mode=2, return_softmax_lse=True, sinks=None, query_rope=qrope, key_rope=krope,
        )
        row_max, row_sum = maximum.reshape(query.shape[:2]), denominator.reshape(query.shape[:2])
        sink_probability = torch.exp(sink[None, :] * factor - row_max) / row_sum
        ctx.sink_factor = factor
        ctx.save_for_backward(query, bank, indices, output, maximum, denominator,
                              sink_probability, qlen, klen, qrope, krope)
        return output

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]:
        """Return query, shared-KV and sink derivatives with one KV cast.

        Args:
            ctx: Autograd context from the sparse forward operation.
            gradient: Upstream attention-output derivative.

        Returns:
            Query, shared-KV and sink derivatives; indices have no derivative.
        """
        # The repaired custom operator is required; no fallback changes its semantics.
        import cann_ops_transformer as ops  # pylint: disable=import-outside-toplevel

        query, bank, indices, output, maximum, denominator, probability, qlen, klen, qrope, krope = ctx.saved_tensors
        dq, dkv, dv, _, _, _ = ops.sparse_flash_attention_grad(
            query, bank, None, indices, gradient.contiguous(), output, maximum, denominator,
            sinks=None, scale_value=query.shape[-1] ** -0.5, sparse_block_size=1,
            query_rope=qrope, key_rope=krope, actual_seq_qlen=qlen, actual_seq_kvlen=klen,
            layout="TND", sparse_mode=0, attention_mode=2,
        )
        if dv.numel() != 0:
            raise RuntimeError("DSpark shared-KV backward must return one combined KV derivative")
        accumulation_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
        product = (gradient.to(accumulation_dtype) * output.to(accumulation_dtype)).sum(-1)
        dsink = -(product * probability).sum(0) * ctx.sink_factor
        return dq, dkv[:-1], None, dsink


def dspark_sparse_attention(query: torch.Tensor, main_kv: torch.Tensor, draft_kv: torch.Tensor,
                            sink: torch.Tensor, window: int, start: int,
                            document_positions: tuple[int, ...],
                            anchor_positions: tuple[int, ...] | None = None) -> torch.Tensor:
    """Attend to the exact DSpark windows without duplicating overlapping KV.

    Args:
        query: Rotated BF16 ``[batch, anchors, block, heads, 512]`` queries.
        main_kv: Rotated ``[batch, main_sequence, 512]`` main KV.
        draft_kv: Rotated ``[batch, anchors, block, 512]`` draft KV.
        sink: Per-head FP32 sink logits, following the reference's dtype conversion.
        window: Main-context window size, excluding the draft block.
        start: First absolute anchor position of this contiguous chunk.
        document_positions: Document-relative positions of the chunk's anchors.
        anchor_positions: Original main positions when the draft anchors are sampled.

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
                                     document_positions, query.device, anchor_positions, include_sink=True)
    bank = torch.cat((main_kv.reshape(-1, width), draft_kv.reshape(-1, width)), dim=0)
    bank = bank.unsqueeze(1).to(query.dtype).contiguous()
    output = _SparseSinkAttention.apply(query.reshape(-1, heads, width).contiguous(), bank, indices, sink)
    return output.view_as(query)
