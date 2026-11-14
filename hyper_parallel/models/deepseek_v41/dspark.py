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
"""Trainable DeepSeek-V4.1 DSpark (``mtp.*``) speculative-draft stack.

DSpark is V4.1's replacement for the DeepSeek-V3 MTP module: a drafter of
``num_nextn_predict_layers`` Transformer stages that predicts
``dspark_block_size`` future tokens in one parallel pass, plus a lightweight
Markov head that models dependencies among the draft tokens and an FP32
confidence head that predicts per-position acceptance probabilities
(DeepSeek-V4.1-Flash technical report, section 2.4.3).

Training contract implemented here, following the report:

* DSpark is trained **without propagating gradients into the backbone** —
  every backbone hidden handed to this module must already be detached by the
  caller; ``main_proj`` consumes the concatenation of the
  ``dspark_target_layer_ids`` backbone hiddens.
* One forward computes base logits for all ``block_size`` draft offsets of
  every sequence position (teacher forcing). The Markov head adds a low-rank
  bias conditioned on the previous ground-truth token of each draft offset.
* With the main context ending at ``t``, the first draft input is ``x[t+1]``;
  draft offset ``j`` predicts ``x[t+2+j]``. The objective is token-weighted
  cross-entropy and the confidence objective is a BCE against greedy-hit events.
  The report does not disclose the production loss coefficients, so both are
  configuration values, mirroring the Indexer-KL precedent.

The published FP4/FP8 ``.scale`` companions are not materialized: this adapter
trains in BF16. Attention preserves the released main-plus-draft context,
adjacent-pair rotary embedding and output inverse rotation. Feature-tap,
shared-noise and teacher-forcing integration require their separate model-level
acceptance; matching attention alone does not establish whole-model fidelity.
"""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from functools import lru_cache
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from torch.nn import functional  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional.sinkhorn import sinkhorn_knopps
from hyper_parallel.core.activation_memory.recompute_state import create_recompute_contexts
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe import DeepseekV41MegaMoeExperts

logger = logging.getLogger(__name__)

# Draft positions multiply the backbone token count. Bound FP32 mHC
# intermediates by independent rows without changing a row's reduction order.
_MHC_ROW_CHUNK = 1024


def _flash_attention_v3(query, key_value, sink, query_lengths, kv_lengths, scale):
    """Run independent noncausal draft blocks with the native scalar-sink backward."""
    if query.device.type != "npu":
        raise ValueError("DSpark flash_v3 attention requires an NPU")
    # CPU-only model construction must not import the optional NPU runtime.
    import torch_npu  # pylint: disable=import-outside-toplevel

    if not hasattr(torch_npu, "npu_fusion_attention_v3"):
        raise RuntimeError("DSpark flash_v3 attention requires npu_fusion_attention_v3")
    return torch_npu.npu_fusion_attention_v3(
        query, key_value, key_value, query.shape[1], "TND", sink=sink, atten_mask=None,
        actual_seq_qlen=query_lengths, actual_seq_kvlen=kv_lengths,
        scale=scale, keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647, sparse_mode=0,
    )[0]


def _checkpoint(function, *args):
    """Activation-checkpoint on device; run directly on CPU.

    Non-reentrant checkpoint replays the device autocast state when it
    recomputes, which initializes the accelerator. On a CPU-only host that
    has ``torch_npu`` installed (the CI unit-test image) that lazy
    initialization raises. Checkpointing is a device-memory optimization
    with no benefit on CPU, so the function runs directly there; the
    autograd result is identical either way.
    """
    on_device = any(isinstance(arg, torch.Tensor) and arg.device.type != "cpu"
                    for arg in args)
    if on_device:
        return torch.utils.checkpoint.checkpoint(function, *args, use_reentrant=False)
    return function(*args)


def _checkpoint_stage(function, *args, offload_inputs=False, **kwargs):
    """Replay a complete stage on all ranks, including its expert collectives."""
    device = next((arg.device for arg in args if isinstance(arg, torch.Tensor) and arg.device.type != "cpu"), None)
    if device is None:
        return function(*args, **kwargs)
    # Every rank must execute the complete collective sequence during replay.
    # The outer hook moves only checkpoint inputs; the inner checkpoint hook
    # discards intermediate activations instead of copying them to Host.
    storage = (torch.autograd.graph.save_on_cpu(pin_memory=False, device_type=device.type)
               if offload_inputs else nullcontext())
    with storage, torch.utils.checkpoint.set_checkpoint_early_stop(False):
        return torch.utils.checkpoint.checkpoint(
            function, *args, **kwargs, use_reentrant=False, context_fn=create_recompute_contexts,
        )


class _StreamCollapse(torch.autograd.Function):
    """Bound both passes without per-chunk full-input slice-gradient allocations."""

    @staticmethod
    def forward(ctx: Any, hidden_streams: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """Keep the release's FP32 stream reduction and BF16 output boundary.

        Args:
            ctx: Autograd context retaining input operands.
            hidden_streams: Residual streams ending in stream and hidden dimensions.
            pre_mix: Per-row stream weights.
        """
        streams, hidden = hidden_streams.shape[-2:]
        rows = hidden_streams.reshape(-1, streams, hidden)
        weights = pre_mix.reshape(-1, streams)
        output = rows.new_empty(rows.shape[0], hidden)
        for start in range(0, rows.shape[0], _MHC_ROW_CHUNK):
            stop = start + _MHC_ROW_CHUNK
            output[start:stop] = (weights[start:stop, :, None] * rows[start:stop].float()).sum(-2)
        ctx.save_for_backward(hidden_streams, pre_mix)
        return output.view(*hidden_streams.shape[:-2], hidden)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate each destination gradient once and fill independent rows.

        Args:
            ctx: Autograd context with saved streams and weights.
            grad_output: Derivative of the collapsed hidden rows.
        """
        hidden_streams, pre_mix = ctx.saved_tensors
        streams, hidden = hidden_streams.shape[-2:]
        rows = hidden_streams.reshape(-1, streams, hidden)
        weights = pre_mix.reshape(-1, streams)
        grad = grad_output.reshape(-1, hidden)
        grad_rows, grad_weights = torch.empty_like(rows), torch.empty_like(weights)
        for start in range(0, rows.shape[0], _MHC_ROW_CHUNK):
            stop = start + _MHC_ROW_CHUNK
            upstream = grad[start:stop].float().unsqueeze(-2)
            grad_rows[start:stop] = upstream * weights[start:stop, :, None]
            grad_weights[start:stop] = (upstream * rows[start:stop].float()).sum(-1)
        return grad_rows.view_as(hidden_streams), grad_weights.view_as(pre_mix)


def _collapse_streams(hidden_streams, pre_mix):
    """Collapse independent residual rows without a full draft-sized FP32 buffer."""
    return _StreamCollapse.apply(hidden_streams, pre_mix)


class _NormalizedHcProjection(torch.autograd.Function):
    """Stream the weightless RMS/projection boundary in forward and backward."""

    @staticmethod
    def forward(ctx: Any, hidden_streams: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Project first, then apply the exact per-row reciprocal RMS.

        Args:
            ctx: Autograd context retaining operands and epsilon.
            hidden_streams: Residual streams to normalize and project.
            weight: Shared projection matrix.
            epsilon: Stability constant in the row normalization.
        """
        rows = hidden_streams.flatten(2).reshape(-1, weight.shape[-1])
        output = torch.empty(rows.shape[0], weight.shape[0], device=rows.device, dtype=torch.float32)
        for start in range(0, rows.shape[0], _MHC_ROW_CHUNK):
            stop = start + _MHC_ROW_CHUNK
            values = rows[start:stop].float()
            inverse_rms = torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon)
            output[start:stop] = functional.linear(values, weight) * inverse_rms  # pylint: disable=not-callable
        ctx.save_for_backward(hidden_streams, weight)
        ctx.epsilon = epsilon
        return output.view(*hidden_streams.shape[:2], weight.shape[0])

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Recompute small FP32 tiles and accumulate the shared projection gradient.

        Args:
            ctx: Autograd context with saved operands and epsilon.
            grad_output: Derivative of the normalized projection.
        """
        hidden_streams, weight = ctx.saved_tensors
        rows = hidden_streams.flatten(2).reshape(-1, weight.shape[-1])
        grad = grad_output.reshape(-1, weight.shape[0])
        grad_rows, grad_weight = torch.empty_like(rows), torch.zeros_like(weight)
        for start in range(0, rows.shape[0], _MHC_ROW_CHUNK):
            stop = start + _MHC_ROW_CHUNK
            values, upstream = rows[start:stop].float(), grad[start:stop].float()
            inverse_rms = torch.rsqrt(values.square().mean(-1, keepdim=True) + ctx.epsilon)
            projected = functional.linear(values, weight)  # pylint: disable=not-callable
            scaled = upstream * inverse_rms
            grad_weight.add_(scaled.T @ values)
            rms_gradient = (upstream * projected).sum(-1, keepdim=True) * inverse_rms.pow(3) / values.shape[-1]
            grad_rows[start:stop] = scaled @ weight - values * rms_gradient
        return grad_rows.view_as(hidden_streams), grad_weight, None


def _apply_rope(tensor, cos, sin, *, inverse=False):
    """Rotate adjacent pairs in FP32, preserving the release's BF16 cast boundary."""
    pairs = tensor.float().unflatten(-1, (-1, 2))
    if inverse:
        sin = -sin
    real = pairs[..., 0] * cos - pairs[..., 1] * sin
    imaginary = pairs[..., 0] * sin + pairs[..., 1] * cos
    return torch.stack((real, imaginary), dim=-1).flatten(-2).to(tensor.dtype)


@lru_cache(maxsize=64)
def _document_geometry(sequence, boundaries):
    """Return owned document-relative positions and exclusive ends without device synchronization."""
    boundaries = (0, sequence) if boundaries is None else tuple(boundaries)
    if (len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != sequence
            or any(end <= start for start, end in zip(boundaries[:-1], boundaries[1:], strict=True))):
        raise ValueError("DSpark document boundaries must strictly partition the complete sequence")
    positions = tuple(position for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
                      for position in range(end - start))
    ends = tuple(end for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
                 for _ in range(end - start))
    return positions, ends


@lru_cache(maxsize=64)
def _draft_pack_metadata(batch, block, window, positions, device):
    """Build static TND metadata without synchronizing device-resident masks."""
    row_width = window + block
    lengths = [min(window, position + 1) + block for position in positions] * batch
    indices = [row * row_width + offset
               for row, length in enumerate(lengths)
               for offset in range(row_width - length, row_width)]
    query_lengths = torch.arange(1, batch * len(positions) + 1, dtype=torch.int64) * block
    kv_lengths = torch.tensor(lengths, dtype=torch.int64).cumsum(0)
    return torch.tensor(indices, device=device, dtype=torch.long), query_lengths, kv_lengths


class DSparkTokenEmbedding(nn.Embedding):
    """Share a frozen noise-token row while FSDP exposes the backbone embedding."""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None,
                 *, noise_token_id: int) -> None:
        """Create the ordinary embedding and validate the shared noise-token identity."""
        if not 0 <= noise_token_id < num_embeddings:
            raise ValueError("DSpark noise token must be inside the backbone vocabulary")
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.noise_token_id = noise_token_id
        self.cached_noise_embedding = None

    @torch.no_grad()
    def _read_noise_embedding(self):
        """Gather only the selected row when TP shards the embedding vocabulary or width."""
        indices = torch.tensor([self.noise_token_id], dtype=torch.long, device=self.weight.device)
        if isinstance(self.weight, DTensor):
            mesh = self.weight.device_mesh
            indices = DTensor.from_local(indices, mesh, [Replicate()] * mesh.ndim)
        row = functional.embedding(indices, self.weight)
        if isinstance(row, DTensor):
            if row.is_partial():
                row = row.reduce_partial()
            row = row.full_tensor()
        return row.reshape(self.embedding_dim).detach()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
        """Return ordinary token embeddings and refresh the frozen noise row.

        Args:
            input: Token IDs consumed by the backbone embedding.

        Returns:
            The ordinary differentiable token embeddings.
        """
        self.cached_noise_embedding = self._read_noise_embedding()
        return super().forward(input)


class DSparkMarkovHead(nn.Module):
    """Low-rank 2-gram head biasing draft logits with the previous token."""

    def __init__(self, vocab_size: int, markov_rank: int) -> None:
        """Create the tied-rank embedding and output projections."""
        super().__init__()
        self.embed = nn.Embedding(vocab_size, markov_rank)
        self.head = nn.Linear(markov_rank, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the vocabulary bias and the rank-space embedding.

        Args:
            token_ids: Ground-truth previous token for each draft offset.

        Returns:
            Vocabulary bias and low-rank token features.
        """
        embedded = self.embed(token_ids)
        return self.head(embedded), embedded


class DSparkConfidenceHead(nn.Module):
    """FP32 acceptance-probability head over hidden and Markov features."""

    def __init__(self, input_dim: int) -> None:
        """Create the FP32 scalar projection."""
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        """Return per-position confidence logits in FP32.

        Args:
            hidden: Unnormalized features from the collapsed residual streams.
            markov_embed: Low-rank features of the previous token.

        Returns:
            Confidence logits for each draft position.
        """
        joined = torch.cat([hidden, markov_embed], dim=-1)
        return self.proj(joined.float()).squeeze(-1)


class DSparkDraftAttention(nn.Module):
    """MQA over a causal main-context window and a noncausal draft block.

    Each anchor owns an independent draft block. Shared rotated KV includes
    the main window ending at that anchor and every position in its block.
    A learned per-head sink participates in softmax normalization.
    """

    def __init__(self, config: Any) -> None:
        """Create the checkpoint-shaped projection set."""
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = int(config.head_dim)
        self.rope_dim = int(round(self.head_dim * float(config.partial_rotary_factor)))
        if self.rope_dim <= 0 or self.rope_dim % 2 or self.rope_dim > self.head_dim:
            raise ValueError("DSpark rotary width must be positive, even and no larger than head_dim")
        self.window = int(config.v41_dspark_window)
        self.attention_chunk_size = int(getattr(config, "v41_dspark_attention_chunk_size", 0))
        self.attention_backend = getattr(config, "v41_dspark_attention_backend", "reference")
        if self.attention_backend not in ("reference", "flash_v3"):
            raise ValueError("DSpark attention backend must be reference or flash_v3")
        self._flash_engaged = False
        self.q_lora_rank = int(config.q_lora_rank)
        self.o_groups = int(config.o_groups)
        self.o_lora_rank = int(config.o_lora_rank)
        if self.num_heads % self.o_groups:
            raise ValueError("num_attention_heads must divide into o_groups")
        self.scaling = self.head_dim ** -0.5
        self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        heads_per_group = self.num_heads // self.o_groups
        self.wo_a = nn.Linear(heads_per_group * self.head_dim, self.o_lora_rank * self.o_groups, bias=False)
        self.wo_b = nn.Linear(self.o_lora_rank * self.o_groups, self.hidden_size, bias=False)
        self.attn_sink = nn.Parameter(torch.zeros(self.num_heads))

    def forward(
            self,
            draft_states: torch.Tensor,
            main_states: torch.Tensor,
            rope_cos: torch.Tensor,
            rope_sin: torch.Tensor,
            *,
            sequence_boundaries: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Attend each draft block to its main window and all of its draft KV.

        Args:
            draft_states: ``[batch, seq, block, hidden]`` collapsed draft input.
            main_states: ``[batch, seq, hidden]`` stage KV source stream.
            rope_cos: FP32 ``[1, seq + block, rope_dim / 2]`` cosine table.
            rope_sin: FP32 ``[1, seq + block, rope_dim / 2]`` sine table.
            sequence_boundaries: Optional complete-document boundaries for packed training.

        Returns:
            ``[batch, seq, block, hidden]`` attention output.
        """
        seq, block = draft_states.shape[1:3]
        if rope_cos.shape[1] < seq + block or rope_sin.shape != rope_cos.shape:
            raise ValueError("DSpark rotary tables must cover the main context and every future draft position")
        key_value = self.kv_norm(self.wkv(main_states))
        rope = self.rope_dim
        positions, _ = _document_geometry(seq, sequence_boundaries)
        main_positions = torch.tensor(positions, device=main_states.device, dtype=torch.long)
        key_value = torch.cat(
            [key_value[..., :-rope], _apply_rope(key_value[..., -rope:],
                                               rope_cos.index_select(1, main_positions),
                                               rope_sin.index_select(1, main_positions))],
            dim=-1,
        )
        chunk = self.attention_chunk_size
        if chunk == 0 or seq <= chunk:
            return self._attend_range(draft_states, key_value, rope_cos, rope_sin, 0, positions)
        outputs = []
        for start in range(0, seq, chunk):
            stop = min(start + chunk, seq)
            arguments = (draft_states[:, start:stop], key_value, rope_cos, rope_sin, start, positions[start:stop])
            if self.training and torch.is_grad_enabled():
                outputs.append(_checkpoint(self._attend_range, *arguments))
            else:
                outputs.append(self._attend_range(*arguments))
        return torch.cat(outputs, dim=1)

    def _attend_range(
            self,
            draft_states: torch.Tensor,
            key_value: torch.Tensor,
            rope_cos: torch.Tensor,
            rope_sin: torch.Tensor,
            start: int,
            document_positions: tuple[int, ...],
    ) -> torch.Tensor:
        """Run independent decode anchors at their absolute main and future positions."""
        batch, seq, block, _ = draft_states.shape
        rope = self.rope_dim
        local_positions = torch.tensor(document_positions, device=draft_states.device, dtype=torch.long)
        positions = (local_positions[:, None] + 1
                     + torch.arange(block, device=draft_states.device)[None, :])
        draft_cos = rope_cos.index_select(1, positions.flatten()).view(1, seq, block, rope // 2)
        draft_sin = rope_sin.index_select(1, positions.flatten()).view(1, seq, block, rope // 2)
        query = self.wq_b(self.q_norm(self.wq_a(draft_states)))
        query = query.view(batch, seq, block, self.num_heads, self.head_dim)
        query_rot = _apply_rope(query[..., -rope:], draft_cos.unsqueeze(-2), draft_sin.unsqueeze(-2))
        query = torch.cat([query[..., :-rope], query_rot], dim=-1)
        draft_kv = self.kv_norm(self.wkv(draft_states))
        draft_kv = torch.cat((draft_kv[..., :-rope],
                              _apply_rope(draft_kv[..., -rope:], draft_cos, draft_sin)), dim=-1)
        combined_kv, valid = self._gather_context(key_value, draft_kv, start, local_positions)
        if self.attention_backend == "flash_v3":
            context = self._flash_window_context(query, combined_kv, document_positions)
        else:
            context = self._reference_window_context(query, combined_kv, valid)
        context = torch.cat((context[..., :-rope], _apply_rope(
            context[..., -rope:], draft_cos.unsqueeze(-2), draft_sin.unsqueeze(-2), inverse=True)), dim=-1)
        grouped = context.reshape(batch, seq, block, self.o_groups, -1)
        wo_a = self.wo_a.weight.view(self.o_groups, self.o_lora_rank, -1)
        reduced = torch.einsum("bsjgd,grd->bsjgr", grouped, wo_a.to(context.dtype))
        return self.wo_b(reduced.flatten(3))

    def _gather_context(self, main_kv, draft_kv, start, local_positions):
        """Gather bounded main windows on the device and append all draft keys/values."""
        batch, seq, block, dim = draft_kv.shape
        positions = torch.arange(start, start + seq, device=main_kv.device)
        backwards = torch.arange(self.window - 1, -1, -1, device=main_kv.device)
        indices = positions[:, None] - backwards[None, :]
        document_starts = positions - local_positions
        valid_main = indices >= document_starts[:, None]
        # Flattened IndexSelect avoids the AI CPU advanced-indexing path and
        # correctly sums gradients for keys shared by overlapping windows.
        main_windows = main_kv.index_select(1, indices.clamp_min(0).flatten())
        main_windows = main_windows.view(batch, seq, self.window, dim)
        valid = torch.cat((valid_main, torch.ones(seq, block, device=main_kv.device, dtype=torch.bool)), dim=-1)
        return torch.cat((main_windows, draft_kv), dim=2), valid

    def _flash_window_context(self, query, key_value, positions):
        """Pack each anchor as an independent noncausal TND segment."""
        batch, seq, block, heads, dim = query.shape
        indices, query_lengths, kv_lengths = _draft_pack_metadata(
            batch, block, self.window, positions, query.device)
        packed_kv = key_value.reshape(-1, dim).index_select(0, indices).unsqueeze(1).to(query.dtype).contiguous()
        sink = self.attn_sink.to(query.dtype).float()
        output = _flash_attention_v3(
            query.reshape(batch * seq * block, heads, dim).contiguous(), packed_kv,
            sink, query_lengths, kv_lengths, self.scaling,
        )
        if not self._flash_engaged:
            logger.info("DSpark native FlashAttention v3 training path engaged with main and draft KV")
            self._flash_engaged = True
        return output.view(batch, seq, block, heads, dim)

    def _reference_window_context(self, query, key_value, valid):
        """Evaluate scalar-sink attention over each anchor's complete context."""
        batch, seq, block, _, _ = query.shape
        scores = torch.einsum("bsjhd,bswd->bshjw", query, key_value.to(query.dtype)) * self.scaling
        scores = scores.masked_fill(~valid[None, :, None, None, :], torch.finfo(scores.dtype).min)
        sink = self.attn_sink.view(1, 1, self.num_heads, 1, 1).to(scores.dtype)
        joined = torch.cat([scores, sink.expand(batch, seq, -1, block, 1)], dim=-1)
        weights = joined.float().softmax(dim=-1).to(query.dtype)[..., :-1]
        return torch.einsum("bshjw,bswd->bsjhd", weights, key_value.to(query.dtype))


class DSparkStage(nn.Module):
    """One trainable DSpark Transformer stage with pipelined-mHC semantics."""

    def __init__(self, config: Any, mlp_factory: Any, stage_index: int = 0) -> None:
        """Create the stage from the shared crop configuration.

        Args:
            config: Crop configuration carrying ``v41_dspark_*`` fields.
            mlp_factory: Zero-argument callable building the stage MoE FFN.
            stage_index: Stable logical index used for layout-independent initialization.
        """
        super().__init__()
        self.hc_mult = int(config.hc_mult)
        self.activation_offload = getattr(config, "v41_dspark_activation_offload", False)
        if not isinstance(self.activation_offload, bool):
            raise ValueError("v41_dspark_activation_offload must be a boolean")
        self.offload_pin_memory = getattr(config, "v41_dspark_offload_pin_memory", True)
        if not isinstance(self.offload_pin_memory, bool):
            raise ValueError("v41_dspark_offload_pin_memory must be a boolean")
        self.attn_hc = _DSparkHcCoefficients(config, initialization_seed=0x44530100 + stage_index)
        self.ffn_hc = _DSparkHcCoefficients(config, initialization_seed=0x44530200 + stage_index)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DSparkDraftAttention(config)
        self.mlp = mlp_factory()

    def _pre_ffn(
            self,
            hidden_streams: torch.Tensor,
            pre_mix: torch.Tensor,
            main_states: torch.Tensor,
            rope_cos: torch.Tensor,
            rope_sin: torch.Tensor,
            sequence_boundaries: tuple[int, ...] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the communication-free stage prefix up to the FFN input.

        Kept separate so activation checkpointing can recompute it without
        replaying the draft-MoE dispatch: collectives inside a recomputed
        region interleave with FSDP's backward collectives and deadlock
        across ranks.

        Args:
            hidden_streams: ``[batch, seq, block, streams, hidden]`` input.
            pre_mix: Previous-stage mHC pre-mix coefficients.
            main_states: Stage KV source stream.
            rope_cos: Rotary cosine table.
            rope_sin: Rotary sine table.
            sequence_boundaries: Complete-document boundaries of the packed training input.

        Returns:
            Post-attention flat stream, FFN input, and the mHC pre/post/
            combine coefficients of the FFN half.
        """
        from hyper_parallel.components.modules.mhc import pipelined_mhc_post  # pylint: disable=C0415

        batch, seq, block, streams, hidden = hidden_streams.shape
        flat = hidden_streams.reshape(batch, seq * block, streams, hidden)
        residual = flat
        attn_pre, attn_post, attn_comb = self.attn_hc(flat)
        collapsed = _collapse_streams(flat, pre_mix)
        attn_input = self.input_layernorm(collapsed).view(batch, seq, block, hidden)
        attn_output = self.self_attn(attn_input, main_states, rope_cos, rope_sin,
                                    sequence_boundaries=sequence_boundaries)
        flat = pipelined_mhc_post(attn_output.reshape(batch, seq * block, hidden), residual, attn_post, attn_comb)
        ffn_pre, ffn_post, ffn_comb = self.ffn_hc(flat)
        collapsed = _collapse_streams(flat, attn_pre)
        ffn_input = self.post_attention_layernorm(collapsed)
        return flat, ffn_input, ffn_pre, ffn_post, ffn_comb

    def forward(
            self,
            hidden_streams: torch.Tensor,
            pre_mix: torch.Tensor,
            main_states: torch.Tensor,
            rope_cos: torch.Tensor,
            rope_sin: torch.Tensor,
            input_ids: torch.Tensor,
            use_checkpoint: bool = False,
            sequence_boundaries: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one DSpark stage over the flattened draft stream.

        Args:
            hidden_streams: ``[batch, seq, block, streams, hidden]`` input.
            pre_mix: Previous-stage mHC pre-mix coefficients.
            main_states: Stage KV source stream.
            rope_cos: Rotary cosine table.
            rope_sin: Rotary sine table.
            input_ids: Flattened draft token ids for the MoE router.
            use_checkpoint: Recompute the communication-free prefix in
                backward instead of storing its activations.
            sequence_boundaries: Complete-document boundaries of the packed training input.

        Returns:
            Updated hidden streams and the FFN-half pre-mix coefficients.
        """
        from hyper_parallel.components.modules.mhc import pipelined_mhc_post  # pylint: disable=C0415

        batch, seq, block, streams, hidden = hidden_streams.shape
        offload = self.activation_offload and self.training and torch.is_grad_enabled()
        # Store backward tensors on Host without replaying EP collectives. The
        # existing local checkpoints still bound temporary activations; their
        # saved inputs now release device storage between draft stages.
        storage = (torch.autograd.graph.save_on_cpu(
            pin_memory=self.offload_pin_memory and hidden_streams.device.type != "cpu",
            device_type=hidden_streams.device.type,
        ) if offload else nullcontext())
        with storage:
            if use_checkpoint:
                flat, ffn_input, ffn_pre, ffn_post, ffn_comb = _checkpoint(
                    self._pre_ffn, hidden_streams, pre_mix, main_states,
                    rope_cos, rope_sin, sequence_boundaries)
            else:
                flat, ffn_input, ffn_pre, ffn_post, ffn_comb = self._pre_ffn(
                    hidden_streams, pre_mix, main_states, rope_cos, rope_sin, sequence_boundaries)
            ffn_output = self.mlp(ffn_input, input_ids=input_ids)
            flat = pipelined_mhc_post(ffn_output, flat, ffn_post, ffn_comb)
            return flat.view(batch, seq, block, streams, hidden), ffn_pre


class _DSparkHcCoefficients(nn.Module):
    """Parameter-owning pipelined-mHC coefficient module for DSpark stages."""

    def __init__(self, config: Any, *, initialization_seed: int = 0x44530100) -> None:
        """Create the ``fn/base/scale`` coefficient layout."""
        super().__init__()
        self.initialization_seed = initialization_seed
        self.hc_mult = int(config.hc_mult)
        self.hc_sinkhorn_iters = int(config.hc_sinkhorn_iters)
        self.hc_eps = float(config.hc_eps)
        self.norm_eps = float(config.rms_norm_eps)
        mix = (self.hc_mult + 2) * self.hc_mult
        self.fn = nn.Parameter(
            torch.randn(mix, self.hc_mult * config.hidden_size) * float(config.initializer_range)
        )
        self.base = nn.Parameter(torch.zeros(mix))
        self.scale = nn.Parameter(torch.ones(3))

    @torch.no_grad()
    def init_weights(self, std: float) -> None:
        """Initialize logical weights on the parameter owner after meta materialization.

        Args:
            std: Standard deviation of the canonical full FP32 random draw.
        """
        target = self.fn.to_local() if isinstance(self.fn, DTensor) else self.fn
        generator = torch.Generator().manual_seed(self.initialization_seed)
        values = torch.randn(self.fn.shape, generator=generator, device="cpu") * std
        values = values.to(target.device, target.dtype)
        if isinstance(self.fn, DTensor):
            values = distribute_tensor(values, self.fn.device_mesh, self.fn.placements).to_local()
        target.copy_(values)
        base = self.base.to_local() if isinstance(self.base, DTensor) else self.base
        scale = self.scale.to_local() if isinstance(self.scale, DTensor) else self.scale
        base.zero_()
        scale.fill_(1.0)

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce pre/post/combine coefficients for the next sublayers.

        Args:
            hidden_streams: Residual streams with trailing stream and hidden dimensions.

        Returns:
            Input-collapse, output-expansion and residual-mixing coefficients.
        """
        # The released coefficient path has no trainable RMSNorm weight. Keep its
        # statistic epsilon separate from the Sinkhorn/coefficient epsilon.
        mix = _NormalizedHcProjection.apply(hidden_streams, self.fn.float(), self.norm_eps)
        streams = self.hc_mult
        pre, post, comb = mix.split([streams, streams, streams * streams], dim=-1)
        pre_bias, post_bias, comb_bias = self.base.split([streams, streams, streams * streams])
        pre_scale, post_scale, comb_scale = self.scale[0], self.scale[1], self.scale[2]
        pre = torch.sigmoid(pre * pre_scale + pre_bias) + self.hc_eps
        post = 2 * torch.sigmoid(post * post_scale + post_bias)
        comb = comb.view(*comb.shape[:-1], streams, streams) * comb_scale + comb_bias.view(streams, streams)
        comb = sinkhorn_knopps(comb, self.hc_sinkhorn_iters, self.hc_eps)
        return pre, post, comb


class DeepseekV41DSpark(nn.Module):
    """Trainable DSpark drafter head attached behind the V4.1 backbone."""

    def __init__(self, config: Any, mlp_factory: Any) -> None:
        """Create ``num_stages`` DSpark stages plus entry/exit components."""
        super().__init__()
        self.block_size = int(config.v41_dspark_block_size)
        self.noise_token_id = int(config.v41_dspark_noise_token_id)
        self.num_targets = len(tuple(config.v41_dspark_target_layer_ids))
        self.hc_mult = int(config.hc_mult)
        self.rope_theta = float(config.rope_theta)
        if not hasattr(config, "v41_dspark_rope_scaling"):
            raise ValueError("DSpark requires v41_dspark_rope_scaling copied from the released text configuration")
        self.rope_scaling = dict(config.v41_dspark_rope_scaling or {})
        self._validate_rotary_settings()
        self.head_dim = int(config.head_dim)
        self.rope_dim = int(round(self.head_dim * float(config.partial_rotary_factor)))
        self.confidence_coeff = float(config.v41_dspark_confidence_coeff)
        # Sequence-chunk granularity of the draft objective: smaller chunks
        # lower the vocabulary-logits peak at the cost of more recompute
        # rounds; the loss value itself is chunking-invariant.
        self.loss_chunk_size = int(getattr(config, "v41_dspark_loss_chunk_size", 64) or 64)
        self.stage_checkpoint = getattr(config, "v41_dspark_stage_checkpoint", False)
        if not isinstance(self.stage_checkpoint, bool):
            raise ValueError("v41_dspark_stage_checkpoint must be a boolean")
        self.stage_checkpoint_offload = getattr(config, "v41_dspark_stage_checkpoint_offload", False)
        if not isinstance(self.stage_checkpoint_offload, bool):
            raise ValueError("v41_dspark_stage_checkpoint_offload must be a boolean")
        if self.stage_checkpoint_offload and not self.stage_checkpoint:
            raise ValueError("DSpark checkpoint input offload requires complete stage checkpointing")
        if self.stage_checkpoint and getattr(config, "v41_dspark_activation_offload", False):
            raise ValueError("DSpark stage checkpointing and activation offload are mutually exclusive")
        if self.block_size <= 0 or self.num_targets == 0:
            raise ValueError("DSpark requires a positive block size and target layers")
        hidden = int(config.hidden_size)
        self.main_proj = nn.Linear(hidden * self.num_targets, hidden, bias=False)
        self.main_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        depth = int(config.v41_dspark_depth)
        self.stages = nn.ModuleList(DSparkStage(config, mlp_factory, index) for index in range(depth))
        self.norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(int(config.vocab_size), int(config.v41_dspark_markov_rank))
        self.confidence_head = DSparkConfidenceHead(hidden + int(config.v41_dspark_markov_rank))
        self._megamoe_runtime_prepared = False

    def _prepare_megamoe_runtime(self) -> None:
        """Share transient expert storage across serial drafter stages."""
        if self._megamoe_runtime_prepared:
            return
        experts = [stage.mlp.experts for stage in self.stages
                   if isinstance(getattr(stage.mlp, "experts", None), DeepseekV41MegaMoeExperts)]
        DeepseekV41MegaMoeExperts.prepare_runtime_group(experts)
        self._megamoe_runtime_prepared = True

    def _validate_rotary_settings(self):
        """Reject unsupported rotary variants instead of silently dropping release settings."""
        if self.rope_theta <= 1:
            raise ValueError("DSpark rope_theta must exceed one")
        if self.rope_scaling.get("rope_type", "default") not in ("default", "yarn"):
            raise ValueError("DSpark supports the released default and YaRN rotary settings")
        if any(float(self.rope_scaling.get(key, default)) <= 0
               for key, default in (("factor", 1), ("beta_fast", 32), ("beta_slow", 1))):
            raise ValueError("DSpark YaRN factor and beta values must be positive")

    def _rope_tables(self, seq: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Build FP32 adjacent-pair YaRN tables including every future draft offset."""
        del dtype
        dim = self.rope_dim
        inv_freq = 1.0 / self.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        original = int(self.rope_scaling.get("original_max_position_embeddings", 0))
        if original > 0:
            beta_fast = float(self.rope_scaling.get("beta_fast", 32))
            beta_slow = float(self.rope_scaling.get("beta_slow", 1))
            low = max(math.floor(dim * math.log(original / (beta_fast * 2 * math.pi))
                                 / (2 * math.log(self.rope_theta))), 0)
            high = min(math.ceil(dim * math.log(original / (beta_slow * 2 * math.pi))
                                 / (2 * math.log(self.rope_theta))), dim - 1)
            ramp = ((torch.arange(dim // 2, device=device, dtype=torch.float32) - low)
                    / max(high - low, 0.001)).clamp(0, 1)
            factor = float(self.rope_scaling.get("factor", 1))
            inv_freq = inv_freq / factor * ramp + inv_freq * (1 - ramp)
        angles = torch.arange(seq + self.block_size, device=device, dtype=torch.float32)[:, None] * inv_freq[None, :]
        return angles.cos()[None], angles.sin()[None]

    def init_weights(self, std: float) -> None:
        """Deterministically initialize drafter-owned bare parameters.

        Meta-device construction discards __init__-time values. Draw each
        small owned parameter as a canonical full CPU tensor, then select
        its local shard. Replica ranks and different FSDP layouts must
        represent the same logical weights.

        Args:
            std: Backbone initializer range for the random draws.
        """
        for stage in self.stages:
            stage.attn_hc.init_weights(std)
            stage.ffn_hc.init_weights(std)
            with torch.no_grad():
                stage.self_attn.attn_sink.zero_()

    def _teacher_forced_targets(self, input_ids, labels, token_mask, sequence_boundaries):
        """Align the released decode inputs and targets without crossing document or image spans."""
        batch, seq = input_ids.shape
        block = self.block_size
        _, document_ends = _document_geometry(seq, sequence_boundaries)
        previous = (torch.arange(seq, device=input_ids.device)[:, None] + 1
                    + torch.arange(block, device=input_ids.device)[None, :])
        previous_flat = previous.reshape(1, -1).expand(batch, -1)
        next_flat = (previous + 1).reshape(1, -1).expand(batch, -1)
        padded = functional.pad(input_ids, (0, block + 1), value=self.noise_token_id)
        markov_inputs = padded.gather(1, previous_flat).view(batch, seq, block)
        targets = padded.gather(1, next_flat).view(batch, seq, block)
        ends = torch.tensor(document_ends, device=input_ids.device)[:, None]
        valid = (previous + 1 < ends).unsqueeze(0).expand(batch, -1, -1)
        # Trainer labels already predict the next token. Their values must not
        # be shifted a second time; only the supervision mask is reused.
        if labels is not None:
            mask = functional.pad(labels != IGNORE_INDEX, (0, block + 1), value=False)
            valid = valid & mask.gather(1, previous_flat).view(batch, seq, block)
        if token_mask is not None:
            # Every text-only draft prefix, including its first input and its
            # target, must stop before a visual span. Main KV may contain images.
            text_indices = (torch.arange(seq, device=input_ids.device)[:, None] + 1
                            + torch.arange(block + 1, device=input_ids.device)[None, :])
            text = functional.pad(token_mask, (0, block + 1), value=False)
            text = text.gather(1, text_indices.reshape(1, -1).expand(batch, -1))
            uninterrupted = (~text.view(batch, seq, block + 1)).cumsum(-1)[..., 1:] == 0
            valid = valid & uninterrupted
        return markov_inputs, targets.masked_fill(~valid, IGNORE_INDEX)

    def _validate_backbone_inputs(self, target_hiddens, input_ids, inputs_embeds, noise_embedding):
        """Require complete, detached backbone surfaces instead of silently training a changed model."""
        if any(hidden.requires_grad for hidden in target_hiddens) or inputs_embeds.requires_grad:
            raise ValueError("DSpark target hiddens and inputs_embeds must be detached from the backbone")
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("DSpark input_ids must contain a nonempty complete sequence")
        expected = (*input_ids.shape, self.norm.weight.shape[0])
        if (len(target_hiddens) != self.num_targets or inputs_embeds.shape != expected
                or any(hidden.shape != expected for hidden in target_hiddens)):
            raise ValueError("DSpark embeddings and feature taps must match the complete token sequence")
        if (noise_embedding is None or noise_embedding.requires_grad
                or noise_embedding.shape != (expected[-1],)):
            raise ValueError("DSpark noise_embedding must be one detached backbone embedding row")

    def forward(
            self,
            target_hiddens: list[torch.Tensor],
            input_ids: torch.Tensor,
            labels: torch.Tensor | None,
            inputs_embeds: torch.Tensor,
            lm_head: nn.Linear,
            *,
            noise_embedding: torch.Tensor,
            token_mask: torch.Tensor | None = None,
            sequence_boundaries: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the DSpark draft objective for one batch.

        Args:
            target_hiddens: Detached collapsed backbone hiddens, one per
                configured target layer, each ``[batch, seq, hidden]``.
            input_ids: ``[batch, seq]`` backbone input tokens.
            labels: Optional ``[batch, seq]`` next-token-shifted labels with
                ``IGNORE_INDEX`` holes. The causal-LM entry point normalizes
                unshifted Omni labels before this boundary. ``None`` supervises
                every valid future token from ``input_ids``.
            inputs_embeds: Detached token embeddings before visual feature injection.
            lm_head: Backbone output head tied into the drafter.
            noise_embedding: Detached backbone embedding row of the configured noise token.
            token_mask: Optional text-token mask; draft blocks do not predict visual placeholders.
            sequence_boundaries: Host boundaries of complete packed documents, shared across batch rows.

        Returns:
            The scalar DSpark loss and a metrics dictionary.
        """
        self._validate_backbone_inputs(target_hiddens, input_ids, inputs_embeds, noise_embedding)
        self._prepare_megamoe_runtime()
        batch, seq = input_ids.shape
        block = self.block_size
        device = input_ids.device
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("DSpark labels must match input_ids")
        if token_mask is not None and (token_mask.shape != input_ids.shape or token_mask.dtype != torch.bool):
            raise ValueError("DSpark token_mask must be a boolean mask matching input_ids")
        _, document_ends = _document_geometry(seq, sequence_boundaries)
        main_states = self.main_norm(self.main_proj(torch.cat(target_hiddens, dim=-1)))

        # Tied-but-frozen backbone surfaces: the drafter reuses the backbone
        # token embeddings (already computed by the backbone forward, handed
        # over detached) and the frozen output head; the report forbids DSpark
        # gradients from reaching the backbone.
        token_embeddings = inputs_embeds
        noise = noise_embedding.to(token_embeddings.dtype)
        draft = torch.empty(batch, seq, block, token_embeddings.shape[-1],
                            dtype=token_embeddings.dtype, device=device)
        first_embeddings = torch.cat((token_embeddings[:, 1:], noise.expand(batch, 1, -1)), dim=1)
        next_is_in_document = (torch.arange(seq, device=device) + 1
                               < torch.tensor(document_ends, device=device))
        draft[:, :, 0] = torch.where(next_is_in_document[None, :, None], first_embeddings, noise)
        draft[:, :, 1:] = noise
        streams = draft.unsqueeze(3).expand(-1, -1, -1, self.hc_mult, -1).contiguous()
        # The expanded stream owns its storage. Release construction buffers
        # before the stage workspace and residual outputs reach their peak.
        del draft, first_embeddings
        pre_mix = streams.new_zeros(batch, seq * block, self.hc_mult, dtype=torch.float32)
        pre_mix[:, :, 0] = 1.0
        rope_cos, rope_sin = self._rope_tables(seq, device, token_embeddings.dtype)
        draft_ids = input_ids.new_full((batch, seq, block), self.noise_token_id)
        first_ids = functional.pad(input_ids[:, 1:], (0, 1), value=self.noise_token_id)
        draft_ids[:, :, 0] = torch.where(next_is_in_document, first_ids, self.noise_token_id)
        draft_ids = draft_ids.reshape(batch, seq * block)
        # The flattened draft stream (seq x block positions, hc streams) makes
        # stage activations the dominant memory term. The default checkpoints
        # only the communication-free prefix; full-stage replay also trades
        # extra expert communication for a smaller saved-activation footprint.
        gradient_active = torch.is_grad_enabled() and self.training
        for stage in self.stages:
            if self.stage_checkpoint and gradient_active:
                streams, pre_mix = _checkpoint_stage(
                    stage, streams, pre_mix, main_states, rope_cos, rope_sin, draft_ids,
                    offload_inputs=self.stage_checkpoint_offload,
                    # Retain the local prefix checkpoint during stage replay:
                    # otherwise its saved activations overlap the expert workspace.
                    use_checkpoint=True, sequence_boundaries=sequence_boundaries,
                )
            else:
                streams, pre_mix = stage(streams, pre_mix, main_states, rope_cos,
                                         rope_sin, draft_ids, use_checkpoint=gradient_active,
                                         sequence_boundaries=sequence_boundaries)
        def _collapse(streams_in: torch.Tensor, pre_mix_in: torch.Tensor) -> torch.Tensor:
            return _collapse_streams(streams_in, pre_mix_in)

        # The FP32 merge buffer otherwise stays resident for backward;
        # recomputing it is cheap and communication-free.
        if gradient_active:
            hidden_out = _checkpoint(_collapse, streams, pre_mix)
        else:
            hidden_out = _collapse(streams, pre_mix)

        markov_inputs, targets = self._teacher_forced_targets(
            input_ids, labels, token_mask, sequence_boundaries)

        # Sequence-chunked objective: the [batch, seq, block, vocab] draft
        # logits are never materialized whole.  Each chunk recomputes its
        # logits inside activation checkpointing, bounding the vocabulary
        # spike to one chunk (same technique as the repository chunk loss).
        # The tied head may live as a sharded distributed tensor once it owns
        # an FSDP unit; gather one frozen full copy for the draft projection.
        raw_head_weight = lm_head.weight
        if hasattr(raw_head_weight, "full_tensor"):
            raw_head_weight = raw_head_weight.full_tensor()
        head_weight = raw_head_weight.detach()
        chunk = self.loss_chunk_size

        def _chunk_objective(hidden_chunk, markov_chunk, target_chunk):
            markov_bias, markov_embed = self.markov_head(markov_chunk)
            # Only vocabulary projection uses the final RMSNorm. The confidence
            # head consumes the unnormalized merged features in the release.
            logits = functional.linear(  # pylint: disable=not-callable
                self.norm(hidden_chunk), head_weight,
            ) + markov_bias
            chunk_mask = target_chunk != IGNORE_INDEX
            ce_sum = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                target_chunk.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            with torch.no_grad():
                chunk_hits = (logits.argmax(dim=-1) == target_chunk) & chunk_mask
                hit_sum = chunk_hits.sum()
            confidence = self.confidence_head(hidden_chunk, markov_embed)
            bce = functional.binary_cross_entropy_with_logits(
                confidence, chunk_hits.float(), reduction="none")
            bce_sum = (bce * chunk_mask).sum()
            return ce_sum, bce_sum, hit_sum.float()

        ce_total = hidden_out.new_zeros((), dtype=torch.float32)
        bce_total = hidden_out.new_zeros((), dtype=torch.float32)
        hit_total = hidden_out.new_zeros((), dtype=torch.float32)
        for start in range(0, seq, chunk):
            stop = min(start + chunk, seq)
            ce_sum, bce_sum, hit_sum = _checkpoint(
                _chunk_objective,
                hidden_out[:, start:stop],
                markov_inputs[:, start:stop],
                targets[:, start:stop],
            )
            ce_total = ce_total + ce_sum
            bce_total = bce_total + bce_sum
            hit_total = hit_total + hit_sum
        supervised = (targets != IGNORE_INDEX).sum()
        denominator = supervised.clamp(min=1).float()
        draft_ce = ce_total / denominator
        confidence_bce = bce_total / denominator
        loss = draft_ce + self.confidence_coeff * confidence_bce
        accuracy = (hit_total / denominator).detach()
        metrics = {
            "dspark_ce": draft_ce.detach(),
            "dspark_confidence_bce": confidence_bce.detach(),
            "dspark_draft_accuracy": accuracy,
            "dspark_supervised_tokens": supervised.detach(),
        }
        return loss, metrics


__all__ = [
    "DSparkConfidenceHead",
    "DSparkDraftAttention",
    "DSparkMarkovHead",
    "DSparkStage",
    "DSparkTokenEmbedding",
    "DeepseekV41DSpark",
]
