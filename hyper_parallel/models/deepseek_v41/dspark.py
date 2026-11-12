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
* The draft objective is token-weighted cross-entropy against the ``j+1``-step
  future token; the confidence objective is a BCE against greedy-hit events.
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
from functools import lru_cache
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from torch.nn import functional  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional.sinkhorn import sinkhorn_knopps
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.data.constants import IGNORE_INDEX

logger = logging.getLogger(__name__)


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


def _apply_rope(tensor, cos, sin, *, inverse=False):
    """Rotate adjacent pairs in FP32, preserving the release's BF16 cast boundary."""
    pairs = tensor.float().unflatten(-1, (-1, 2))
    if inverse:
        sin = -sin
    real = pairs[..., 0] * cos - pairs[..., 1] * sin
    imaginary = pairs[..., 0] * sin + pairs[..., 1] * cos
    return torch.stack((real, imaginary), dim=-1).flatten(-2).to(tensor.dtype)


@lru_cache(maxsize=64)
def _draft_pack_metadata(batch, sequence, block, window, start, device):
    """Build static TND metadata without synchronizing device-resident masks."""
    row_width = window + block
    lengths = [min(window, start + anchor + 1) + block for anchor in range(sequence)] * batch
    indices = [row * row_width + offset
               for row, length in enumerate(lengths)
               for offset in range(row_width - length, row_width)]
    query_lengths = torch.arange(1, batch * sequence + 1, dtype=torch.int64) * block
    kv_lengths = torch.tensor(lengths, dtype=torch.int64).cumsum(0)
    return torch.tensor(indices, device=device, dtype=torch.long), query_lengths, kv_lengths


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
    ) -> torch.Tensor:
        """Attend each draft block to its main window and all of its draft KV.

        Args:
            draft_states: ``[batch, seq, block, hidden]`` collapsed draft input.
            main_states: ``[batch, seq, hidden]`` stage KV source stream.
            rope_cos: FP32 ``[1, seq + block, rope_dim / 2]`` cosine table.
            rope_sin: FP32 ``[1, seq + block, rope_dim / 2]`` sine table.

        Returns:
            ``[batch, seq, block, hidden]`` attention output.
        """
        seq, block = draft_states.shape[1:3]
        if rope_cos.shape[1] < seq + block or rope_sin.shape != rope_cos.shape:
            raise ValueError("DSpark rotary tables must cover the main context and every future draft position")
        key_value = self.kv_norm(self.wkv(main_states))
        rope = self.rope_dim
        key_value = torch.cat(
            [key_value[..., :-rope], _apply_rope(key_value[..., -rope:], rope_cos[:, :seq], rope_sin[:, :seq])],
            dim=-1,
        )
        chunk = self.attention_chunk_size
        if chunk == 0 or seq <= chunk:
            return self._attend_range(draft_states, key_value, rope_cos, rope_sin, 0)
        outputs = []
        for start in range(0, seq, chunk):
            stop = min(start + chunk, seq)
            arguments = (draft_states[:, start:stop], key_value, rope_cos, rope_sin, start)
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
    ) -> torch.Tensor:
        """Run independent decode anchors at their absolute main and future positions."""
        batch, seq, block, _ = draft_states.shape
        rope = self.rope_dim
        positions = (torch.arange(start + 1, start + seq + 1, device=draft_states.device)[:, None]
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
        combined_kv, valid = self._gather_context(key_value, draft_kv, start)
        if self.attention_backend == "flash_v3":
            context = self._flash_window_context(query, combined_kv, start)
        else:
            context = self._reference_window_context(query, combined_kv, valid)
        context = torch.cat((context[..., :-rope], _apply_rope(
            context[..., -rope:], draft_cos.unsqueeze(-2), draft_sin.unsqueeze(-2), inverse=True)), dim=-1)
        grouped = context.reshape(batch, seq, block, self.o_groups, -1)
        wo_a = self.wo_a.weight.view(self.o_groups, self.o_lora_rank, -1)
        reduced = torch.einsum("bsjgd,grd->bsjgr", grouped, wo_a.to(context.dtype))
        return self.wo_b(reduced.flatten(3))

    def _gather_context(self, main_kv, draft_kv, start):
        """Gather bounded main windows on the device and append all draft keys/values."""
        batch, seq, block, dim = draft_kv.shape
        positions = torch.arange(start, start + seq, device=main_kv.device)
        backwards = torch.arange(self.window - 1, -1, -1, device=main_kv.device)
        indices = positions[:, None] - backwards[None, :]
        valid_main = indices >= 0
        # Flattened IndexSelect avoids the AI CPU advanced-indexing path and
        # correctly sums gradients for keys shared by overlapping windows.
        main_windows = main_kv.index_select(1, indices.clamp_min(0).flatten())
        main_windows = main_windows.view(batch, seq, self.window, dim)
        valid = torch.cat((valid_main, torch.ones(seq, block, device=main_kv.device, dtype=torch.bool)), dim=-1)
        return torch.cat((main_windows, draft_kv), dim=2), valid

    def _flash_window_context(self, query, key_value, start):
        """Pack each anchor as an independent noncausal TND segment."""
        batch, seq, block, heads, dim = query.shape
        indices, query_lengths, kv_lengths = _draft_pack_metadata(
            batch, seq, block, self.window, start, query.device)
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

    def __init__(self, config: Any, mlp_factory: Any) -> None:
        """Create the stage from the shared crop configuration.

        Args:
            config: Crop configuration carrying ``v41_dspark_*`` fields.
            mlp_factory: Zero-argument callable building the stage MoE FFN.
        """
        super().__init__()
        self.hc_mult = int(config.hc_mult)
        self.attn_hc = _DSparkHcCoefficients(config)
        self.ffn_hc = _DSparkHcCoefficients(config)
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

        Returns:
            Post-attention flat stream, FFN input, and the mHC pre/post/
            combine coefficients of the FFN half.
        """
        from hyper_parallel.components.modules.mhc import pipelined_mhc_post  # pylint: disable=C0415

        batch, seq, block, streams, hidden = hidden_streams.shape
        flat = hidden_streams.reshape(batch, seq * block, streams, hidden)
        residual = flat
        attn_pre, attn_post, attn_comb = self.attn_hc(flat)
        collapsed = (pre_mix.unsqueeze(-1) * flat.float()).sum(dim=2).to(flat.dtype)
        attn_input = self.input_layernorm(collapsed).view(batch, seq, block, hidden)
        attn_output = self.self_attn(attn_input, main_states, rope_cos, rope_sin)
        flat = pipelined_mhc_post(attn_output.reshape(batch, seq * block, hidden), residual, attn_post, attn_comb)
        ffn_pre, ffn_post, ffn_comb = self.ffn_hc(flat)
        collapsed = (attn_pre.unsqueeze(-1) * flat.float()).sum(dim=2).to(flat.dtype)
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

        Returns:
            Updated hidden streams and the FFN-half pre-mix coefficients.
        """
        from hyper_parallel.components.modules.mhc import pipelined_mhc_post  # pylint: disable=C0415

        batch, seq, block, streams, hidden = hidden_streams.shape
        if use_checkpoint:
            flat, ffn_input, ffn_pre, ffn_post, ffn_comb = _checkpoint(
                self._pre_ffn, hidden_streams, pre_mix, main_states,
                rope_cos, rope_sin)
        else:
            flat, ffn_input, ffn_pre, ffn_post, ffn_comb = self._pre_ffn(
                hidden_streams, pre_mix, main_states, rope_cos, rope_sin)
        ffn_output = self.mlp(ffn_input, input_ids=input_ids)
        flat = pipelined_mhc_post(ffn_output, flat, ffn_post, ffn_comb)
        return flat.view(batch, seq, block, streams, hidden), ffn_pre


class _DSparkHcCoefficients(nn.Module):
    """Parameter-owning pipelined-mHC coefficient module for DSpark stages."""

    def __init__(self, config: Any) -> None:
        """Create the ``fn/base/scale`` coefficient layout."""
        super().__init__()
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

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce pre/post/combine coefficients for the next sublayers.

        Args:
            hidden_streams: Residual streams with trailing stream and hidden dimensions.

        Returns:
            Input-collapse, output-expansion and residual-mixing coefficients.
        """
        # The released coefficient path has no trainable RMSNorm weight. Keep its
        # statistic epsilon separate from the Sinkhorn/coefficient epsilon.
        flattened = hidden_streams.flatten(start_dim=2).float()
        inverse_rms = torch.rsqrt(flattened.square().mean(-1, keepdim=True) + self.norm_eps)
        mix = functional.linear(flattened, self.fn.float())  # pylint: disable=not-callable
        mix = mix * inverse_rms
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
        if self.block_size <= 0 or self.num_targets == 0:
            raise ValueError("DSpark requires a positive block size and target layers")
        hidden = int(config.hidden_size)
        self.main_proj = nn.Linear(hidden * self.num_targets, hidden, bias=False)
        self.main_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        depth = int(config.v41_dspark_depth)
        self.stages = nn.ModuleList(DSparkStage(config, mlp_factory) for _ in range(depth))
        # v1 deviation (recorded): the released checkpoint feeds the noise
        # placeholder through the tied backbone embedding row. This independent
        # trainable vector differs from that frozen, shared embedding contract.
        # An all-zero noise vector puts zero vectors through the stage
        # RMSNorms, whose backward scales like 1/sqrt(eps) and compounds
        # across stages into overflow-level gradients.
        self.noise_embedding = nn.Parameter(
            torch.randn(hidden) * float(config.initializer_range))
        self.norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(int(config.vocab_size), int(config.v41_dspark_markov_rank))
        self.confidence_head = DSparkConfidenceHead(hidden + int(config.v41_dspark_markov_rank))

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
        def _local(parameter: torch.Tensor) -> torch.Tensor:
            return parameter.to_local() if hasattr(parameter, "to_local") else parameter

        def _fill(parameter: torch.Tensor, seed: int) -> None:
            target = _local(parameter)
            generator = torch.Generator().manual_seed(seed)
            values = torch.randn(parameter.shape, generator=generator, device="cpu") * std
            values = values.to(target.device, target.dtype)
            if isinstance(parameter, DTensor):
                values = distribute_tensor(values, parameter.device_mesh, parameter.placements).to_local()
            with torch.no_grad():
                target.copy_(values)

        _fill(self.noise_embedding, 0x44530001)
        for index, stage in enumerate(self.stages):
            _fill(stage.attn_hc.fn, 0x44530100 + index)
            _fill(stage.ffn_hc.fn, 0x44530200 + index)
            with torch.no_grad():
                _local(stage.attn_hc.base).zero_()
                _local(stage.ffn_hc.base).zero_()
                _local(stage.attn_hc.scale).fill_(1.0)
                _local(stage.ffn_hc.scale).fill_(1.0)
                _local(stage.self_attn.attn_sink).zero_()

    def forward(
            self,
            target_hiddens: list[torch.Tensor],
            input_ids: torch.Tensor,
            labels: torch.Tensor | None,
            inputs_embeds: torch.Tensor,
            lm_head: nn.Linear,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the DSpark draft objective for one batch.

        Args:
            target_hiddens: Detached collapsed backbone hiddens, one per
                configured target layer, each ``[batch, seq, hidden]``.
            input_ids: ``[batch, seq]`` backbone input tokens.
            labels: Optional ``[batch, seq]`` supervised labels with
                ``IGNORE_INDEX`` holes; ``None`` supervises every position
                from ``input_ids`` (full LM-draft objective).
            embed_tokens: Backbone embedding tied into the drafter.
            lm_head: Backbone output head tied into the drafter.

        Returns:
            The scalar DSpark loss and a metrics dictionary.
        """
        if any(hidden.requires_grad for hidden in target_hiddens):
            raise ValueError("DSpark target hiddens must be detached from the backbone")
        batch, seq = input_ids.shape
        block = self.block_size
        device = input_ids.device
        main_states = self.main_norm(self.main_proj(torch.cat(target_hiddens, dim=-1)))

        # Tied-but-frozen backbone surfaces: the drafter reuses the backbone
        # token embeddings (already computed by the backbone forward, handed
        # over detached) and the frozen output head; the report forbids DSpark
        # gradients from reaching the backbone.
        if inputs_embeds.requires_grad:
            raise ValueError("DSpark inputs_embeds must be detached from the backbone")
        token_embeddings = inputs_embeds
        noise = self.noise_embedding.to(token_embeddings.dtype)
        draft = torch.empty(batch, seq, block, token_embeddings.shape[-1],
                            dtype=token_embeddings.dtype, device=device)
        draft[:, :, 0] = token_embeddings
        draft[:, :, 1:] = noise
        streams = draft.unsqueeze(3).expand(-1, -1, -1, self.hc_mult, -1).contiguous()
        pre_mix = streams.new_zeros(batch, seq * block, self.hc_mult, dtype=torch.float32)
        pre_mix[:, :, 0] = 1.0
        rope_cos, rope_sin = self._rope_tables(seq, device, token_embeddings.dtype)
        draft_ids = input_ids.unsqueeze(-1).expand(-1, -1, block).reshape(batch, seq * block)
        # The flattened draft stream (seq x block positions, hc streams) makes
        # stage activations the dominant memory term; each stage recomputes
        # its communication-free prefix in backward while the draft-MoE
        # dispatch stays outside the recomputed region (DSparkStage._pre_ffn).
        gradient_active = torch.is_grad_enabled() and self.training
        for stage in self.stages:
            streams, pre_mix = stage(streams, pre_mix, main_states, rope_cos,
                                     rope_sin, draft_ids, use_checkpoint=gradient_active)
        def _collapse(streams_in: torch.Tensor, pre_mix_in: torch.Tensor) -> torch.Tensor:
            merged = (pre_mix_in.unsqueeze(-1)
                      * streams_in.reshape(batch, seq * block, self.hc_mult, -1).float())
            merged = merged.sum(dim=2).to(streams_in.dtype).view(batch, seq, block, -1)
            return merged

        # The FP32 merge buffer otherwise stays resident for backward;
        # recomputing it is cheap and communication-free.
        if gradient_active:
            hidden_out = _checkpoint(_collapse, streams, pre_mix)
        else:
            hidden_out = _collapse(streams, pre_mix)

        # Teacher-forced targets: draft offset j at position t predicts the
        # token at t + 1 + j; the Markov head is conditioned on t + j.  When
        # the trainer keeps labels on the loss-only path (the model never sees
        # them), the drafter supervises every next token from input_ids — the
        # standard LM-draft objective; labels only add the SFT ignore mask.
        pad_ids = functional.pad(input_ids, (0, block), value=self.noise_token_id)
        offsets = torch.arange(block, device=device)
        gather = torch.arange(seq, device=device)[:, None] + offsets[None, :]
        gather_flat = gather.reshape(1, -1).expand(batch, -1)
        markov_inputs = pad_ids.gather(1, gather_flat).view(batch, seq, block)
        # Draft targets always come from input_ids: offset j at position t
        # predicts input_ids[t + 1 + j].  Trainer labels are already shifted
        # (labels[t] supervises input_ids[t + 1]), so they must not be
        # gathered as values; they only contribute the SFT ignore mask,
        # aligned via labels[t + j] <-> input_ids[t + 1 + j].
        pad_target_ids = functional.pad(input_ids, (0, block), value=IGNORE_INDEX)
        targets = pad_target_ids.gather(1, (gather + 1).reshape(1, -1).expand(batch, -1)).view(batch, seq, block)
        if labels is not None:
            pad_label_mask = functional.pad(labels, (0, block), value=IGNORE_INDEX)
            label_mask = pad_label_mask.gather(1, gather_flat).view(batch, seq, block) == IGNORE_INDEX
            targets = targets.masked_fill(label_mask, IGNORE_INDEX)

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
                hit_sum = ((logits.argmax(dim=-1) == target_chunk) & chunk_mask).sum()
            confidence = self.confidence_head(hidden_chunk, markov_embed)
            if bool(chunk_mask.any()):
                with torch.no_grad():
                    chunk_hits = (logits.argmax(dim=-1) == target_chunk) & chunk_mask
                bce_sum = functional.binary_cross_entropy_with_logits(
                    confidence[chunk_mask], chunk_hits[chunk_mask].float(), reduction="sum"
                )
            else:
                bce_sum = confidence.sum() * 0.0
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
    "DeepseekV41DSpark",
]
