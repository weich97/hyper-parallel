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
"""Component tests for the trainable DeepSeek-V4.1 DSpark drafter."""

from __future__ import annotations

import copy
import functools
import json
import math
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import mock_open, patch

import torch
from torch import nn

from tests.common.mark_utils import arg_mark
from tests.ut.auto_models.models.deepseek_v41.test_modeling_deepseek_v41 import _config, DeepseekV4Config

from hyper_parallel import DeviceMesh, Replicate, Shard, init_empty_weights
from hyper_parallel.core.dtensor.device_mesh import clear_device_mesh_cache
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.models._transformers.model_builder import _initialize_model_weights
from hyper_parallel.models.deepseek_v41.adapter.validation.cropped_model import build_layer_cropped_config
from hyper_parallel.models.deepseek_v41.dspark import (
    DeepseekV41DSpark, DSparkTokenEmbedding, _DSparkHcCoefficients,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41ForCausalLM, DeepseekV41Model
from hyper_parallel.models.deepseek_v41.adapter.distributed.mega_mhc import DeepseekV41HyperMegaMhc


def _small_config() -> SimpleNamespace:
    """Return a CPU-sized DSpark configuration."""
    return SimpleNamespace(
        hidden_size=32,
        num_attention_heads=4,
        head_dim=16,
        partial_rotary_factor=0.5,
        q_lora_rank=24,
        o_groups=2,
        o_lora_rank=8,
        rms_norm_eps=1.0e-6,
        rope_theta=10000.0,
        v41_dspark_rope_scaling={},
        hc_mult=2,
        hc_sinkhorn_iters=4,
        hc_eps=1.0e-6,
        vocab_size=64,
        initializer_range=0.02,
        v41_dspark_depth=1,
        v41_dspark_block_size=3,
        v41_dspark_window=4,
        v41_dspark_markov_rank=8,
        v41_dspark_noise_token_id=63,
        v41_dspark_target_layer_ids=(1, 2),
        v41_dspark_confidence_coeff=0.1,
        v41_dspark_loss_chunk_size=4,
    )


class _TinyMlp(nn.Module):
    """Minimal MoE stand-in matching the stage FFN calling convention."""

    def __init__(self, hidden: int) -> None:
        """Create the single projection used as the stage FFN."""
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(
            self,
            hidden_states: torch.Tensor,
            input_ids: torch.Tensor | None = None,
            image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project the flattened draft stream like the crop MoE forward."""
        del input_ids, image_mask
        return self.proj(hidden_states)


def _build(config: SimpleNamespace) -> tuple[DeepseekV41DSpark, nn.Linear]:
    """Construct the drafter plus the tied backbone head surface."""
    torch.manual_seed(7)
    dspark = DeepseekV41DSpark(config, mlp_factory=lambda: _TinyMlp(config.hidden_size))
    head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    return dspark, head


def _noise(config):
    """Provide a deterministic frozen embedding row independent of drafter parameters."""
    return torch.linspace(0.01, 0.03, config.hidden_size)


def _batch(config: SimpleNamespace, batch: int = 2, seq: int = 10):
    """Return deterministic inputs with an ignored prefix."""
    generator = torch.Generator().manual_seed(11)
    input_ids = torch.randint(1, config.vocab_size - 1, (batch, seq), generator=generator)
    labels = input_ids.clone()
    labels[:, :3] = IGNORE_INDEX
    targets = [torch.randn(batch, seq, config.hidden_size) for _ in config.v41_dspark_target_layer_ids]
    inputs_embeds = torch.randn(batch, seq, config.hidden_size)
    return input_ids, labels, targets, inputs_embeds


def _flash_v3_torch_reference(query, key_value, sink, query_lengths, kv_lengths, scale):
    """Emulate independent noncausal TND segments at the native API boundary."""
    outputs = []
    q_start = k_start = 0
    for q_end, k_end in zip(query_lengths.tolist(), kv_lengths.tolist(), strict=True):
        q_rows = q_end - q_start
        scores = torch.einsum("thd,kd->thk", query[q_start:q_end], key_value[k_start:k_end, 0]) * scale
        logits = torch.cat((scores, sink[None, :, None].expand(q_rows, -1, 1)), dim=-1)
        weights = logits.float().softmax(-1)[..., :-1].to(query.dtype)
        outputs.append(torch.einsum("thk,kd->thd", weights, key_value[k_start:k_end, 0]))
        q_start, k_start = q_end, k_end
    return torch.cat(outputs)


def _published_frequencies(config, positions):
    """Use the release's complex-valued YaRN formula independently of the adapter."""
    dim = int(config.head_dim * config.partial_rotary_factor)
    frequencies = 1 / config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    settings = config.v41_dspark_rope_scaling
    original = settings.get("original_max_position_embeddings", 0)
    if original:
        low = max(math.floor(dim * math.log(original / (settings["beta_fast"] * 2 * math.pi))
                             / (2 * math.log(config.rope_theta))), 0)
        high = min(math.ceil(dim * math.log(original / (settings["beta_slow"] * 2 * math.pi))
                             / (2 * math.log(config.rope_theta))), dim - 1)
        ramp = ((torch.arange(dim // 2) - low) / max(high - low, 0.001)).clamp(0, 1)
        frequencies = frequencies / settings["factor"] * ramp + frequencies * (1 - ramp)
    angles = torch.outer(torch.arange(positions), frequencies)
    return torch.polar(torch.ones_like(angles), angles)


def _published_rotation(tensor, frequencies, inverse=False):
    """Apply adjacent complex-pair rotation, as in the publisher's inference code."""
    complex_input = torch.view_as_complex(tensor.float().unflatten(-1, (-1, 2)).contiguous())
    shape = (1, frequencies.shape[0], *([1] * (tensor.ndim - 3)), frequencies.shape[1])
    rotation = frequencies.conj() if inverse else frequencies
    return torch.view_as_real(complex_input * rotation.view(shape)).flatten(-2).to(tensor.dtype)


def _published_draft_attention(module, draft, main, frequencies):
    """Decode each anchor independently with its main window and complete draft block.

    This uses the released DSparkAttention equations in FP32/BF16 training
    precision; it does not import the production window or rotary helpers.
    """
    batch, sequence, block, _ = draft.shape
    rope = module.rope_dim
    main_kv = module.kv_norm(module.wkv(main))
    main_kv = torch.cat((main_kv[..., :-rope],
                         _published_rotation(main_kv[..., -rope:], frequencies[:sequence])), dim=-1)
    outputs = []
    for anchor in range(sequence):
        states = draft[:, anchor]
        positions = frequencies[anchor + 1:anchor + 1 + block]
        query = module.wq_b(module.q_norm(module.wq_a(states))).view(
            batch, block, module.num_heads, module.head_dim)
        query = torch.cat((query[..., :-rope], _published_rotation(query[..., -rope:], positions)), dim=-1)
        draft_kv = module.kv_norm(module.wkv(states))
        draft_kv = torch.cat((draft_kv[..., :-rope],
                              _published_rotation(draft_kv[..., -rope:], positions)), dim=-1)
        keys = torch.cat((main_kv[:, max(0, anchor - module.window + 1):anchor + 1], draft_kv), dim=1)
        logits = torch.einsum("bqhd,bkd->bhqk", query, keys) * module.scaling
        sink = module.attn_sink.view(1, -1, 1, 1).expand(batch, -1, block, 1)
        probabilities = torch.cat((logits, sink), dim=-1).float().softmax(-1)[..., :-1].to(query.dtype)
        result = torch.einsum("bhqk,bkd->bqhd", probabilities, keys)
        result = torch.cat((result[..., :-rope],
                            _published_rotation(result[..., -rope:], positions, inverse=True)), dim=-1)
        grouped = result.reshape(batch, block, module.o_groups, -1)
        weight = module.wo_a.weight.view(module.o_groups, module.o_lora_rank, -1)
        outputs.append(module.wo_b(torch.einsum("bqgd,grd->bqgr", grouped, weight).flatten(2)))
    return torch.stack(outputs, dim=1)


class DSparkPublishedAttentionTest(unittest.TestCase):
    """Check the training adapter against independent released decode equations."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_draft_block_is_noncausal_but_main_context_is_causal(self) -> None:
        """Feature: Distinct main-context and intra-draft visibility.

        Description: Backpropagate from the first draft output at one anchor.
        Expectation: Later draft KV receives gradients; future main tokens and other draft blocks do not.
        """
        config = _small_config()
        drafter, _ = _build(config)
        draft = torch.randn(1, 6, 3, config.hidden_size).requires_grad_()
        main = torch.randn(1, 6, config.hidden_size).requires_grad_()
        cos, sin = drafter._rope_tables(6, torch.device("cpu"), torch.bfloat16)
        self.assertEqual(cos.dtype, torch.float32)
        output = drafter.stages[0].self_attn(draft, main, cos, sin)
        output[:, 2, 0].sum().backward()
        self.assertGreater(draft.grad[:, 2, -1].abs().sum().item(), 0)
        self.assertGreater(main.grad[:, :3].abs().sum().item(), 0)
        torch.testing.assert_close(main.grad[:, 3:], torch.zeros_like(main.grad[:, 3:]), rtol=0, atol=0)
        torch.testing.assert_close(draft.grad[:, :2], torch.zeros_like(draft.grad[:, :2]), rtol=0, atol=0)
        torch.testing.assert_close(draft.grad[:, 3:], torch.zeros_like(draft.grad[:, 3:]), rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_draft_context_rotary_positions_and_gradients_match_release(self) -> None:
        """Feature: Published DSpark attention semantics.

        Description: Compare outputs and all gradients across early windows and uneven query tiles.
        Expectation: Draft KV, future positions, shared rotated values and inverse RoPE match the release.
        """
        for chunk in (0, 3):
            with self.subTest(chunk=chunk):
                config = _small_config()
                config.v41_dspark_rope_scaling = {"factor": 4.0, "original_max_position_embeddings": 8,
                                                 "beta_fast": 4.0, "beta_slow": 1.0}
                config.v41_dspark_attention_chunk_size = chunk
                drafter, _ = _build(config)
                actual_module = drafter.stages[0].self_attn
                expected_module = copy.deepcopy(actual_module)
                draft = torch.randn(2, 7, config.v41_dspark_block_size, config.hidden_size).requires_grad_()
                main = torch.randn(2, 7, config.hidden_size).requires_grad_()
                expected_draft = draft.detach().clone().requires_grad_()
                expected_main = main.detach().clone().requires_grad_()
                frequencies = _published_frequencies(config, 7 + config.v41_dspark_block_size)
                expected = _published_draft_attention(expected_module, expected_draft, expected_main, frequencies)
                cos, sin = drafter._rope_tables(7, torch.device("cpu"), torch.float32)
                actual = actual_module(draft, main, cos, sin)
                torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-6)
                cotangent = torch.randn_like(expected)
                actual.backward(cotangent)
                expected.backward(cotangent)
                torch.testing.assert_close(draft.grad, expected_draft.grad, rtol=3.0e-5, atol=3.0e-6)
                torch.testing.assert_close(main.grad, expected_main.grad, rtol=3.0e-5, atol=3.0e-6)
                for (name, parameter), (_, reference) in zip(
                        actual_module.named_parameters(), expected_module.named_parameters(), strict=True):
                    torch.testing.assert_close(parameter.grad, reference.grad, rtol=5.0e-5, atol=5.0e-6, msg=name)


class _TokenInjection(torch.nn.Module):
    """Observe Engram's call boundary with a deterministic token-dependent injection."""

    def __init__(self, config: DeepseekV4Config) -> None:
        """Create a small differentiable injection without external hash-table assets."""
        super().__init__()
        self.embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.calls = 0

    def forward(self, hidden: torch.Tensor, input_ids: torch.Tensor,
                _segment_starts: torch.Tensor | None, *, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Count the injection and add token features to each residual stream."""
        self.calls += 1
        value = self.embedding(input_ids)
        if token_mask is not None:
            value = value * token_mask.unsqueeze(-1)
        return hidden + value.unsqueeze(2)


class DSparkComponentTest(unittest.TestCase):
    """Validate structure, gradient isolation, and target alignment."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dspark_taps_pre_attention_stream_mean_after_engram(self) -> None:
        """Feature: Published DSpark backbone feature taps.

        Description: Observe real attention-boundary inputs with ordinary and shifted mHC.
        Expectation: Targets are detached unweighted means after one Engram injection and before attention.
        """
        torch.manual_seed(170)
        config = _config()
        config.num_hidden_layers = 2
        config.layer_types = ["sliding_attention"] * 2
        config.mlp_layer_types = ["moe"] * 2
        config.v41_compress_ratios = [0, 0]
        config.v41_dspark_target_layer_ids = [0, 1]
        with patch("hyper_parallel.models.deepseek_v41.modeling_deepseek_v41.Path.open",
                   mock_open(read_data='{"layer_ids": []}')):
            reference = DeepseekV41Model(config)
        for layer in reference.layers:
            layer.engram = _TokenInjection(config)
        shifted = copy.deepcopy(reference)
        for layer in shifted.layers:
            layer.attn_hc = DeepseekV41HyperMegaMhc(module=layer.attn_hc)
            layer.ffn_hc = DeepseekV41HyperMegaMhc(module=layer.ffn_hc)
        observed = []

        def _record_input(_module, arguments):
            """Capture the reference attention-boundary stream without autograd state."""
            observed.append(arguments[0].detach().mean(dim=2))

        handles = [layer.attn_hc.register_forward_pre_hook(_record_input) for layer in reference.layers]
        try:
            ids = torch.tensor([[1, 2, 3, 4]])
            expected = reference(input_ids=ids).last_hidden_state
            actual = shifted(input_ids=ids).last_hidden_state
        finally:
            for handle in handles:
                handle.remove()
        torch.testing.assert_close(actual, expected)
        self.assertEqual(len(observed), 2)
        for model in (reference, shifted):
            for layer, target, expected_target in zip(model.layers, model.dspark_target_hiddens, observed, strict=True):
                self.assertEqual(layer.engram.calls, 1)
                self.assertFalse(target.requires_grad)
                torch.testing.assert_close(target, expected_target)


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_noise_row_is_shared_frozen_and_refreshed(self) -> None:
        """Feature: Tied backbone token embeddings.

        Description: Refresh the noise row after a backbone weight update and run draft backward.
        Expectation: Noise follows the shared vocabulary row without an extra parameter or backbone gradient.
        """
        config = _small_config()
        drafter, head = _build(config)
        embedding = DSparkTokenEmbedding(config.vocab_size, config.hidden_size,
                                         noise_token_id=config.v41_dspark_noise_token_id)
        ids, labels, targets, _ = _batch(config)
        previous = None
        for iteration in range(2):
            if iteration:
                with torch.no_grad():
                    embedding.weight[config.v41_dspark_noise_token_id].add_(0.25)
            embeds = embedding(ids)
            noise = embedding.cached_noise_embedding
            torch.testing.assert_close(noise, embedding.weight[config.v41_dspark_noise_token_id], rtol=0, atol=0)
            self.assertFalse(noise.requires_grad)
            if previous is not None:
                torch.testing.assert_close(noise, previous + 0.25, rtol=0, atol=0)
            previous = noise.clone()
            loss, _ = drafter(targets, ids, labels, embeds.detach(), head, noise_embedding=noise)
            loss.backward()
            self.assertIsNone(embedding.weight.grad)
        self.assertEqual(set(dict(embedding.named_parameters())), {"weight"})
        self.assertNotIn("noise_embedding", dict(drafter.named_parameters()))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_first_draft_input_and_markov_context_follow_main_context(self) -> None:
        """Feature: Released DSpark decode alignment.

        Description: Observe the actual first-stage input and Markov inputs in teacher forcing.
        Expectation: Main context ending at t sees x[t+1] plus noise, then predicts x[t+2+j].
        """
        config = _small_config()
        drafter, head = _build(config)
        ids = torch.arange(1, 7)[None]
        embeds = torch.arange(6 * config.hidden_size).float().view(1, 6, -1)
        targets = [torch.zeros_like(embeds) for _ in config.v41_dspark_target_layer_ids]
        stage_inputs, markov_inputs = [], []
        handles = [drafter.stages[0].register_forward_pre_hook(
            lambda _module, args: stage_inputs.append(args[0].detach().clone())),
            drafter.markov_head.register_forward_pre_hook(
                lambda _module, args: markov_inputs.append(args[0].detach().clone()))]
        try:
            drafter(targets, ids, None, embeds, head, noise_embedding=_noise(config))
        finally:
            for handle in handles:
                handle.remove()
        actual = stage_inputs[0][..., 0, :]
        torch.testing.assert_close(actual[:, :-1, 0], embeds[:, 1:], rtol=0, atol=0)
        torch.testing.assert_close(actual[:, -1, 0], _noise(config)[None], rtol=0, atol=0)
        torch.testing.assert_close(actual[:, :, 1:], _noise(config).expand_as(actual[:, :, 1:]), rtol=0, atol=0)
        expected = torch.tensor([[[2, 3, 4], [3, 4, 5], [4, 5, 6],
                                  [5, 6, 63], [6, 63, 63], [63, 63, 63]]])
        torch.testing.assert_close(torch.cat(markov_inputs, dim=1), expected, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_packed_draft_matches_independent_documents_and_gradients(self) -> None:
        """Feature: Packed-document isolation.

        Description: Compare uneven packed attention tiles and losses with two independent documents.
        Expectation: Position reset, KV windows and teacher-forcing boundaries preserve loss and every gradient.
        """
        for backend in ("reference", "flash_v3"):
            with self.subTest(backend=backend):
                config = _small_config()
                reference, head = _build(config)
                config.v41_dspark_attention_backend = backend
                config.v41_dspark_attention_chunk_size = 3
                packed, _ = _build(config)
                ids, _, hiddens, embeds = _batch(config, batch=2, seq=10)
                counts, sums = [], []
                for start, stop in ((0, 4), (4, 10)):
                    loss, metrics = reference([value[:, start:stop] for value in hiddens],
                                              ids[:, start:stop], None, embeds[:, start:stop], head,
                                              noise_embedding=_noise(config))
                    count = metrics["dspark_supervised_tokens"]
                    counts.append(count)
                    sums.append(loss * count)
                expected = sum(sums) / sum(counts)
                with patch("hyper_parallel.models.deepseek_v41.dspark._flash_attention_v3",
                           side_effect=_flash_v3_torch_reference):
                    actual, metrics = packed(hiddens, ids, None, embeds, head,
                                             noise_embedding=_noise(config), sequence_boundaries=(0, 4, 10))
                    actual.backward()
                expected.backward()
                torch.testing.assert_close(actual, expected, rtol=2.0e-6, atol=1.0e-7)
                self.assertEqual(metrics["dspark_supervised_tokens"], sum(counts))
                for (name, left), (_, right) in zip(
                        reference.named_parameters(), packed.named_parameters(), strict=True):
                    torch.testing.assert_close(right.grad, left.grad, rtol=4.0e-5, atol=3.0e-7, msg=name)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_draft_targets_do_not_cross_visual_spans_or_document_ends(self) -> None:
        """Feature: Multimodal and packed supervision masks.

        Description: Place a visual token inside a text draft and a document boundary near its end.
        Expectation: Only uninterrupted text prefixes inside each document receive draft supervision.
        """
        config = _small_config()
        drafter, _ = _build(config)
        ids = torch.arange(1, 11)[None]
        text = torch.tensor([[True, True, True, False, True, True, True, True, True, True]])
        labels = torch.zeros_like(ids)
        labels[:, 5] = IGNORE_INDEX
        _, actual = drafter._teacher_forced_targets(ids, labels, text, (0, 7, 10))
        expected = torch.full((1, 10, 3), IGNORE_INDEX)
        expected[0, 0, 0] = 3
        expected[0, 3, 0] = 6
        expected[0, 7, 0] = 10
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        hiddens = [torch.randn(1, 10, config.hidden_size) for _ in config.v41_dspark_target_layer_ids]
        embeds = torch.randn(1, 10, config.hidden_size)
        head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        loss, metrics = drafter(hiddens, ids, None, embeds, head, noise_embedding=_noise(config),
                                token_mask=torch.zeros_like(text))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(metrics["dspark_supervised_tokens"].item(), 0)
        for parameter in drafter.parameters():
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter.grad), rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_hc_coefficients_use_unweighted_published_normalization(self) -> None:
        """Feature: Published DSpark hyper-connection normalization.

        Description: Compare coefficients and gradients with the FP32 projection/RMS formula.
        Expectation: The norm uses norm epsilon and has no additional trainable affine weight.
        """
        for epsilon in (1.0e-6, 1.0e-20):
            with self.subTest(epsilon=epsilon):
                config = _small_config()
                config.rms_norm_eps = epsilon
                module = _DSparkHcCoefficients(config)
                generator = torch.Generator().manual_seed(31)
                hidden = (torch.randn(2, 3, config.hc_mult, config.hidden_size, generator=generator)
                          * 1.0e-4).requires_grad_()
                reference = hidden.detach().clone().requires_grad_()
                weight = module.get_parameter("fn").detach().clone().requires_grad_()
                flat = reference.flatten(2)
                projected = (flat @ weight.T) / (flat.square().mean(-1, keepdim=True) + epsilon).sqrt()
                expected_pre = projected[..., :config.hc_mult].sigmoid() + config.hc_eps
                expected_post = 2 * projected[..., config.hc_mult:2 * config.hc_mult].sigmoid()
                actual_pre, actual_post, _ = module(hidden)
                torch.testing.assert_close(actual_pre, expected_pre, rtol=1.0e-6, atol=1.0e-7)
                torch.testing.assert_close(actual_post, expected_post, rtol=1.0e-6, atol=1.0e-7)
                (actual_pre.square().sum() + actual_post.square().sum()).backward()
                (expected_pre.square().sum() + expected_post.square().sum()).backward()
                torch.testing.assert_close(hidden.grad, reference.grad, rtol=2.0e-4, atol=2.0e-4)
                torch.testing.assert_close(module.fn.grad, weight.grad, rtol=2.0e-4, atol=2.0e-5)
                self.assertEqual(set(dict(module.named_parameters())), {"fn", "base", "scale"})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_confidence_reads_features_before_output_normalization(self) -> None:
        """Feature: Separate vocabulary and confidence feature branches.

        Description: Observe a full drafter call and backpropagate confidence loss alone.
        Expectation: Confidence receives the raw merged stream and never trains output RMSNorm.
        """
        config = _small_config()
        dspark, head = _build(config)
        with torch.no_grad():
            dspark.norm.weight.copy_(torch.linspace(0.5, 2.0, config.hidden_size))
        stage_outputs, confidence_inputs = [], []

        def _record_stage(_module, _inputs, output):
            """Keep the final residual streams without retaining their autograd graph."""
            stage_outputs.append(tuple(value.detach().clone() for value in output))

        def _record_confidence(_module, inputs):
            """Observe the actual confidence-head features across objective chunks."""
            confidence_inputs.append(inputs[0].detach().clone())

        handles = [dspark.stages[-1].register_forward_hook(_record_stage),
                   dspark.confidence_head.register_forward_pre_hook(_record_confidence)]
        ids, labels, hiddens, embeds = _batch(config)
        try:
            with patch("hyper_parallel.models.deepseek_v41.dspark.functional.cross_entropy",
                       side_effect=lambda logits, *args, **kwargs: logits.sum() * 0.0):
                loss, _ = dspark(hiddens, ids, labels, embeds, head, noise_embedding=_noise(config))
                loss.backward()
        finally:
            for handle in handles:
                handle.remove()
        streams, pre_mix = stage_outputs[-1]
        expected = (streams.reshape(ids.shape[0], -1, config.hc_mult, config.hidden_size).float()
                    * pre_mix.unsqueeze(-1)).sum(2).to(streams.dtype)
        expected = expected.view(ids.shape[0], ids.shape[1], config.v41_dspark_block_size, config.hidden_size)
        torch.testing.assert_close(torch.cat(confidence_inputs, dim=1), expected, rtol=0, atol=0)
        torch.testing.assert_close(dspark.norm.weight.grad, torch.zeros_like(dspark.norm.weight), rtol=0, atol=0)
        self.assertGreater(dspark.confidence_head.proj.weight.grad.abs().sum().item(), 0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_initialization_preserves_logical_weights_across_shards_and_replicas(self):
        """Initialize the same global parameters under replicated and uneven layouts.

        Feature: DSpark bare-parameter initialization after FSDP wrapping.
        Description: Compare every local shard with its independently initialized global tensor.
        Expectation: Changing the shard count or replica rank preserves all logical parameter values.
        """
        clear_device_mesh_cache()
        self.addCleanup(clear_device_mesh_cache)
        config = _small_config()
        reference, _ = _build(config)
        reference.init_weights(config.initializer_range)
        parameter_names = ("stages.0.attn_hc.fn", "stages.0.ffn_hc.fn")
        for shards in (1, 2, 3, 16):
            for rank in range(2 * shards):
                with self.subTest(shards=shards, rank=rank), \
                        patch.dict("hyper_parallel.core.dtensor.dtensor._LAYOUT_CACHE", {}, clear=True), \
                        patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=rank), \
                        patch("hyper_parallel.models.deepseek_v41.dspark.torch.distributed.is_initialized",
                              return_value=True):
                    clear_device_mesh_cache()
                    mesh = DeviceMesh("cpu", torch.arange(2 * shards).reshape(2, shards),
                                      mesh_dim_names=("replicate", "shard"), _init_backend=False)
                    candidate, _ = _build(config)
                    for name in parameter_names:
                        parameter = candidate.get_parameter(name)
                        placement = Shard(0, uneven_shard=parameter.shape[0] % shards != 0)
                        distributed = distribute_tensor(torch.empty_like(parameter), mesh, (Replicate(), placement))
                        module_name, _, field = name.rpartition(".")
                        setattr(candidate.get_submodule(module_name), field, nn.Parameter(distributed))

                    candidate.init_weights(config.initializer_range)

                    for name in parameter_names:
                        full = reference.get_parameter(name)
                        chunks = torch.chunk(full, shards, dim=0)
                        shard_index = rank % shards
                        expected = chunks[shard_index] if shard_index < len(chunks) else full[:0]
                        actual = candidate.get_parameter(name).to_local()
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0,
                                                   msg=f"Parameter {name}, shards={shards}, rank={rank}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_native_model_initialization_reaches_drafter_parameter_owners(self) -> None:
        """Feature: DSpark initialization through the native model lifecycle.

        Description: Materialize a meta model whose drafter has no direct parameters.
        Expectation: Native initialization fills every bare coefficient and attention sink.
        """
        config = _config()
        for name, value in vars(_small_config()).items():
            if name.startswith("v41_dspark_"):
                setattr(config, name, value)
        config.v41_dspark_depth = 2
        config.v41_dspark_noise_token_id = config.vocab_size - 1
        config.v41_dspark_target_layer_ids = [0]
        with patch("hyper_parallel.models.deepseek_v41.modeling_deepseek_v41.Path.open",
                   mock_open(read_data='{"layer_ids": []}')), init_empty_weights():
            model = DeepseekV41ForCausalLM(config)
        model.to_empty(device="cpu")
        with torch.no_grad():
            for parameter in model.dspark.parameters():
                parameter.fill_(float("nan"))
        self.assertEqual(list(model.dspark.parameters(recurse=False)), [])

        _initialize_model_weights(model)

        for index, stage in enumerate(model.dspark.stages):
            for owner, seed in ((stage.attn_hc, 0x44530100 + index),
                                (stage.ffn_hc, 0x44530200 + index)):
                expected = torch.randn(owner.fn.shape, generator=torch.Generator().manual_seed(seed))
                expected = expected * config.initializer_range
                torch.testing.assert_close(owner.fn, expected, rtol=0, atol=0)
                torch.testing.assert_close(owner.base, torch.zeros_like(owner.base), rtol=0, atol=0)
                torch.testing.assert_close(owner.scale, torch.ones_like(owner.scale), rtol=0, atol=0)
            torch.testing.assert_close(stage.self_attn.attn_sink, torch.zeros_like(stage.self_attn.attn_sink),
                                       rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_attention_query_tiles_keep_cross_tile_causal_context(self):
        """Feature: Query chunking.

        Description: Compare full drafter loss and gradients with an uneven query tiling.
        Expectation: Loss and every parameter gradient match the unchunked reference.
        """
        config = _small_config()
        reference, head = _build(config)
        config.v41_dspark_attention_chunk_size = 3
        candidate, _ = _build(config)
        ids, labels, hiddens, embeds = _batch(config)
        expected, _ = reference(hiddens, ids, labels, embeds, head, noise_embedding=_noise(config))
        actual, _ = candidate(hiddens, ids, labels, embeds, head, noise_embedding=_noise(config))
        torch.testing.assert_close(actual, expected, rtol=2.0e-6, atol=1.0e-7)
        expected.backward()
        actual.backward()
        for (name, left), (_, right) in zip(reference.named_parameters(), candidate.named_parameters(), strict=True):
            torch.testing.assert_close(right.grad, left.grad, rtol=2.0e-5, atol=1.0e-7, msg=name)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_flash_attention_adapter_preserves_draft_offsets_and_gradients(self):
        """Feature: Native attention adapter.

        Description: Validate batch isolation and nonzero tile starts against full reference attention.
        Expectation: Offset, batch and gradient results match the portable reference.
        """
        for chunk in (0, 3):
            with self.subTest(chunk=chunk):
                config = _small_config()
                reference, head = _build(config)
                config.v41_dspark_attention_backend = "flash_v3"
                config.v41_dspark_attention_chunk_size = chunk
                candidate, _ = _build(config)
                ids, labels, hiddens, embeds = _batch(config)
                expected, _ = reference(hiddens, ids, labels, embeds, head, noise_embedding=_noise(config))
                with patch("hyper_parallel.models.deepseek_v41.dspark._flash_attention_v3",
                           side_effect=_flash_v3_torch_reference) as native:
                    actual, _ = candidate(hiddens, ids, labels, embeds, head, noise_embedding=_noise(config))
                    actual.backward()
                self.assertGreater(native.call_count, 0)
                expected.backward()
                torch.testing.assert_close(actual, expected, rtol=2.0e-6, atol=1.0e-7)
                for (name, left), (_, right) in zip(
                        reference.named_parameters(), candidate.named_parameters(), strict=True):
                    torch.testing.assert_close(right.grad, left.grad, rtol=3.0e-5, atol=1.0e-7, msg=name)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_forward_produces_finite_losses_and_metrics(self):
        """Feature: Drafter forward contract.

        Description: A full forward returns finite loss values and metric entries.
        Expectation: All declared loss and metric entries are present and finite.
        """
        config = _small_config()
        dspark, head = _build(config)
        input_ids, labels, targets, embeds = _batch(config)
        loss, metrics = dspark(targets, input_ids, labels, embeds, head, noise_embedding=_noise(config))
        self.assertTrue(torch.isfinite(loss).item(), "DSpark loss must be finite")
        for name in ("dspark_ce", "dspark_confidence_bce", "dspark_draft_accuracy",
                     "dspark_supervised_tokens"):
            self.assertIn(name, metrics, f"missing metric {name}")
            self.assertTrue(torch.isfinite(metrics[name].float()).item(), f"{name} must be finite")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_requires_detached_target_hiddens(self):
        """Feature: Backbone gradient isolation.

        Description: Grad-carrying backbone hiddens are rejected up front.
        Expectation: Gradient-carrying target features raise a descriptive error.
        """
        config = _small_config()
        dspark, head = _build(config)
        input_ids, labels, targets, embeds = _batch(config)
        targets[0] = targets[0].clone().requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "detached"):
            dspark(targets, input_ids, labels, embeds, head, noise_embedding=_noise(config))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_gradients_stay_inside_the_drafter(self):
        """Feature: Drafter parameter ownership.

        Description: The DSpark objective must not reach tied backbone surfaces.
        Expectation: Drafter components receive gradients while the tied backbone head remains frozen.
        """
        config = _small_config()
        dspark, head = _build(config)
        input_ids, labels, targets, embeds = _batch(config)
        loss, _ = dspark(targets, input_ids, labels, embeds, head, noise_embedding=_noise(config))
        loss.backward()
        self.assertIsNone(head.weight.grad, "tied head must stay frozen under DSpark loss")
        touched = [name for name, parameter in dspark.named_parameters()
                   if parameter.grad is not None and parameter.grad.abs().sum() > 0]
        self.assertTrue(any(name.startswith("stages.0.self_attn") for name in touched))
        self.assertTrue(any(name.startswith("main_proj") for name in touched))
        self.assertTrue(any(name.startswith("markov_head") for name in touched))
        self.assertTrue(any(name.startswith("confidence_head") for name in touched))
        self.assertNotIn("noise_embedding", dict(dspark.named_parameters()))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_supervised_targets_align_with_future_tokens(self):
        """Feature: Configured teacher-forcing alignment.

        Description: Draft offset j at position t must supervise the token at t+2+j.
        Expectation: Only in-range future tokens contribute to the supervision count.
        """
        config = _small_config()
        config.v41_dspark_block_size = 2
        dspark, head = _build(config)
        batch, seq = 1, 6
        input_ids = torch.arange(1, seq + 1).unsqueeze(0)
        labels = input_ids.clone()
        targets = [torch.zeros(batch, seq, config.hidden_size) for _ in config.v41_dspark_target_layer_ids]
        embeds = torch.zeros(batch, seq, config.hidden_size)
        _, metrics = dspark(targets, input_ids, labels, embeds, head, noise_embedding=_noise(config))
        # Main context ends at t; the known draft input is t+1, so
        # offsets predict t+2 and t+3 (four and three valid positions).
        self.assertEqual(int(metrics["dspark_supervised_tokens"]), (seq - 2) + (seq - 3))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_shifted_labels_keep_targets_from_input_ids(self):
        """Feature: Trainer label masking.

        Description: Pre-shifted trainer labels must only mask, never shift the targets.
        Expectation: Cross-entropy targets use input token values with the declared shifted label mask.


        The trainer hands labels over already shifted (labels[t] supervises
        input_ids[t + 1]). Main KV ends at t, the known draft input is t+1,
        and draft offset j predicts input_ids[t + 2 + j]. Labels only mask
        this target through labels[t + 1 + j].
        """
        config = _small_config()
        config.v41_dspark_block_size = 2
        dspark, head = _build(config)
        batch, seq = 1, 6
        input_ids = torch.arange(1, seq + 1).unsqueeze(0)
        labels = torch.cat(
            [input_ids[:, 1:], torch.full((batch, 1), IGNORE_INDEX)], dim=1)
        hiddens = [torch.zeros(batch, seq, config.hidden_size)
                   for _ in config.v41_dspark_target_layer_ids]
        embeds = torch.zeros(batch, seq, config.hidden_size)
        original = torch.nn.functional.cross_entropy
        captured = []

        def _spy(logits, target, **kwargs):
            captured.append(target.detach().clone())
            return original(logits, target, **kwargs)

        with patch("hyper_parallel.models.deepseek_v41.dspark.functional.cross_entropy",
                   side_effect=_spy):
            _, metrics = dspark(hiddens, input_ids, labels, embeds, head, noise_embedding=_noise(config))
        got = torch.cat(captured).view(batch, seq, 2)
        expected = torch.tensor([[[3, 4], [4, 5], [5, 6], [6, IGNORE_INDEX],
                                  [IGNORE_INDEX, IGNORE_INDEX],
                                  [IGNORE_INDEX, IGNORE_INDEX]]])
        self.assertTrue(torch.equal(got, expected),
                        msg=f"draft targets misaligned: {got.tolist()}")
        self.assertEqual(int(metrics["dspark_supervised_tokens"]), 7)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_markov_head_biases_the_draft_logits(self):
        """Feature: Markov-head engagement.

        Description: Zeroing the Markov head must change the draft cross-entropy.
        Expectation: Removing the Markov bias changes draft cross-entropy.
        """
        config = _small_config()
        dspark, head = _build(config)
        input_ids, labels, targets, embeds = _batch(config)
        with torch.no_grad():
            baseline, _ = dspark(targets, input_ids, labels, embeds, head, noise_embedding=_noise(config))
            dspark.markov_head.head.weight.mul_(0.0)
            ablated, _ = dspark(targets, input_ids, labels, embeds, head, noise_embedding=_noise(config))
        self.assertNotAlmostEqual(float(baseline), float(ablated), places=6,
                                  msg="Markov bias must influence the draft objective")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_none_labels_supervise_from_input_ids(self):
        """Feature: Unmasked draft objective.

        Description: Without labels the drafter supervises all next tokens.
        Expectation: Missing labels retain all valid future-token targets.
        """
        config = _small_config()
        config.v41_dspark_block_size = 2
        dspark, head = _build(config)
        batch, seq = 1, 6
        input_ids = torch.arange(1, seq + 1).unsqueeze(0)
        targets = [torch.zeros(batch, seq, config.hidden_size) for _ in config.v41_dspark_target_layer_ids]
        embeds = torch.zeros(batch, seq, config.hidden_size)
        _, metrics = dspark(targets, input_ids, None, embeds, head, noise_embedding=_noise(config))
        self.assertEqual(int(metrics["dspark_supervised_tokens"]), (seq - 2) + (seq - 3))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_zero_depth_configuration_is_rejected(self):
        """Feature: Drafter construction validation.

        Description: Constructing with no stages/targets is a configuration error.
        Expectation: An invalid direct drafter configuration fails before execution.
        """
        config = _small_config()
        config.v41_dspark_block_size = 0
        with self.assertRaises(ValueError):
            DeepseekV41DSpark(config, mlp_factory=lambda: _TinyMlp(config.hidden_size))


def _write_engram_assets(
        directory: str,
        num_hidden_layers: int = 4,
        layer_ids: tuple[int, ...] = (1,),
        head_dim: int = 4,
) -> Path:
    """Write a minimal but internally consistent scaled Engram asset."""
    available_primes = (
        [[17, 19], [23, 29]],
        [[31, 37], [41, 43]],
    )
    if len(layer_ids) > len(available_primes):
        raise ValueError("the test asset helper supports at most two Engram layers")
    primes = list(available_primes[:len(layer_ids)])
    assets = {
        "source_model_type": "deepseek_v41",
        "num_hidden_layers": num_hidden_layers,
        "layer_ids": list(layer_ids),
        "bucket_base": 16,
        "max_ngram_size": 3,
        "num_heads": 2,
        "head_dim": head_dim,
        "primes": primes,
        "num_embeddings": [
            sum(value for row in layer_primes for value in row)
            for layer_primes in primes
        ],
        "multipliers": [[101, 103, 107] for _ in layer_ids],
        "token_map": list(range(64)),
        "pad_token_id": 0,
    }
    path = Path(directory) / "engram.json"
    path.write_text(json.dumps(assets), encoding="utf-8")
    return path


def _write_released_config(directory: str) -> Path:
    """Write a release-shaped config whose dimensions have exact crop ratios."""
    text_config = {
        "vocab_size": 64,
        "hidden_size": 256,
        "moe_intermediate_size": 128,
        "num_hidden_layers": 40,
        # Drafter hyper-parameters as published in DeepSeek-V4.1-Flash.
        "num_nextn_predict_layers": 3,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 63,
        "dspark_target_layer_ids": [37, 38, 39],
        "dspark_markov_rank": 256,
        "dspark_n_routed_experts": 128,
        "dspark_num_experts_per_tok": 3,
        "num_attention_heads": 32,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "qk_rope_head_dim": 32,
        "q_lora_rank": 128,
        "o_lora_rank": 128,
        "o_groups": 8,
        "hidden_act": "silu",
        "swiglu_limit": 10.0,
        "rms_norm_eps": 1.0e-6,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "tie_word_embeddings": False,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "n_routed_experts": 384,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "scoring_func": "sqrtsoftplus",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        "sliding_window": 128,
        "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
        "compress_rope_theta": 160000.0,
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "index_n_heads": 32,
        "index_head_dim": 64,
        "index_topk": 512,
        "candidate_source_layer_id": 20,
        "candidate_topk_blocks": 2048,
        "candidate_block_size": 8,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1.0e-6,
        "engram_layer_ids": [1, 14],
        "engram_head_dim": 32,
    }
    source = {
        "model_type": "deepseek_v41",
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "image_token_id": 3,
        "text_config": text_config,
        "vision_config": {
            "num_hidden_layers": 32,
            "hidden_size": 128,
            "num_attention_heads": 16,
            "intermediate_size": 192,
            "patch_size": 14,
            "rope_theta": 10000.0,
            "downsample_ratio": 3,
            "max_image_tokens": 1024,
            "min_pixels": 295936,
            "max_wh_ratio": None,
        },
    }
    path = Path(directory) / "config.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


# Leaf parameter names of the released DeepSeek-V4.1-Flash drafter, taken from
# the mtp.* entries of model.safetensors.index.json (2401 entries, three
# stages). Stage and expert indices are collapsed to "N".
_RELEASED_DRAFTER_LEAVES = {
    "hc_attn_base", "hc_attn_fn", "hc_attn_scale",
    "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale",
    "attn_norm.weight", "ffn_norm.weight",
    "attn.attn_sink", "attn.kv_norm.weight", "attn.q_norm.weight",
    "attn.wkv.weight", "attn.wo_a.weight", "attn.wo_b.weight",
    "attn.wq_a.weight", "attn.wq_b.weight",
    "ffn.gate.weight", "ffn.gate.bias", "ffn.gate.bias_vl",
    "ffn.shared_experts.w1.weight", "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w3.weight",
    "ffn.experts.N.w1.weight", "ffn.experts.N.w2.weight", "ffn.experts.N.w3.weight",
    "main_norm.weight", "main_proj.weight", "norm.weight",
    "markov_head.embed.weight", "markov_head.head.weight",
    "confidence_head.proj.weight",
}

# Released leaf -> our leaf. Only unambiguous correspondences are listed; the
# rest are recorded as known differences below.
_DRAFTER_RENAMES = {
    "hc_attn_base": "stages.N.attn_hc.base",
    "hc_attn_fn": "stages.N.attn_hc.fn",
    "hc_attn_scale": "stages.N.attn_hc.scale",
    "hc_ffn_base": "stages.N.ffn_hc.base",
    "hc_ffn_fn": "stages.N.ffn_hc.fn",
    "hc_ffn_scale": "stages.N.ffn_hc.scale",
    "attn_norm.weight": "stages.N.input_layernorm.weight",
    "ffn_norm.weight": "stages.N.post_attention_layernorm.weight",
    "attn.attn_sink": "stages.N.self_attn.attn_sink",
    "attn.kv_norm.weight": "stages.N.self_attn.kv_norm.weight",
    "attn.q_norm.weight": "stages.N.self_attn.q_norm.weight",
    "attn.wkv.weight": "stages.N.self_attn.wkv.weight",
    "attn.wo_a.weight": "stages.N.self_attn.wo_a.weight",
    "attn.wo_b.weight": "stages.N.self_attn.wo_b.weight",
    "attn.wq_a.weight": "stages.N.self_attn.wq_a.weight",
    "attn.wq_b.weight": "stages.N.self_attn.wq_b.weight",
    "ffn.gate.weight": "stages.N.mlp.gate.weight",
    "ffn.gate.bias": "stages.N.mlp.gate.bias",
    "ffn.gate.bias_vl": "stages.N.mlp.gate.bias_vl",
    "ffn.shared_experts.w1.weight": "stages.N.mlp.shared_experts.gate_proj.weight",
    "ffn.shared_experts.w2.weight": "stages.N.mlp.shared_experts.down_proj.weight",
    "ffn.shared_experts.w3.weight": "stages.N.mlp.shared_experts.up_proj.weight",
    "main_norm.weight": "main_norm.weight",
    "main_proj.weight": "main_proj.weight",
    "norm.weight": "norm.weight",
    "markov_head.embed.weight": "markov_head.embed.weight",
    "markov_head.head.weight": "markov_head.head.weight",
    "confidence_head.proj.weight": "confidence_head.proj.weight",
}

# Released names with no 1:1 counterpart here, and why.
_RELEASED_ONLY = {
    # The release stores 128 experts per stage separately; the crop MoE packs
    # them into two grouped tensors.
    "ffn.experts.N.w1.weight", "ffn.experts.N.w2.weight", "ffn.experts.N.w3.weight",
}

# Ours with no counterpart in the released weights, and why.
_OURS_ONLY = {
    # Grouped expert tensors, see above.
    "stages.N.mlp.experts.gate_up_proj", "stages.N.mlp.experts.down_proj",
}


class TestDrafterMatchesReleasedLayout(unittest.TestCase):
    """The drafter's parameters correspond to the released mtp.* layout.

    FP8 ``.scale`` companions are out of scope for bf16 training and are
    excluded; every other released name must map onto a parameter here, and
    every parameter here must be accounted for.
    """

    @staticmethod
    def _drafter_leaves() -> set:
        """Build the drafter on meta and collapse index positions to "N"."""
        from hyper_parallel.models.deepseek_v41.adapter.validation.cropped_model import (  # pylint: disable=C0415
            build_deepseek_v41_validation_config,
        )
        with tempfile.TemporaryDirectory() as directory:
            config_path = _write_released_config(directory)
            assets = _write_engram_assets(directory, num_hidden_layers=40, head_dim=8)
            config = build_deepseek_v41_validation_config(
                str(config_path.parent), str(assets), dspark_depth=3, enable_vision=True)
            with init_empty_weights():
                model = DeepseekV41ForCausalLM(config)
            return {re.sub(r"\.\d+\.", ".N.", name)
                    for name, _ in model.dspark.named_parameters()}

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_every_released_name_is_accounted_for(self):
        """Feature: Released parameter inventory.

        Description: No released parameter family is silently missing.
        Expectation: Every released leaf maps to a local parameter or an explicit remaining difference.
        """
        ours = self._drafter_leaves()
        for released in sorted(_RELEASED_DRAFTER_LEAVES):
            with self.subTest(released=released):
                if released in _RELEASED_ONLY:
                    continue
                self.assertIn(_DRAFTER_RENAMES[released], ours)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_no_undocumented_parameters_here(self):
        """Feature: Local parameter inventory.

        Description: Anything we add beyond the released layout stays documented.
        Expectation: All additional local parameters have a recorded correspondence or difference.
        """
        ours = self._drafter_leaves()
        mapped = set(_DRAFTER_RENAMES.values())
        self.assertEqual(sorted(ours - mapped - _OURS_ONLY), [])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_hyperparameters_match_the_release(self):
        """Feature: Released drafter configuration.

        Description: Depth, block width, expert counts and the noise token come from the release.
        Expectation: Depth, block width, expert configuration and noise token match the fixture.
        """
        with tempfile.TemporaryDirectory() as directory:
            config_path = _write_released_config(directory)
            released = json.loads(config_path.read_text(encoding="utf-8"))["text_config"]
            assets = _write_engram_assets(directory, num_hidden_layers=40, head_dim=8)
            from hyper_parallel.models.deepseek_v41.adapter.validation.cropped_model import (  # pylint: disable=C0415
                build_deepseek_v41_validation_config,
            )
            config = build_deepseek_v41_validation_config(
                str(config_path.parent), str(assets), dspark_depth=3)
        self.assertEqual(config.v41_dspark_block_size, released["dspark_block_size"])
        self.assertEqual(config.v41_dspark_markov_rank, released["dspark_markov_rank"])
        self.assertEqual(config.v41_dspark_noise_token_id, released["dspark_noise_token_id"])
        self.assertEqual(config.v41_dspark_target_layer_ids, released["dspark_target_layer_ids"])
        self.assertEqual(config.v41_dspark_n_routed_experts,
                         min(released["dspark_n_routed_experts"], 16))


class TestDSparkCropSwitches(unittest.TestCase):
    """The crop entry carries the DSpark drafter switches into the config."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_released_yarn_settings_survive_the_v4_config_translation(self) -> None:
        """Feature: Released drafter rotary configuration.

        Description: Translate flat source YaRN parameters through V4's nested main/compress representation.
        Expectation: DSpark retains the source YaRN settings independently of V4's default main branch.
        """
        from hyper_parallel.models.deepseek_v41.adapter.validation.cropped_model import (  # pylint: disable=C0415
            build_deepseek_v41_validation_config,
        )
        settings = {"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 65536,
                    "beta_fast": 32, "beta_slow": 1}
        with tempfile.TemporaryDirectory() as directory:
            path = _write_released_config(directory)
            source = json.loads(path.read_text(encoding="utf-8"))
            source["text_config"]["rope_scaling"] = settings
            source["text_config"]["max_position_embeddings"] = 1048576
            path.write_text(json.dumps(source), encoding="utf-8")
            assets = _write_engram_assets(directory, num_hidden_layers=40, head_dim=8)
            config = build_deepseek_v41_validation_config(str(path.parent), str(assets), dspark_depth=1)
        self.assertEqual(config.v41_dspark_rope_scaling, settings)
        self.assertEqual(config.rope_parameters["main"]["rope_type"], "default")
        drafter = DeepseekV41DSpark(config, lambda: _TinyMlp(config.hidden_size))
        cos, sin = drafter._rope_tables(7, torch.device("cpu"), torch.bfloat16)
        expected = _published_frequencies(config, 7 + config.v41_dspark_block_size)
        torch.testing.assert_close(cos[0], expected.real, rtol=1.0e-7, atol=1.0e-7)
        torch.testing.assert_close(sin[0], expected.imag, rtol=1.0e-7, atol=1.0e-7)

    @staticmethod
    def _build(**kwargs):
        """Build a validation config from the release-shaped fixtures."""
        with tempfile.TemporaryDirectory() as directory:
            config_path = _write_released_config(directory)
            assets = _write_engram_assets(directory, num_hidden_layers=40, head_dim=8)
            from hyper_parallel.models.deepseek_v41.adapter.validation.cropped_model import (  # pylint: disable=C0415
                build_deepseek_v41_validation_config,
            )
            return build_deepseek_v41_validation_config(
                str(config_path.parent), str(assets), **kwargs)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_drafter_is_off_by_default(self):
        """Feature: Optional drafter configuration.

        Description: No drafter parameters are requested unless depth is positive.
        Expectation: The default configuration does not enable drafter parameters.
        """
        config = self._build()
        self.assertEqual(config.v41_dspark_depth, 0)
        self.assertEqual(config.v41_dspark_target_layer_ids, [])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_attention_backend_is_explicit_and_validated(self):
        """Feature: Attention backend selection.

        Description: Keep the reference default and reject unrecognized native backends.
        Expectation: Supported backends are retained and unknown names are rejected.
        """
        self.assertEqual(self._build().v41_dspark_attention_backend, "reference")
        config = self._build(dspark_depth=1, dspark_attention_backend="flash_v3")
        self.assertEqual(config.v41_dspark_attention_backend, "flash_v3")
        with self.assertRaisesRegex(ValueError, "dspark_attention_backend"):
            self._build(dspark_depth=1, dspark_attention_backend="typo")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_depth_populates_the_released_layout(self):
        """Feature: Drafter target defaults.

        Description: A positive depth targets the last three layers, as released.
        Expectation: Positive depth selects the expected final backbone layers.
        """
        config = self._build(dspark_depth=1)
        self.assertEqual(config.v41_dspark_depth, 1)
        self.assertEqual(config.v41_dspark_target_layer_ids, [37, 38, 39])
        self.assertEqual(config.v41_dspark_window, 128)
        self.assertLessEqual(config.v41_dspark_top_k, config.v41_dspark_n_routed_experts)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_model_builds_the_drafter_only_when_depth_is_positive(self):
        """Feature: Model component construction.

        Description: The config field must actually reach the model, not just the config.
        Expectation: The model contains a drafter exactly when the configuration enables one.


        A run with the drafter silently absent looks like a run with the
        drafter present but ineffective, so assert the object exists.
        """
        from hyper_parallel.models.deepseek_v41.adapter.validation.cropped_model import (  # pylint: disable=C0415
            build_deepseek_v41_validation_config,
        )
        with tempfile.TemporaryDirectory() as directory:
            config_path = _write_released_config(directory)
            assets = _write_engram_assets(directory, num_hidden_layers=40, head_dim=8)
            build = functools.partial(
                build_deepseek_v41_validation_config, str(config_path.parent), str(assets))
            with init_empty_weights():
                off = DeepseekV41ForCausalLM(build())
                on = DeepseekV41ForCausalLM(build(dspark_depth=1))
        self.assertIsNone(off.dspark)
        self.assertIsInstance(on.dspark, DeepseekV41DSpark)
        self.assertGreater(on.dspark_loss_coeff, 0.0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_explicit_targets_override_the_default(self):
        """Feature: Explicit backbone feature taps.

        Description: Callers can point the drafter at other backbone layers.
        Expectation: User-specified target layers replace the default target set.
        """
        config = self._build(dspark_depth=1, dspark_target_layer_ids=[10, 11])
        self.assertEqual(config.v41_dspark_target_layer_ids, [10, 11])


if __name__ == "__main__":
    unittest.main()



class TestHFLayerCropConfig(unittest.TestCase):
    """Exercise the real config helpers after upstream crop API changes."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_released_engram_assets_preserve_table_shapes(self) -> None:
        """Feature: HF prefix crop with Engram.

        Description: Retain an Engram layer with released-size Host hashing assets.
        Expectation: Keep the original table shape and reject scaled assets.
        """
        with tempfile.TemporaryDirectory() as directory:
            source = json.loads(_write_released_config(directory).read_text(encoding="utf-8"))
            text = source["text_config"]
            text.update(engram_num_embeddings=[100000, 101000], engram_vocab_size=4096)
            assets = {
                "source_model_type": source["model_type"], "num_hidden_layers": text["num_hidden_layers"],
                "layer_ids": text["engram_layer_ids"], "num_embeddings": text["engram_num_embeddings"],
                "head_dim": text["engram_head_dim"], "bucket_base": text["engram_vocab_size"],
                "table_pad_multiple": 1,
            }
            path = Path(directory) / "released_engram.json"
            path.write_text(json.dumps(assets), encoding="utf-8")
            destination = Path(directory) / "crop"
            config = build_layer_cropped_config(
                source, destination, text_layers=2, vision_layers=2,
                engram_assets_path=str(path), engram_storage_backend="host",
            )
            self.assertEqual(config.v41_engram_layer_ids, [1], "The crop must retain the first Engram layer")
            self.assertEqual(config.v41_engram_num_embeddings, [text["engram_num_embeddings"][0]],
                             "HF loading must not resize the retained table")
            self.assertEqual(config.engram_storage_backend, "host", "Host storage must reach the model config")
            for field, value in (("head_dim", 1), ("bucket_base", 2), ("padded_num_embeddings", [1, 2])):
                with self.subTest(field=field):
                    path.write_text(json.dumps(dict(assets, **{field: value})), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "Released Engram assets"):
                        build_layer_cropped_config(source, destination, text_layers=2, engram_assets_path=str(path))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_retained_engram_requires_released_assets(self) -> None:
        """Feature: HF prefix crop with Engram.

        Description: Request two decoder layers without released hashing metadata.
        Expectation: Fail before constructing or reading a differently sized table.
        """
        with tempfile.TemporaryDirectory() as directory:
            source = json.loads(_write_released_config(directory).read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "released hashing assets"):
                build_layer_cropped_config(source, Path(directory) / "crop", text_layers=2)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_vision_depth_and_original_dimensions(self) -> None:
        """
        Feature: Runtime HF prefix-layer crop configuration.
        Description: Build real configs with zero, two or three vision layers using current upstream helpers.
        Expectation: Requested depths are applied without resizing text, vision or expert dimensions.
        """
        with tempfile.TemporaryDirectory() as directory:
            source = json.loads(_write_released_config(directory).read_text(encoding="utf-8"))
            source["text_config"]["engram_vocab_size"] = 1024
            for depth in (0, 2, 3):
                with self.subTest(vision_layers=depth):
                    destination = Path(directory) / f"crop_{depth}"
                    config = build_layer_cropped_config(source, destination, text_layers=1, vision_layers=depth)
                    self.assertEqual(config.num_hidden_layers, 1,
                                     f"text depth={config.num_hidden_layers}, expected=1")
                    self.assertEqual(config.hidden_size, 256,
                                     f"hidden size={config.hidden_size}, expected=256")
                    self.assertEqual(config.num_local_experts, 384,
                                     f"experts={config.num_local_experts}, expected=384")
                    self.assertEqual(config.v41_vision_enabled, depth > 0,
                                     f"vision enabled={config.v41_vision_enabled}, expected={depth > 0}")
                    self.assertEqual(config.v41_vision_num_hidden_layers, depth,
                                     f"vision depth={config.v41_vision_num_hidden_layers}, expected={depth}")
                    self.assertEqual(config.v41_vision_hidden_size, 128,
                                     f"vision width={config.v41_vision_hidden_size}, expected=128")
