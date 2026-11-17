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
"""Verify unbiased anchor sampling, context identity and fixed-size dispatch."""

import copy
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.models.deepseek_v41.dspark import DSparkDraftAttention, _AnchorSelection, _sample_draft_anchors
from hyper_parallel.models.deepseek_v41.adapter.dspark_sparse_attention import _sparse_draft_metadata
from tests.common.mark_utils import arg_mark
from tests.ut.auto_models.models.deepseek_v41.test_dspark import (
    _batch, _build, _flash_v3_torch_reference, _noise, _small_config,
)


class DSparkAnchorSamplingTest(unittest.TestCase):
    """Compare sampled objectives with exhaustive CPU reference expectations."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_eligible_union_and_padding(self):
        """Image/label holes do not enter the pool; padding retains the dispatch size."""
        targets = torch.full((2, 7, 3), IGNORE_INDEX)
        targets[0, 1, 0] = 1
        targets[0, 4, 1] = 2
        targets[1, 5, 2] = 3
        selected = _sample_draft_anchors(targets, 5)
        expected = _AnchorSelection((1, 4, 5, 0, 0), 3, 3)
        self.assertEqual(selected, expected, f"selected={selected}, expected={expected}")
        self.assertEqual(selected.weight, 1.0, f"weight={selected.weight}, expected={1.0}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_exhaustive_sampling_matches_dense_loss_and_gradients(self):
        """Average over every one-anchor draw recovers the complete weighted objective."""
        config = _small_config()
        dense, head = _build(config)
        sampled = copy.deepcopy(dense)
        sampled.anchor_budget = 1
        input_ids, labels, hiddens, embeds = _batch(config)
        token_mask = torch.ones_like(input_ids, dtype=torch.bool)
        token_mask[0, 6] = False
        boundaries = (0, 5, 10)
        args = (hiddens, input_ids, labels, embeds, head)
        kwargs = dict(noise_embedding=_noise(config), token_mask=token_mask, sequence_boundaries=boundaries)
        _, targets = dense._teacher_forced_targets(input_ids, labels, token_mask, boundaries)
        eligible = (targets != IGNORE_INDEX).any(dim=(0, 2)).nonzero().flatten().tolist()
        full_loss, _ = dense(*args, **kwargs)
        expected_gradients = torch.autograd.grad(full_loss, tuple(dense.parameters()))
        losses, draws = [], []
        for anchor in eligible:
            selection = _AnchorSelection((anchor,), 1, len(eligible))
            with patch("hyper_parallel.models.deepseek_v41.dspark._sample_draft_anchors", return_value=selection):
                loss, _ = sampled(*args, **kwargs)
            losses.append(loss.detach())
            draws.append(torch.autograd.grad(loss, tuple(sampled.parameters())))
        torch.testing.assert_close(torch.stack(losses).mean(), full_loss.detach(), rtol=2e-6, atol=2e-6)
        for index, expected in enumerate(expected_gradients):
            observed = torch.stack([draw[index] for draw in draws]).mean(0)
            torch.testing.assert_close(observed, expected, rtol=2e-4, atol=2e-6)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_empty_supervision_has_connected_zero_gradients(self):
        """An empty local sampling pool still executes fixed-size connected stages."""
        config = _small_config()
        config.v41_dspark_num_anchors = 4
        module, head = _build(config)
        input_ids, labels, hiddens, embeds = _batch(config)
        labels.fill_(IGNORE_INDEX)
        loss, metrics = module(hiddens, input_ids, labels, embeds, head, noise_embedding=_noise(config))
        expected = _AnchorSelection((0, 0, 0, 0), 0, 0)
        self.assertEqual(module.last_anchor_selection, expected,
                         f"selection={module.last_anchor_selection}, expected={expected}")
        self.assertEqual(loss.item(), 0.0, f"loss={loss.item()}, expected={0.0}")
        self.assertEqual(metrics["dspark_supervised_tokens"].item(), 0,
                         f"tokens={metrics['dspark_supervised_tokens'].item()}, expected={0}")
        gradients = torch.autograd.grad(loss, tuple(module.parameters()))
        for gradient in gradients:
            torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_full_budget_preserves_loss_and_rng(self):
        """Selecting the complete sequence must retain the previous training behavior."""
        config = _small_config()
        module, head = _build(config)
        input_ids, labels, hiddens, embeds = _batch(config)
        args = (hiddens, input_ids, labels, embeds, head)
        state = torch.get_rng_state()
        baseline, _ = module(*args, noise_embedding=_noise(config))
        module.anchor_budget = input_ids.shape[1]
        candidate, _ = module(*args, noise_embedding=_noise(config))
        torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), state), f"rng_equal={torch.equal(torch.get_rng_state(), state)}")
        self.assertIsNone(module.last_anchor_selection)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_sparse_metadata_preserves_noncontiguous_anchor_positions(self):
        """Compaction must not turn original positions into contiguous local indices."""
        actual = _sparse_draft_metadata(1, 16, 3, 4, 0, (2, 1), torch.device("cpu"), (2, 9))
        expected = ([0, 1, 2, 16, 17, 18], [8, 9, 19, 20, 21])
        for row, selected in zip((actual[0, 0], actual[3, 0]), expected, strict=True):
            observed = row[row >= 0].tolist()
            self.assertEqual(observed, selected, f"observed={observed}, expected={selected}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_compacted_attention_matches_selected_dense_rows(self):
        """Packed-document positions and chunk replay retain selected outputs and all gradients."""
        config = _small_config()
        config.v41_dspark_attention_chunk_size = 2
        module, _ = _build(config)
        cos, sin = module._rope_tables(10, torch.device("cpu"), torch.float32)
        selected = (1, 4, 5, 8)
        indices = torch.tensor(selected)
        for backend in ("reference", "flash_v3"):
            with self.subTest(backend=backend):
                config.v41_dspark_attention_backend = backend
                reference = DSparkDraftAttention(config)
                candidate = copy.deepcopy(reference)
                draft = torch.randn(2, 10, 3, 32, requires_grad=True)
                main = torch.randn(2, 10, 32, requires_grad=True)
                compact = draft.detach().index_select(1, indices).requires_grad_()
                candidate_main = main.detach().clone().requires_grad_()
                kwargs = dict(sequence_boundaries=(0, 5, 10))
                with patch("hyper_parallel.models.deepseek_v41.dspark._flash_attention_v3",
                           side_effect=_flash_v3_torch_reference):
                    expected = reference(draft, main, cos, sin, **kwargs).index_select(1, indices)
                    actual = candidate(compact, candidate_main, cos, sin, anchor_positions=selected, **kwargs)
                    upstream = torch.randn_like(expected)
                    expected_gradients = torch.autograd.grad(
                        expected, (draft, main, *reference.parameters()), upstream)
                    actual_gradients = torch.autograd.grad(
                        actual, (compact, candidate_main, *candidate.parameters()), upstream)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(actual_gradients[0], expected_gradients[0].index_select(1, indices),
                                           rtol=2e-5, atol=2e-6)
                for actual_gradient, expected_gradient in zip(actual_gradients[1:], expected_gradients[1:], strict=True):
                    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-5, atol=2e-6)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_sampling_replays_from_rng_state(self):
        """Restoring the CPU generator state replays the same sorted sample without replacement."""
        targets = torch.ones(2, 23, 3, dtype=torch.long)
        state = torch.get_rng_state()
        first = _sample_draft_anchors(targets, 8)
        torch.set_rng_state(state)
        second = _sample_draft_anchors(targets, 8)
        self.assertEqual(first, second, f"first={first}, second={second}")
        self.assertEqual(len(set(first.positions)), 8, f"positions={first.positions}, expected_unique={8}")
        self.assertEqual(first.positions, tuple(sorted(first.positions)), f"positions={first.positions}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_invalid_anchor_budgets_are_rejected(self):
        """Only explicit nonnegative integer budgets select this training protocol."""
        for budget in (-1, 2.5, True):
            config = _small_config()
            config.v41_dspark_num_anchors = budget
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, "nonnegative integer"):
                _build(config)
