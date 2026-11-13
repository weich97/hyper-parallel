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
"""Unit tests for the DeepSeek-V4.1 Chunk Loss adapter."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch  # pylint: disable=wrong-import-position
from torch import nn  # pylint: disable=wrong-import-position
from torch.nn import functional  # pylint: disable=wrong-import-position

from hyper_parallel.components.losses import ChunkedCausalLMLoss  # pylint: disable=wrong-import-position
from hyper_parallel.data.constants import IGNORE_INDEX  # pylint: disable=wrong-import-position
from hyper_parallel.models.deepseek_v41.adapter.chunk_loss import (  # pylint: disable=wrong-import-position
    DeepseekV41ChunkedCausalLMLoss,
    bind_chunk_loss,
)
from tests.common.mark_utils import arg_mark  # pylint: disable=wrong-import-position


class _Backbone(nn.Module):
    """Small decoder that exposes the V4.1 last-hidden-state contract."""

    def __init__(self) -> None:
        """Create the tiny decoder embedding."""
        super().__init__()
        self.embed_tokens = nn.Embedding(17, 8)

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        """Return local hidden states without using the LM head."""
        if kwargs.get("use_cache"):
            raise AssertionError("the Chunk Loss path requested a KV cache")
        return SimpleNamespace(last_hidden_state=self.embed_tokens(input_ids))


class _CausalLM(nn.Module):
    """Small facade with the V4.1 architecture and V4 config identity."""

    def __init__(self) -> None:
        """Create a V4.1-shaped causal LM for CPU tests."""
        super().__init__()
        self.config = SimpleNamespace(
            model_type="deepseek_v4", architectures=["DeepseekV41ForCausalLM"]
        )
        self.model = _Backbone()
        self.lm_head = nn.Linear(8, 17, bias=False)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        """Return ordinary full logits for the reference and passthrough test."""
        hidden = self.model(input_ids).last_hidden_state
        return SimpleNamespace(logits=self.lm_head(hidden))


class TestDeepseekV41ChunkLoss(unittest.TestCase):
    """Check adapter lookup, loss parity, and unchanged ordinary forwards."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_chunk_loss_matches_full_logits_and_gradients(self) -> None:
        """Feature: Bounded vocabulary projection.

        Description: Compare masked chunk-loss targets with eager full logits.
        Expectation: Loss and input/head gradients agree without a full head forward.
        """
        torch.manual_seed(31)
        model = _CausalLM()
        reference = _CausalLM()
        reference.load_state_dict(model.state_dict())
        input_ids = torch.randint(0, 17, (1, 7))
        targets = torch.randint(0, 17, (1, 7))
        targets[0, -1] = IGNORE_INDEX
        loss_mask = torch.ones_like(targets)
        loss_mask[0, 2] = 0
        masked_targets = targets.masked_fill(loss_mask.eq(0), IGNORE_INDEX)

        logits = reference(input_ids).logits.float()
        eager_loss = functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            masked_targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        eager_loss.backward()

        loss_fn = ChunkedCausalLMLoss(chunk_size=3)
        setup = SimpleNamespace(
            mesh_context=SimpleNamespace(tp_size=1, pp_size=1, loss_parallel=False)
        )
        loss_fn.bind_model(model, setup)
        prepared = loss_fn.prepare_model_inputs(
            {"input_ids": input_ids, "labels": targets},
            {"shift_labels": targets, "loss_mask": loss_mask},
        )
        with patch.object(model.lm_head, "forward", side_effect=AssertionError("full logits")):
            output = model(**prepared)
        self.assertIsNone(output.logits)
        chunk_loss = loss_fn(model_output=output, labels=targets)
        chunk_loss.backward()

        torch.testing.assert_close(chunk_loss, eager_loss, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(model.lm_head.weight.grad, reference.lm_head.weight.grad)
        torch.testing.assert_close(
            model.model.embed_tokens.weight.grad,
            reference.model.embed_tokens.weight.grad,
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_binding_preserves_ordinary_forward(self) -> None:
        """Feature: Ordinary inference compatibility.

        Description: Bind twice and forward without Chunk Loss targets.
        Expectation: The original logits path and idempotent wrapper are preserved.
        """
        model = _CausalLM()
        input_ids = torch.tensor([[1, 2, 3]])
        expected = model(input_ids=input_ids).logits

        bind_chunk_loss(model)
        bound_forward = model.forward
        bind_chunk_loss(model)

        self.assertIs(model.forward, bound_forward)
        torch.testing.assert_close(model(input_ids=input_ids).logits, expected)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_binding_rejects_other_architecture(self) -> None:
        """Feature: Model identity validation.

        Description: Request the V4.1 override for another architecture.
        Expectation: Reject the incompatible facade before rewriting forward.
        """
        model = _CausalLM()
        model.config.architectures = ["DeepseekV4ForCausalLM"]
        with self.assertRaises(TypeError):
            bind_chunk_loss(model)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_omni_target_and_mask_alignment(self) -> None:
        """Feature: Omni causal target alignment.

        Description: Prepare ignored visual/padding targets and an explicit mask.
        Expectation: Shift targets and masks together without mutating source inputs.
        """
        labels = torch.tensor([[-100, 3, 4, -100, 6, -100]])
        mask = torch.tensor([[0, 1, 0, 0, 1, 0]], dtype=torch.bool)
        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]), "labels": labels}
        loss = DeepseekV41ChunkedCausalLMLoss(chunk_size=2)
        loss.bind_model(_CausalLM())
        prepared = loss.prepare_model_inputs(inputs, {"labels": labels, "loss_mask": mask})
        torch.testing.assert_close(prepared["chunk_loss_targets"], torch.tensor([[3, 4, -100, 6, -100, -100]]))
        torch.testing.assert_close(prepared["chunk_loss_mask"], torch.tensor([[1, 0, 0, 1, 0, 0]], dtype=torch.bool))
        self.assertNotIn("labels", prepared)
        self.assertIs(inputs["labels"], labels)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_explicit_shifted_targets_keep_precedence(self) -> None:
        """Feature: Pre-shifted text targets.

        Description: Provide both original labels and explicit shifted targets.
        Expectation: Preserve the shifted tensors without a second causal shift.
        """
        loss = DeepseekV41ChunkedCausalLMLoss(chunk_size=2)
        loss.bind_model(_CausalLM())
        labels = torch.tensor([[1, 2, 3]])
        shifted = torch.tensor([[2, 3, -100]])
        mask = torch.tensor([[1, 1, 0]])
        prepared = loss.prepare_model_inputs(
            {"input_ids": labels, "labels": labels},
            {"labels": labels, "shift_labels": shifted, "loss_mask": mask},
        )
        self.assertIs(prepared["chunk_loss_targets"], shifted)
        self.assertIs(prepared["chunk_loss_mask"], mask)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_sequential_schedule_rejects_missing_drafter_and_stale_phase(self) -> None:
        """Feature: Sequential objective schedule identity.
        Description: Request an absent drafter or a phase without its matching backbone.
        Expectation: Fail before constructing or backpropagating an invalid objective.
        """
        with self.assertRaisesRegex(ValueError, "boolean"):
            DeepseekV41ChunkedCausalLMLoss(sequential_dspark=1)
        loss = DeepseekV41ChunkedCausalLMLoss(sequential_dspark=True)
        with self.assertRaisesRegex(ValueError, "enabled drafter"):
            loss.bind_model(_CausalLM())
        model = _CausalLM()
        bind_chunk_loss(model)
        inputs = dict(input_ids=torch.tensor([[1, 2]]), chunk_loss_targets=torch.tensor([[2, -100]]),
                      chunk_loss_mask=None, chunk_loss_chunk_size=1, chunk_loss_ignore_index=-100)
        with self.assertRaisesRegex(ValueError, "schedule identity"):
            model(**inputs, chunk_loss_phase="backbone")
        with self.assertRaisesRegex(RuntimeError, "pending backbone"):
            model(**inputs, chunk_loss_phase="dspark", chunk_loss_schedule=object())
        model._hp_pending_dspark_schedule = (object(), inputs["input_ids"], inputs["chunk_loss_targets"])
        with self.assertRaisesRegex(RuntimeError, "has not completed"):
            model(**inputs)
        with self.assertRaisesRegex(RuntimeError, "pending backbone"):
            model(**inputs, chunk_loss_phase="dspark", chunk_loss_schedule=object())

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_omni_chunk_loss_rejects_unverified_cp_boundary(self) -> None:
        """Feature: Context-parallel boundary guard.

        Description: Request Omni local shifting with two CP shards.
        Expectation: Reject the unverified boundary and leave forward unchanged.
        """
        model = _CausalLM()
        original = model.forward
        loss = DeepseekV41ChunkedCausalLMLoss(chunk_size=2)
        setup = SimpleNamespace(mesh_context=SimpleNamespace(cp_size=2))
        with self.assertRaisesRegex(NotImplementedError, "cp_size=1"):
            loss.bind_model(model, setup)
        self.assertEqual(model.forward, original)


if __name__ == "__main__":
    unittest.main()
