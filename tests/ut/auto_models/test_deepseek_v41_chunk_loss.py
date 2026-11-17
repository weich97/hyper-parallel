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
    bind_chunk_loss,
)


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

    def test_chunk_loss_matches_full_logits_and_gradients(self) -> None:
        """Pre-shifted masked targets give the eager CE loss and gradients."""
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

    def test_binding_preserves_ordinary_forward(self) -> None:
        """Without Chunk Loss targets the original logits path still runs."""
        model = _CausalLM()
        input_ids = torch.tensor([[1, 2, 3]])
        expected = model(input_ids=input_ids).logits

        bind_chunk_loss(model)
        bound_forward = model.forward
        bind_chunk_loss(model)

        self.assertIs(model.forward, bound_forward)
        torch.testing.assert_close(model(input_ids=input_ids).logits, expected)

    def test_binding_rejects_other_architecture(self) -> None:
        """A native V4 facade cannot receive V4.1's training override."""
        model = _CausalLM()
        model.config.architectures = ["DeepseekV4ForCausalLM"]
        with self.assertRaises(TypeError):
            bind_chunk_loss(model)


if __name__ == "__main__":
    unittest.main()
