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
"""Check selected-KV geometry independently of the optional accelerator package."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.models.deepseek_v41.adapter.dspark_sparse_attention import (
    _sink_key_components,
    _SparseSinkAttention,
    _sparse_draft_metadata,
)
from tests.common.mark_utils import arg_mark


class DSparkSparseMetadataTest(unittest.TestCase):
    """Preserve batching, document boundaries and noncausal draft-block context."""

    def tearDown(self) -> None:
        """Release cached test tensors between geometries."""
        _sparse_draft_metadata.cache_clear()
        _sink_key_components.cache_clear()

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_packed_documents_and_batches_keep_only_legal_keys(self):
        """Feature: Packed sparse draft contexts.

        Description: Enumerate legal main and draft keys for multiple documents and batches.
        Expectation: Every query selects exactly its own context before negative padding.
        """
        batch, main_length, block, window, start = 2, 8, 3, 4, 2
        positions = (2, 3, 0, 1)
        actual = _sparse_draft_metadata(batch, main_length, block, window, start, positions, torch.device("cpu"))
        expected_shape = (batch * len(positions) * block, 1, 1024)
        self.assertEqual(tuple(actual.shape), expected_shape,
                         f"actual_shape={tuple(actual.shape)}, expected_shape={expected_shape}")
        contexts = ([0, 1, 2], [0, 1, 2, 3], [4], [4, 5])
        for batch_index in range(batch):
            for anchor, context in enumerate(contexts):
                main = [index + batch_index * main_length for index in context]
                draft_begin = batch * main_length + (batch_index * len(positions) + anchor) * block
                expected = main + list(range(draft_begin, draft_begin + block))
                for offset in range(block):
                    row_index = (batch_index * len(positions) + anchor) * block + offset
                    row = actual[row_index, 0]
                    observed = row[row >= 0].tolist()
                    self.assertEqual(observed, expected, f"observed={observed}, expected={expected}")
                    padding = row[len(expected):].unique().tolist()
                    self.assertEqual(padding, [-1], f"padding={padding}, expected={[-1]}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_late_chunk_uses_original_positions(self):
        """Feature: Chunked sparse draft attention.

        Description: Start a query chunk after the first 1024 main positions.
        Expectation: Main-bank indices retain absolute positions.
        """
        actual = _sparse_draft_metadata(1, 16384, 5, 128, 1024, (1024, 1025), torch.device("cpu"))
        first = list(range(897, 1025)) + list(range(16384, 16389))
        second = list(range(898, 1026)) + list(range(16389, 16394))
        self.assertEqual(actual[0, 0, :133].tolist(), first,
                         f"observed={actual[0, 0, :133].tolist()}, expected={first}")
        self.assertEqual(actual[5, 0, :133].tolist(), second,
                         f"observed={actual[5, 0, :133].tolist()}, expected={second}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_invalid_geometries_are_rejected(self):
        """Feature: Sparse draft geometry validation.

        Description: Supply empty, oversized and out-of-range context specifications.
        Expectation: Invalid geometry raises before a device memory operation.
        """
        for geometry in ((0, 8, 3, 4, 0, (0,)), (1, 8, 3, 4, 0, (1,)),
                         (1, 8, 3, 4, 8, (0,)), (1, 8, 3, 1024, 0, (0,)),
                         (1, 8, 3, 4, 0, ())):
            with self.subTest(geometry=geometry), self.assertRaisesRegex(ValueError, "geometry"):
                _sparse_draft_metadata(*geometry, torch.device("cpu"))

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_virtual_sink_is_last_valid_key_for_each_sampled_anchor(self):
        """Feature: Virtual sink indexing for sampled anchors.

        Description: Combine multiple batches, noncontiguous anchors and variable context lengths.
        Expectation: Exactly one virtual sink follows the real keys and precedes padding.
        """
        positions = (1, 0, 3)
        actual = _sparse_draft_metadata(2, 12, 5, 4, 0, positions, torch.device("cpu"),
                                        anchor_positions=(1, 7, 10), include_sink=True)
        sink_index = 2 * 12 + 2 * 3 * 5
        for index, row in enumerate(actual[:, 0]):
            count = positions[(index // 5) % 3] + 1 + 5
            expected = [sink_index] + [-1] * (1024 - count - 1)
            self.assertEqual(row[count:].tolist(), expected,
                             f"query={index}, expected_sink_position={count}")
        with self.assertRaisesRegex(ValueError, "geometry"):
            _sparse_draft_metadata(1, 2048, 5, 1019, 1024, (1024,), torch.device("cpu"), include_sink=True)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_virtual_sink_scale_preserves_bf16_logits(self):
        """Feature: Low-precision virtual sink encoding.

        Description: Represent the inverse attention scale with three BF16 mantissas.
        Expectation: The resulting scalar logit scale differs from unity by less than 1e-8.
        """
        components, factor = _sink_key_components(512, torch.bfloat16)
        self.assertLess(abs(factor - 1), 1e-8, f"inverse_scale_factor={factor}, components={components}")
        for component in components:
            quantized = torch.tensor(component, dtype=torch.bfloat16).item()
            self.assertEqual(quantized, component, f"component={component}, quantized={quantized}")


def _emulate_native(query, bank, indices, scale, query_rope, key_rope):
    """Use a materialized mathematical softmax, independent of the NPU implementation."""
    valid = indices[:, 0] >= 0
    selected = indices[:, 0].clamp_min(0).long()
    keys = bank[selected, 0]
    rotary_keys = key_rope[selected, 0]
    scores = (torch.einsum("qhd,qkd->qhk", query, keys)
              + torch.einsum("qhr,qkr->qhk", query_rope, rotary_keys)) * scale
    scores = scores.masked_fill(~valid[:, None], -torch.inf)
    maximum = scores.amax(-1)
    denominator = (scores - maximum[..., None]).exp().sum(-1)
    output = torch.einsum("qhk,qkd->qhd", scores.softmax(-1), keys)
    return output, maximum, denominator


class DSparkSparseSinkTest(unittest.TestCase):
    """Check forward semantics and all derivatives against an explicit scalar sink."""

    @staticmethod
    def _forward(query, key, value, indices, scale, *, query_rope, key_rope, **_kwargs):
        del value
        return _emulate_native(query, key, indices, scale, query_rope, key_rope)

    @staticmethod
    def _backward(query, bank, value, indices, gradient, _output, _maximum, _denominator,
                  *, scale_value, query_rope, key_rope, **_kwargs):
        del value
        with torch.enable_grad():
            differentiable_query = query.detach().requires_grad_()
            differentiable_bank = bank.detach().requires_grad_()
            output = _emulate_native(differentiable_query, differentiable_bank, indices,
                                     scale_value, query_rope, key_rope)[0]
            dq, dkv = torch.autograd.grad(output, (differentiable_query, differentiable_bank), gradient)
        return dq, dkv, bank.new_empty(0), None, None, None

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_sink_forward_and_backward_match_independent_scalar_softmax(self):
        """Feature: Virtual sink forward and backward semantics.

        Description: Compare selected attention with an independent explicit scalar-sink softmax.
        Expectation: Outputs and all query, shared-KV and sink derivatives agree in FP64.
        """
        generator = torch.Generator().manual_seed(317)
        query = torch.randn(6, 3, 8, dtype=torch.float64, generator=generator, requires_grad=True)
        bank = torch.randn(9, 1, 8, dtype=torch.float64, generator=generator, requires_grad=True)
        sink = torch.tensor([-7.0, 0.25, 8.0], dtype=torch.float64, requires_grad=True)
        valid_keys = [[0, 1], [1, 2, 8], [0, 5, 7, 8], [4], [2, 3, 4], [0, 8]]
        indices = torch.full((6, 1, 16), -1, dtype=torch.int32)
        expected = []
        for row, selected in enumerate(valid_keys):
            indices[row, 0, :len(selected)] = torch.tensor(selected)
            indices[row, 0, len(selected)] = bank.shape[0]
            values = bank[selected, 0]
            scores = query[row] @ values.T / query.shape[-1] ** 0.5
            probabilities = torch.cat((scores, sink[:, None]), dim=-1).softmax(-1)
            expected.append(probabilities[:, :-1] @ values)
        expected = torch.stack(expected)
        upstream = torch.randn(expected.shape, dtype=torch.float64, generator=generator)
        expected_gradients = torch.autograd.grad(expected, (query, bank, sink), upstream)
        with patch.dict("sys.modules", {
            "torch_npu": SimpleNamespace(npu_sparse_flash_attention=self._forward),
            "cann_ops_transformer": SimpleNamespace(sparse_flash_attention_grad=self._backward),
        }):
            actual = _SparseSinkAttention.apply(query, bank, indices, sink)
            actual_gradients = torch.autograd.grad(actual, (query, bank, sink), upstream)
        for name, reference, result in zip(("output", "query", "shared_kv", "sink"),
                                           (expected, *expected_gradients), (actual, *actual_gradients), strict=True):
            with self.subTest(tensor=name):
                torch.testing.assert_close(result, reference, rtol=1e-10, atol=1e-10,
                                           msg=f"Virtual sink changed {name}")
