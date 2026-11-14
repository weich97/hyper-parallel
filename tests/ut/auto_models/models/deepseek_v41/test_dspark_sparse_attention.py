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

import torch

from hyper_parallel.models.deepseek_v41.adapter.dspark_sparse_attention import _sparse_draft_metadata
from tests.common.mark_utils import arg_mark


class DSparkSparseMetadataTest(unittest.TestCase):
    """Preserve batching, document boundaries and noncausal draft-block context."""

    def tearDown(self):
        """Release cached test tensors between geometries."""
        _sparse_draft_metadata.cache_clear()

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_packed_documents_and_batches_keep_only_legal_keys(self):
        """Every draft token shares exactly its anchor's context and independent block."""
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
        """Chunk-local positions must not replace absolute positions in the main bank."""
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
        """Invalid geometry cannot reach a device memory operation."""
        for geometry in ((0, 8, 3, 4, 0, (0,)), (1, 8, 3, 4, 0, (1,)),
                         (1, 8, 3, 4, 8, (0,)), (1, 8, 3, 1024, 0, (0,)),
                         (1, 8, 3, 4, 0, ())):
            with self.subTest(geometry=geometry), self.assertRaisesRegex(ValueError, "geometry"):
                _sparse_draft_metadata(*geometry, torch.device("cpu"))
