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
"""Subset and full SWE-Smith preparation must preserve identical task semantics."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

import pyarrow.parquet as pq
import torch

from examples.swe_smith.infer import load_prompts
from examples.swe_smith.prepare_data import adapt_row, prepare_data, record_path


def instance(identity: str, image: str = "fixture/one") -> dict:
    """Construct representative metadata without requiring a task image."""
    return {"instance_id": identity, "problem_statement": "Fix " + identity, "image_name": image,
            "repo": "fixture/repo", "FAIL_TO_PASS": ["tests/test_api.py::test_bug"],
            "PASS_TO_PASS": ["tests/test_api.py::test_ok", "tests/test_other.py::test_other"]}


class TestPrepareData(unittest.TestCase):
    """Exercise selection, identity, privacy, split isolation and bounded writes."""

    def setUp(self) -> None:
        """Create source records used by both training and inference data paths."""
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.splits = {"train": self.root / "train.jsonl", "validation": self.root / "validation.jsonl"}
        self.rows = [instance("first"), instance("second"), instance("third", "fixture/two")]
        self.splits["train"].write_text("\n".join(json.dumps(row) for row in self.rows))
        self.splits["validation"].write_text(json.dumps(instance("validation", "fixture/two")))

    def test_full_and_subset_use_the_same_records_and_prompts(self) -> None:
        """Full and selected splits retain identical trusted records and public prompts."""
        full, subset = self.root / "full", self.root / "subset"
        result = prepare_data(self.splits, full)
        self.assertEqual(result["splits"]["train"]["instance_ids"], ["first", "second", "third"])
        self.assertEqual(result["splits"]["validation"]["instance_ids"], ["validation"])
        prepare_data(self.splits, subset, instance_ids=["second", "validation"])
        full_rows = pq.read_table(full / "train.parquet").to_pylist()
        selected = pq.read_table(subset / "train.parquet").to_pylist()
        self.assertEqual(selected, [full_rows[1]])
        public = json.dumps(selected)
        for private in ["FAIL_TO_PASS", "PASS_TO_PASS", "test_bug", "fixture/one"]:
            self.assertNotIn(private, public)
        prompt = adapt_row(selected[0], 0)
        raw = record_path(subset, "second").read_bytes()
        self.assertEqual(prompt.ground_truth["record_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(prompt.metadata["task_type"], "swe_smith")

    def test_inference_retains_token_metadata_and_rejects_missing_instances(self) -> None:
        """Real prepared data must satisfy the shared trajectory contract before model startup."""
        output = self.root / "inference"
        prepare_data(self.splits, output)
        tokenizer = MagicMock(pad_token_id=0)
        tokenizer.apply_chat_template.return_value = {"input_ids": [7, 8, 9], "attention_mask": [1, 1, 1]}
        prompts = load_prompts([output / "train.parquet"], tokenizer, ["second"])
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0].prompt_id, "swe-smith:second")
        self.assertTrue(torch.equal(prompts[0].metadata["input_ids"], torch.tensor([7, 8, 9])))
        with self.assertRaisesRegex(ValueError, "missing"):
            load_prompts([output / "train.parquet"], tokenizer, ["missing"])

    def test_selection_limit_applies_after_filter_without_losing_missing_id_checks(self) -> None:
        result = prepare_data(self.splits, self.root / "limited", instance_ids=["third", "second"], max_instances=1)
        self.assertEqual(result["splits"]["train"]["instance_ids"], ["second"])
        with self.assertRaisesRegex(ValueError, "not present"):
            prepare_data(self.splits, self.root / "missing", instance_ids=["first", "absent"], max_instances=1)

    def test_replication_is_explicit_and_uses_distinct_sampling_identities(self) -> None:
        prepare_data(self.splits, self.root / "repeat", instance_ids=["first"], repeats=2)
        rows = pq.read_table(self.root / "repeat/train.parquet").to_pylist()
        prompts = [adapt_row(row, index) for index, row in enumerate(rows)]
        self.assertNotEqual(prompts[0].prompt_id, prompts[1].prompt_id)
        self.assertEqual(prompts[0].ground_truth, prompts[1].ground_truth)
        self.assertEqual(pq.read_table(self.root / "repeat/validation.parquet").num_rows, 0)

    def test_overlapping_splits_invalid_nodes_and_overwrite_are_rejected(self) -> None:
        """Reject overlapping identities, unsafe pytest nodes and nonempty output directories."""
        self.splits["validation"].write_text(json.dumps(self.rows[0]))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            prepare_data(self.splits, self.root / "overlap")
        invalid = instance("invalid")
        invalid["FAIL_TO_PASS"] = []
        self.splits["train"].write_text(json.dumps(invalid))
        with self.assertRaisesRegex(ValueError, "at least one"):
            prepare_data({"train": self.splits["train"]}, self.root / "invalid")
        with self.assertRaisesRegex(ValueError, "empty prepared"):
            prepare_data(self.splits, self.root / "invalid")

    def test_multiple_write_batches_preserve_order(self) -> None:
        self.splits["train"].write_text("\n".join(json.dumps(instance(f"case-{index}")) for index in range(260)))
        prepare_data({"train": self.splits["train"]}, self.root / "batches")
        rows = pq.read_table(self.root / "batches/train.parquet").to_pylist()
        self.assertEqual(len(rows), 260)
        self.assertEqual([row["task_id"] for row in rows], [f"swe-smith:case-{index}" for index in range(260)])
