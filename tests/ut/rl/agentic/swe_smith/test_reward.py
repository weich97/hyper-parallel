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
"""Trusted task identity, test scope and independently scored frozen submissions."""

from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Sequence
import unittest
from unittest.mock import AsyncMock, patch

import pyarrow.parquet as pq

from rl.agentic.envs.docker_workspace import CommandResult, InvalidSubmissionError
from examples.swe_smith import reward
from examples.swe_smith.prepare_data import adapt_row, prepare_data, record_path
from tests.ut.rl.agentic.swe_smith.test_prepare_data import instance


class TestReward(unittest.IsolatedAsyncioTestCase):
    """Grading infrastructure failures must never become ordinary zero rewards."""

    def setUp(self) -> None:
        """Keep trusted task fixtures alive through each asynchronous test and its cleanup."""
        # unittest cleanup retains the directory across setup and the asynchronous test body.
        directory = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.row = instance("task")
        source = self.root / "source.jsonl"
        source.write_text(json.dumps(self.row))
        prepared = self.root / "prepared"
        prepare_data({"train": source}, prepared)
        self.prompt = adapt_row(pq.read_table(prepared / "train.parquet").to_pylist()[0], 0)
        images = {"schema_version": 1, "complete": True, "images": {"fixture/one": {
            "image": "sha256:" + "a" * 64, "python": "/env/bin/python"}}}
        image_path = self.root / "images.json"
        image_path.write_text(json.dumps(images))
        self.config = {"data_dir": str(prepared), "images_path": str(image_path),
                       "images_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                       "artifact_dir": str(self.root)}
        self.task = reward.SWESmithTask(self.prompt, self.config)
        self.archive = self.root / "submission.tar"
        self.artifact = {"patch": "source patch", "content_hash": "content", "patch_sha256": "patch",
                         "changes": {"modified": ["pkg/source.py"], "added": [], "deleted": []}}

    def test_exact_same_file_scope_and_full_p2p_option(self) -> None:
        self.assertEqual(reward.evaluation_nodes(self.row), self.row["FAIL_TO_PASS"] + self.row["PASS_TO_PASS"][:1])
        self.assertEqual(reward.evaluation_nodes(self.row, False), self.row["FAIL_TO_PASS"] + self.row["PASS_TO_PASS"])
        changed = {**self.row, "PASS_TO_PASS": ["tests/test_api.py_extra::not_same_file"]}
        self.assertEqual(reward.evaluation_nodes(changed), self.row["FAIL_TO_PASS"])

    def test_statuses_preserve_space_parameter_ids_and_reject_contradictions(self) -> None:
        node = "tests/test_api.py::test_bug[a - b]"
        output = f"\x1b[32mPASSED\x1b[0m {node}\n{node} PASSED [100%]\n"
        self.assertEqual(reward.parse_statuses(output, [node]), {node: "PASSED"})
        self.assertEqual(reward.parse_statuses(f"FAILED {node} - AssertionError\n", [node]), {node: "FAILED"})
        with self.assertRaisesRegex(RuntimeError, "Contradictory"):
            reward.parse_statuses(output + f"FAILED {node} - failed\n", [node])

    def test_image_declared_test_paths_normalize_without_rewriting_dataset_records(self) -> None:
        original = {**self.row, "FAIL_TO_PASS": ["../dev/tests/test_api.py::test_bug"]}
        mapped = reward.normalize_test_nodes(original, {"../dev/tests/test_api.py": "tests/test_api.py"})
        self.assertEqual(mapped["FAIL_TO_PASS"], self.row["FAIL_TO_PASS"])
        self.assertEqual(original["FAIL_TO_PASS"], ["../dev/tests/test_api.py::test_bug"])
        for mapping in [{}, {"../dev/tests/test_api.py": "../../outside.py"}]:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                reward.normalize_test_nodes(original, mapping)

    def test_modified_data_image_identity_or_incomplete_manifest_are_rejected(self) -> None:
        """Reject changed task records, image identities and incomplete image manifests."""
        for field, value in [("images_sha256", "wrong"), ("images_path", str(self.root / "missing.json"))]:
            with self.subTest(field=field), self.assertRaises((ValueError, FileNotFoundError)):
                reward.SWESmithTask(self.prompt, {**self.config, field: value})
        image_path = Path(self.config["images_path"])
        original = image_path.read_bytes()
        images = json.loads(original)
        for image in (None, "image:latest", "sha256:abcd", "sha256:" + "G" * 64):
            images["images"]["fixture/one"]["image"] = image
            image_path.write_text(json.dumps(images), encoding="utf-8")
            config = {**self.config, "images_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()}
            with self.subTest(image=image), self.assertRaisesRegex(ValueError, "fixed local image ID"):
                reward.SWESmithTask(self.prompt, config)
        image_path.write_bytes(original)
        path = record_path(Path(self.config["data_dir"]), "task")
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "task identity changed"):
            reward.SWESmithTask(self.prompt, self.config)

    async def test_frozen_patch_and_report_share_the_actual_artifact_identity(self) -> None:
        """The independent grader reports the exact frozen source artifact identity."""
        for resolved in [True, False]:
            with self.subTest(resolved=resolved), \
                    patch.object(reward, "build_submission", return_value=dict(self.artifact)), \
                    patch.object(self.task, "_evaluate_patch", new=AsyncMock(return_value={
                        "resolved": resolved, "status": "resolved" if resolved else "unresolved", "statuses": {}})):
                result = await self.task.evaluate(self.archive)
                self.assertEqual(result.value, float(resolved))
                self.assertEqual(result.metadata["artifact_hash"], "content")
                self.assertEqual(result.metadata["patch_sha256"], "patch")
                self.assertEqual(self.archive.with_suffix(".patch").read_text(), "source patch")

    async def test_invalid_submission_zero_and_infrastructure_failure_propagation(self) -> None:
        with patch.object(reward, "build_submission", side_effect=InvalidSubmissionError("test changed")):
            result = await self.task.evaluate(self.archive)
            self.assertEqual(result.value, 0.)
            self.assertEqual(result.metadata["status"], "invalid_submission")
        with patch.object(reward, "build_submission", return_value=dict(self.artifact)), \
                patch.object(self.task, "_evaluate_patch", new=AsyncMock(side_effect=RuntimeError("missing tests"))):
            with self.assertRaisesRegex(RuntimeError, "missing tests"):
                await self.task.evaluate(self.archive)

    async def test_setup_snapshot_precedes_model_tools_and_uses_same_image(self) -> None:
        workspace = SimpleNamespace(image=self.task.workspace_image, snapshot=AsyncMock())
        refs = {"problem_commit": "problem", "test_commit": "tests"}
        with patch.object(reward, "prepare_candidate", new=AsyncMock(return_value=refs)):
            await self.task.prepare(workspace)
        self.assertEqual(self.task.refs, refs)
        workspace.snapshot.assert_awaited_once_with(self.task.baseline, **self.task.artifact_limits)
        workspace.image = "other"
        with self.assertRaisesRegex(RuntimeError, "differs"):
            await self.task.prepare(workspace)

    async def test_grader_requires_every_selected_test_and_preserves_failure_cleanup(self) -> None:
        """Every selected test is accounted for and cleanup preserves the primary failure."""
        self.task.refs = {"problem_commit": "a" * 40, "test_commit": "b" * 40}
        complete = "\n".join("PASSED " + node for node in self.task.nodes)
        for output, code, accepted in [(complete, 0, True), ("", 0, False), (complete, 1, False)]:
            grader = SimpleNamespace(start=AsyncMock(), copy_in=AsyncMock(), close=AsyncMock(),
                                     exec=AsyncMock(side_effect=[CommandResult(1, "", ""),
                                                                CommandResult(code, output, "")]))
            with self.subTest(output=output, code=code), \
                    patch.object(reward, "DockerWorkspace", return_value=grader) as factory, \
                    patch.object(reward, "checkout_problem", new=AsyncMock(return_value=self.task.refs)), \
                    patch.object(reward, "checked", new=AsyncMock(return_value=CommandResult(0, "[]", ""))), \
                    patch.object(reward, "restore_tests", new=AsyncMock()) as restore:
                invoke = self.task._evaluate_patch("patch", self.artifact["changes"], self.root / "eval.log")
                if accepted:
                    self.assertTrue((await invoke)["resolved"])
                else:
                    with self.assertRaises(RuntimeError):
                        await invoke
                self.assertEqual(factory.call_args.args[0], self.task.workspace_image)
                restore.assert_awaited_once_with(grader, self.task.refs, self.task.nodes)
                self.assertEqual(grader.exec.call_args.args[0][-len(self.task.nodes):], self.task.nodes)
                grader.close.assert_awaited_once()

    async def test_new_source_syntax_is_checked_before_running_tests(self) -> None:
        """Reject invalid candidate source syntax before executing selected tests."""
        self.task.refs = {"problem_commit": "a" * 40, "test_commit": "b" * 40}
        grader = SimpleNamespace(start=AsyncMock(), copy_in=AsyncMock(), close=AsyncMock(), exec=AsyncMock())
        returns = [CommandResult(0, "[]", "")] * 3 + [CommandResult(0, '[{"path":"pkg/new.py"}]', "")]
        changes = {"modified": [], "added": ["pkg/new.py"], "deleted": []}
        with patch.object(reward, "DockerWorkspace", return_value=grader), \
                patch.object(reward, "checkout_problem", new=AsyncMock(return_value=self.task.refs)), \
                patch.object(reward, "checked", new=AsyncMock(side_effect=returns)) as checked:
            result = await self.task._evaluate_patch("patch", changes, self.root / "eval.log")
        self.assertEqual(result["status"], "syntax_error")
        self.assertEqual(json.loads(checked.call_args.kwargs["stdin"]), ["pkg/new.py"])
        grader.exec.assert_not_awaited()
        grader.close.assert_awaited_once()

    async def test_collection_failure_requires_changed_source_and_clean_baseline(self) -> None:
        """Collection failures become model failures only with changed-source and clean-baseline evidence."""
        self.task.refs = {"problem_commit": "a" * 40, "test_commit": "b" * 40}
        for source, kind, candidate_code, baseline_code, baseline_nodes, accepted in [
            ("/testbed/pkg/source.py", "NameError", 4, 0, self.task.nodes, True),
            ("/testbed/pkg/source.py", "NameError", 1, 0, self.task.nodes, True),
            ("/testbed/pkg/source.py", "NameError", 4, 2, [], False),
            ("/testbed/pkg/source.py", "NameError", 4, 0, [], False),
            ("/env/pytest.py", "NameError", 4, 0, self.task.nodes, False),
            ("/testbed/pkg/source.py", "MemoryError", 4, 0, self.task.nodes, False),
        ]:
            error = {"type": kind, "path": source, "line": 2, "message": "missing name"}
            candidate = {"exitstatus": 4, "errors": [error], "nodes": []}
            baseline = {"exitstatus": baseline_code, "errors": [], "nodes": baseline_nodes}
            grader = SimpleNamespace(start=AsyncMock(), copy_in=AsyncMock(), close=AsyncMock(),
                                     exec=AsyncMock(side_effect=[CommandResult(1, "", ""),
                                                                CommandResult(candidate_code, "collection failed", ""),
                                                                CommandResult(4, "candidate collection", ""),
                                                                CommandResult(baseline_code, "baseline", "")]))

            async def checked(workspace: SimpleNamespace, argv: Sequence[str], *,
                              candidate: dict, baseline: dict, **kwargs: Any) -> CommandResult:
                """Return structured collection reports without invoking a Docker process."""
                del workspace, kwargs
                value = candidate if argv[-1] == "/workspace/candidate-collection.json" else baseline
                return CommandResult(0, json.dumps(value) if "read_text" in " ".join(argv) else "[]", "")

            with self.subTest(source=source, kind=kind, baseline_code=baseline_code, baseline_nodes=baseline_nodes), \
                    patch.object(reward, "DockerWorkspace", return_value=grader), \
                    patch.object(reward, "checkout_problem", new=AsyncMock(return_value=self.task.refs)), \
                    patch.object(reward, "checked", new=partial(checked, candidate=candidate, baseline=baseline)), \
                    patch.object(reward, "restore_tests", new=AsyncMock()):
                invoke = self.task._evaluate_patch("patch", self.artifact["changes"], self.root / "eval.log")
                if accepted:
                    report = await invoke
                    self.assertFalse(report["resolved"])
                    self.assertEqual(report["status"], "collection_error")
                    self.assertTrue(report["baseline_collection_verified"])
                    self.assertEqual(report["errors"], [error])
                    self.assertEqual(report["statuses"], {})
                else:
                    with self.assertRaises(RuntimeError):
                        await invoke
                grader.close.assert_awaited_once()

    def test_collection_plugin_reports_original_exception_and_successful_nodes(self) -> None:
        """The collection hook retains original exception locations and collected node identities."""
        runner = self.root / "collect_tests.py"
        runner.write_text(reward._COLLECT)  # pylint: disable=protected-access
        source = self.root / "broken_source.py"
        test = self.root / "test_api.py"
        test.write_text("import broken_source\ndef test_example():\n    pass\n")
        report = self.root / "collection.json"
        environment = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
        for content, kind in [("class Broken:\n    annotation: MissingName\n", "NameError"),
                              ("raise ImportError('broken candidate import')\n", "ImportError"),
                              ("class Broken:\n    annotation: str\n", None)]:
            source.write_text(content)
            result = subprocess.run([sys.executable, "-I", str(runner), str(report),
                                     str(test) + "::test_example"], cwd=self.root, env=environment,
                                    capture_output=True, text=True, timeout=30, check=False)
            value = json.loads(report.read_text())
            with self.subTest(kind=kind):
                self.assertEqual(value["exitstatus"], result.returncode)
                if kind is None:
                    self.assertEqual(result.returncode, 0, f"Collection failed: {result.stdout}{result.stderr}")
                    self.assertEqual(value["errors"], [])
                    self.assertEqual(len(value["nodes"]), 1)
                    self.assertTrue(value["nodes"][0].endswith("test_api.py::test_example"))
                else:
                    self.assertIn(result.returncode, (2, 4))
                    self.assertEqual(value["errors"][0]["type"], kind)
                    self.assertEqual(value["errors"][0]["path"], str(source))
