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
"""Problem and hidden-test commits are checked independently of the image default branch."""

from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from rl.agentic.envs.docker_workspace import CommandResult
from examples.swe_smith.workspace import checkout_problem, restore_tests, test_paths as selected_test_paths


class TestWorkspace(unittest.IsolatedAsyncioTestCase):
    """Verify setup through explicit trusted Git commands without starting Docker."""

    async def test_problem_branch_and_test_parent_are_resolved_to_fixed_commits(self) -> None:
        problem, tests = "a" * 40, "b" * 40
        outputs = [problem, "", "Remove F2P Tests", tests, "Bug Patch", ""]
        workspace = SimpleNamespace(exec=AsyncMock(side_effect=[CommandResult(0, value, "") for value in outputs]))
        refs = await checkout_problem(workspace, {"instance_id": "task", "FAIL_TO_PASS": ["tests/test_x.py::test_x"]})
        self.assertEqual(refs, {"problem_commit": problem, "test_commit": tests})
        self.assertIn("refs/heads/task^{commit}", workspace.exec.call_args_list[0].args[0])
        await restore_tests(SimpleNamespace(exec=AsyncMock(return_value=CommandResult(0, "", ""))),
                            refs, ["tests/test_x.py::test_x"])

    async def test_wrong_branch_layout_is_an_environment_failure(self) -> None:
        outputs = ["a" * 40, "", "unrelated commit", "b" * 40, "Bug Patch"]
        workspace = SimpleNamespace(exec=AsyncMock(side_effect=[CommandResult(0, value, "") for value in outputs]))
        with self.assertRaisesRegex(RuntimeError, "layout"):
            await checkout_problem(workspace, {"instance_id": "task", "FAIL_TO_PASS": ["test_x.py::test_x"]})

    def test_test_paths_reject_options_traversal_and_absolute_paths(self) -> None:
        for node in ["../test.py::test", "/tmp/test.py::test", "--disable-warnings", "a\\b.py::test"]:
            with self.subTest(node=node), self.assertRaises(ValueError):
                selected_test_paths([node])
