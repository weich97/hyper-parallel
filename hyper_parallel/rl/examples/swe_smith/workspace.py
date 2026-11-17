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
"""Prepare verified SWE-Smith bug branches and restore controller-owned tests."""

from pathlib import PurePosixPath
import re
from typing import Any, Mapping, Sequence

from rl.agentic.envs.docker_workspace import CommandResult, DockerWorkspace


async def checked(workspace: DockerWorkspace, argv: Sequence[str], **kwargs: Any) -> CommandResult:
    """Run trusted setup plumbing, where nonzero status is an environment error."""
    result = await workspace.exec(argv, **kwargs)
    if result.returncode:
        raise RuntimeError(f"SWE-Smith setup failed: {list(argv)!r}: {result.stderr[:1000]}")
    return result


def test_paths(nodes: Sequence[str]) -> list[str]:
    """Extract canonical test paths without accepting Git or pytest options."""
    paths = sorted({node.split("::", 1)[0] for node in nodes})
    for name in paths:
        path = PurePosixPath(name)
        if (not name or name.startswith("-") or path.is_absolute() or ".." in path.parts
                or str(path) != name or "\\" in name or "\x00" in name):
            raise ValueError(f"Invalid SWE-Smith test path: {name!r}")
    return paths


async def checkout_problem(workspace: DockerWorkspace, instance: Mapping) -> dict[str, str]:
    """Resolve and verify the two-commit problem branch before exposing repository state."""
    git = ["git", "-c", "safe.directory=/testbed"]
    identity = instance["instance_id"]
    branch = await checked(workspace, [*git, "rev-parse", "--verify", f"refs/heads/{identity}^{{commit}}"],
                           cwd="/testbed")
    commit = branch.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("Problem branch did not resolve to a full commit ID")
    await checked(workspace, [*git, "checkout", "--detach", "--force", commit], cwd="/testbed")
    message = await checked(workspace, [*git, "show", "-s", "--format=%s", commit], cwd="/testbed")
    parent = await checked(workspace, [*git, "rev-parse", f"{commit}^"], cwd="/testbed")
    test_commit = parent.stdout.strip()
    parent_message = await checked(workspace, [*git, "show", "-s", "--format=%s", test_commit], cwd="/testbed")
    if ("remove f2p tests" not in message.stdout.lower() or "bug patch" not in parent_message.stdout.lower()
            or not re.fullmatch(r"[0-9a-f]{40}", test_commit)):
        raise RuntimeError("Unsupported SWE-Smith problem/test branch layout")
    for name in test_paths(instance["FAIL_TO_PASS"]):
        await checked(workspace, [*git, "cat-file", "-e", f"{test_commit}:{name}"], cwd="/testbed")
    return {"problem_commit": commit, "test_commit": test_commit}


async def prepare_candidate(workspace: DockerWorkspace, instance: Mapping) -> dict[str, str]:
    """Copy only the problem worktree and remove candidate access to Git history."""
    refs = await checkout_problem(workspace, instance)
    await checked(workspace, ["bash", "-ec", "cp -a /testbed/. /workspace/; "
                             "rm -rf /workspace/.git; "
                             r"find /workspace -type d \( -name __pycache__ -o -name .pytest_cache \) "
                             "-prune -exec rm -rf {} +; rm -rf /testbed; ln -s /workspace /testbed"])
    return refs


async def restore_tests(workspace: DockerWorkspace, refs: Mapping[str, str], nodes: Sequence[str]) -> None:
    """Restore only trusted test files in the independent grader."""
    paths = test_paths(nodes)
    if not paths:
        raise ValueError("SWE-Smith evaluation has no test paths")
    await checked(workspace, ["git", "-c", "safe.directory=/testbed", "restore",
                             "--source", refs["test_commit"], "--", *paths], cwd="/testbed")
