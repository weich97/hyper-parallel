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
"""Independent SWE-Smith grading of frozen source changes."""

import asyncio
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence

from rl.agentic.core.types import RewardResult
from rl.agentic.envs.docker_workspace import DockerWorkspace, InvalidSubmissionError
from examples.swe_smith.artifacts import build_submission
from examples.swe_smith.prepare_data import record_path, validate_instance
from examples.swe_smith.workspace import checked, checkout_problem, prepare_candidate, restore_tests, test_paths


_LOGGER = logging.getLogger(__name__)
_STATUS = re.compile(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(.+)$")
_VERBOSE = re.compile(r"^(.+?)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)(?:\s+.*)?$")
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_COLLECTION_SOURCE_ERRORS = frozenset({
    "NameError", "UnboundLocalError", "ImportError", "ModuleNotFoundError", "AttributeError",
    "TypeError", "ValueError", "AssertionError", "ZeroDivisionError", "IndexError", "KeyError",
})
_SYNTAX = """
import json,pathlib,sys
errors=[]
for name in json.loads(sys.stdin.read()):
    try:
        compile((pathlib.Path('/testbed')/name).read_bytes(),name,'exec',dont_inherit=True)
    except SyntaxError as error:
        errors.append({'path':name,'line':error.lineno,'message':error.msg})
print(json.dumps(errors))
"""
_COLLECT = """
import json,pathlib,sys,traceback
import pytest

class Audit:
    def __init__(self):
        self.errors=[]
    def pytest_exception_interact(self,node,call,report):
        if report.when=='collect' and call.excinfo is not None:
            error=call.excinfo.value
            while error.__cause__ is not None or error.__context__ is not None:
                error=error.__cause__ or error.__context__
            frame=traceback.extract_tb(error.__traceback__)[-1]
            self.errors.append({'type':type(error).__name__,'path':frame.filename,
                                'line':frame.lineno,'message':str(error)})
    def pytest_sessionfinish(self,session,exitstatus):
        pathlib.Path(sys.argv[1]).write_text(json.dumps({
            'exitstatus':int(exitstatus),'errors':self.errors,
            'nodes':[item.nodeid for item in session.items]}))

# The trusted runner lives outside the worktree; pytest must still import task source.
sys.path.insert(0,'/testbed')
raise SystemExit(pytest.main(['--collect-only','--color=no','-q','-p','no:xdist',
                             '-p','no:cacheprovider',*sys.argv[2:]],plugins=[Audit()]))
"""


def evaluation_nodes(instance: Mapping[str, Any], f2p_only: bool = True) -> list[str]:
    """Match Agent Lightning's F2P and same-file P2P scope by exact file identity."""
    fail = list(instance["FAIL_TO_PASS"])
    files = set(test_paths(fail))
    passed = [node for node in instance["PASS_TO_PASS"] if not f2p_only or node.split("::", 1)[0] in files]
    nodes = fail + passed
    test_paths(nodes)
    return nodes


def normalize_test_nodes(instance: Mapping[str, Any], path_map: Mapping[str, str]) -> dict[str, Any]:
    """Resolve declared image-specific paths without allowing testbed traversal."""
    if not isinstance(path_map, Mapping):
        raise ValueError("Image test_path_map must be a mapping")
    normalized = dict(instance)
    for field in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        nodes = []
        for node in instance[field]:
            path, separator, suffix = node.partition("::")
            resolved = path_map.get(path, path)
            if not isinstance(resolved, str):
                raise ValueError("Image test paths must resolve to strings")
            test_paths([resolved])
            nodes.append(resolved + separator + suffix)
        normalized[field] = nodes
    return validate_instance(normalized)


def parse_statuses(output: str, expected_nodes: Sequence[str] | None = None) -> dict[str, str]:
    """Parse pytest -rA status lines, rejecting contradictory duplicate statuses."""
    statuses = {}
    for line in _ANSI.sub("", output).splitlines():
        line = line.strip()
        summary, verbose = _STATUS.match(line), _VERBOSE.match(line)
        if summary or verbose:
            status, node = (summary[1], summary[2]) if summary else (verbose[2], verbose[1])
            if summary:
                matches = [item for item in expected_nodes or () if node == item or node.startswith(item + " - ")]
                node = matches[0] if len(matches) == 1 else node.split(" - ", 1)[0]
            if expected_nodes is not None and node not in expected_nodes:
                continue
            if node in statuses and statuses[node] != status:
                raise RuntimeError(f"Contradictory test status: {node}")
            statuses[node] = status
    return statuses


class SWESmithTask:
    """Own trusted task, image and evaluator identity separately from candidate tools."""

    def __init__(self, prompt: Any, config: Mapping[str, Any]) -> None:
        """Validate task and image manifests before creating any candidate container."""
        truth = prompt.ground_truth
        if (prompt.metadata.get("task_type") != "swe_smith" or not isinstance(truth, Mapping)
                or set(truth) != {"instance_id", "record_sha256"}):
            raise ValueError("SWE-Smith requires the trusted prepared task identity")
        raw = record_path(Path(config["data_dir"]), truth["instance_id"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != truth["record_sha256"]:
            raise ValueError("Prepared SWE-Smith task identity changed")
        self.instance = validate_instance(json.loads(raw))
        if self.instance["instance_id"] != truth["instance_id"]:
            raise ValueError("Prepared SWE-Smith instance identity differs")
        self._load_image(config)
        self.timeout = float(config.get("run_timeout", 120))
        self.output_limit = int(config.get("output_limit_bytes", 8 * 1024 * 1024))
        self.artifact_limits = {"max_bytes": int(config.get("artifact_max_bytes", 128 * 1024 * 1024)),
                                "max_files": int(config.get("artifact_max_files", 65536))}
        self.workers = int(config.get("pytest_workers", 1))
        scope = config.get("f2p_only", True)
        if (not isinstance(scope, bool) or not math.isfinite(self.timeout) or self.timeout <= 0
                or self.output_limit <= 0 or self.workers <= 0 or min(self.artifact_limits.values()) <= 0):
            raise ValueError("SWE-Smith scope and finite positive evaluator budgets are required")
        self.nodes = evaluation_nodes(self.instance, scope)
        self.f2p_only = scope
        self.resources = {key: config[key] for key in
                          ("memory", "cpus", "pids_limit", "run_id", "docker_host") if key in config}
        self.baseline = Path(config["artifact_dir"]) / "baseline.tar"
        self.refs: dict[str, str] = {}

    def _load_image(self, config: Mapping[str, Any]) -> None:
        """Resolve the verified image manifest and its trusted evaluator executable."""
        image_raw = Path(config["images_path"]).read_bytes()
        self.images_hash = hashlib.sha256(image_raw).hexdigest()
        if self.images_hash != config["images_sha256"]:
            raise ValueError("SWE-Smith image manifest identity changed")
        images = json.loads(image_raw)
        if images.get("schema_version") != 1 or images.get("complete") is not True:
            raise ValueError("Unsupported SWE-Smith image manifest")
        image = images["images"][self.instance["image_name"]]
        self.instance = normalize_test_nodes(self.instance, image.get("test_path_map", {}))
        self.workspace_image = image["image"]
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(self.workspace_image)):
            raise ValueError("SWE-Smith requires a fixed local image ID")
        self.python = image["python"]
        if not isinstance(self.python, str) or not self.python.startswith("/"):
            raise ValueError("SWE-Smith requires a verified absolute Python executable")

    async def prepare(self, workspace: DockerWorkspace) -> None:
        """Snapshot trusted problem state before starting model-controlled tools."""
        if workspace.image != self.workspace_image:
            raise RuntimeError("Candidate image differs from the trusted task image")
        self.refs = await prepare_candidate(workspace, self.instance)
        await workspace.snapshot(self.baseline, **self.artifact_limits)

    async def _collect_tests(self, grader: DockerWorkspace, phase: str, log_path: Path) -> dict[str, Any]:
        """Capture collection exceptions through pytest hooks rather than parsing display text."""
        report_path = f"/workspace/{phase}-collection.json"
        result = await grader.exec([self.python, "/workspace/collect_tests.py", report_path, *self.nodes],
                                   cwd="/testbed", timeout=self.timeout, output_limit_bytes=self.output_limit)
        log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
        report = await checked(grader, [self.python, "-I", "-c",
                                       "import pathlib,sys;print(pathlib.Path(sys.argv[1]).read_text())", report_path],
                               cwd="/")
        value = json.loads(report.stdout)
        if (not isinstance(value, dict) or value.get("exitstatus") != result.returncode
                or not isinstance(value.get("errors"), list) or not isinstance(value.get("nodes"), list)):
            raise RuntimeError("SWE-Smith collection audit is incomplete")
        return value

    async def _candidate_collection_failure(self, grader: DockerWorkspace, changed_sources: Sequence[str],
                                            log_path: Path) -> list[dict[str, Any]]:
        """Attribute collection failure only when changed source fails and the clean baseline collects."""
        candidate = await self._collect_tests(grader, "candidate", log_path.with_suffix(".collection.log"))
        errors = candidate["errors"]
        allowed = {str(Path("/testbed") / name) for name in changed_sources}
        if (candidate["exitstatus"] not in (2, 4) or not errors
                or any(not isinstance(error, dict) or error.get("path") not in allowed
                       or error.get("type") not in _COLLECTION_SOURCE_ERRORS for error in errors)):
            raise RuntimeError("SWE-Smith collection failure is not attributable to changed source")
        refs = await checkout_problem(grader, self.instance)
        if refs != self.refs:
            raise RuntimeError("Collection baseline problem/test commits differ")
        await checked(grader, ["git", "-c", "safe.directory=/testbed", "clean", "-fd"], cwd="/testbed")
        await restore_tests(grader, refs, self.nodes)
        baseline = await self._collect_tests(grader, "baseline", log_path.with_suffix(".baseline-collection.log"))
        if baseline["exitstatus"] or baseline["errors"] or not set(self.nodes).issubset(baseline["nodes"]):
            raise RuntimeError("Trusted problem baseline cannot collect every selected test")
        return errors

    async def _run_tests(self, grader: DockerWorkspace, changed_sources: Sequence[str],
                         log_path: Path) -> dict[str, Any]:
        """Run selected tests and attribute incomplete collection before assigning reward."""
        probe = await grader.exec([self.python, "-c", "import xdist"], cwd="/testbed")
        parallel = ["-n", str(self.workers)] if probe.returncode == 0 else ["-p", "no:xdist"]
        command = [self.python, "-m", "pytest", "--color=no", "-v", "-rA", "-p", "no:cacheprovider"]
        result = await grader.exec([*command, *parallel, *self.nodes], cwd="/testbed", timeout=self.timeout,
                                   output_limit_bytes=self.output_limit)
        log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
        statuses = parse_statuses(result.stdout + result.stderr, self.nodes)
        if result.returncode in (1, 2, 4) and not set(self.nodes).issubset(statuses):
            errors = await self._candidate_collection_failure(grader, changed_sources, log_path)
            return {"resolved": False, "status": "collection_error", "errors": errors,
                    "baseline_collection_verified": True, "statuses": statuses}
        if (result.returncode not in (0, 1) or not set(self.nodes).issubset(statuses)
                or any(statuses[node] not in {"PASSED", "XFAIL", "FAILED", "ERROR", "SKIPPED", "XPASS"}
                       for node in self.nodes)):
            raise RuntimeError("SWE-Smith evaluation did not report every selected test")
        resolved = all(statuses[node] in {"PASSED", "XFAIL"} for node in self.nodes)
        if resolved and result.returncode:
            raise RuntimeError("Passing statuses contradict pytest failure exit")
        return {"resolved": resolved, "status": "resolved" if resolved else "unresolved", "statuses": statuses}

    async def _evaluate_patch(self, patch: str, changes: Mapping[str, Sequence[str]],
                              log_path: Path) -> dict[str, Any]:
        """Apply the frozen source patch and grade it in an independent task container."""
        grader = DockerWorkspace(self.workspace_image, network="none", name_prefix="hyper-swe-grade", **self.resources)
        try:
            await grader.start()
            refs = await checkout_problem(grader, self.instance)
            if refs != self.refs:
                raise RuntimeError("Grader problem/test commits differ from the candidate baseline")
            with tempfile.TemporaryDirectory(prefix="hyper-smith-grade-") as directory:
                root = Path(directory)
                (root / "submission.patch").write_text(patch, encoding="utf-8")
                (root / "collect_tests.py").write_text(_COLLECT, encoding="utf-8")
                await grader.copy_in(root)
            before = sorted(name for name in changes["modified"] + changes["deleted"] if name.endswith((".py", ".pyi")))
            after = sorted(name for name in changes["modified"] + changes["added"] if name.endswith((".py", ".pyi")))
            baseline_syntax = await checked(grader, [self.python, "-I", "-c", _SYNTAX], cwd="/",
                                            stdin=json.dumps(before))
            if json.loads(baseline_syntax.stdout):
                raise RuntimeError("Trusted problem baseline has invalid Python syntax")
            if patch:
                git = ["git", "-c", "safe.directory=/testbed", "apply"]
                await checked(grader, [*git, "--check", "/workspace/submission.patch"], cwd="/testbed")
                await checked(grader, [*git, "/workspace/submission.patch"], cwd="/testbed")
            candidate_syntax = await checked(grader, [self.python, "-I", "-c", _SYNTAX], cwd="/",
                                             stdin=json.dumps(after))
            errors = json.loads(candidate_syntax.stdout)
            if errors:
                return {"resolved": False, "status": "syntax_error", "errors": errors, "statuses": {}}
            await restore_tests(grader, refs, self.nodes)
            return await self._run_tests(grader, after, log_path)
        finally:
            original = sys.exc_info()[1]
            try:
                await grader.close()
            except Exception:  # pylint: disable=broad-exception-caught
                if original is None:
                    raise
                _LOGGER.exception("SWE-Smith grader cleanup failed while preserving the primary error")

    async def evaluate(self, archive: Path) -> RewardResult:
        """Grade a frozen submission; unknown failures propagate rather than becoming zero."""
        metadata = {"instance_id": self.instance["instance_id"], "image": self.workspace_image,
                    "images_sha256": self.images_hash, "evaluator_version": "swe-smith-pytest-v3",
                    "f2p_only": self.f2p_only, **self.refs}
        try:
            build = asyncio.create_task(asyncio.to_thread(
                build_submission, archive, self.baseline,
                protected_paths=test_paths(self.instance["FAIL_TO_PASS"] + self.instance["PASS_TO_PASS"]),
                **self.artifact_limits,
            ))
            try:
                artifact = await asyncio.shield(build)
            except asyncio.CancelledError:
                await asyncio.gather(build, return_exceptions=True)
                raise
        except InvalidSubmissionError as error:
            return RewardResult(0., {"success": 0.}, {**metadata, "status": "invalid_submission", "reason": str(error)})
        patch = artifact.pop("patch")
        archive.with_suffix(".manifest.json").write_text(json.dumps(artifact, indent=2) + "\n")
        archive.with_suffix(".patch").write_text(patch)
        changes = artifact["changes"]
        report = await self._evaluate_patch(patch, changes, archive.with_suffix(".eval.log"))
        archive.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
        value = float(report["resolved"])
        return RewardResult(value, {"success": value}, {**metadata, **report, "artifact": str(archive),
                                                       "artifact_hash": artifact["content_hash"],
                                                       "patch_sha256": artifact["patch_sha256"]})


async def score_submission(archive: Path, task: SWESmithTask) -> RewardResult:
    """Score the frozen repository artifact in the SWE-Smith program's reward callback."""
    return await task.evaluate(archive)
