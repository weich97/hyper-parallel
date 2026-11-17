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
"""Derive bounded source patches from frozen bytes against a trusted baseline."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile
from typing import Any, Mapping, Sequence

from rl.agentic.envs.docker_workspace import InvalidSubmissionError, read_snapshot


def _ignored(name: str) -> bool:
    return any(part in {".git", ".pytest_cache", "__pycache__"} or part.endswith(".egg-info")
               for part in PurePosixPath(name).parts)


def _read(archive: Path, max_bytes: int, max_files: int) -> tuple[dict[str, bytes], dict[str, int]]:
    files = read_snapshot(archive, max_bytes=max_bytes, max_files=max_files)
    files = {name: data for name, data in files.items() if not _ignored(name)}
    with tarfile.open(archive, "r:") as stream:
        modes = {str(PurePosixPath(member.name)): 0o755 if member.mode & 0o111 else 0o644
                 for member in stream if member.isfile() and str(PurePosixPath(member.name)) in files}
    return files, modes


def _hash(files: Mapping[str, bytes], modes: Mapping[str, int]) -> str:
    entries = {name: {"sha256": hashlib.sha256(data).hexdigest(), "mode": modes[name]}
               for name, data in sorted(files.items())}
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def _source(name: str, roots: Sequence[str]) -> bool:
    path = PurePosixPath(name)
    test_path = (path.name in {"setup.py", "conftest.py", "sitecustomize.py", "usercustomize.py"}
                 or path.name.startswith("test_") or path.name.endswith("_test.py"))
    test_path = test_path or any(part in {"test", "tests", "testing", "__tests__"} for part in path.parts)
    return (path.suffix in {".py", ".pyi"} and not test_path
            and any(path.is_relative_to(root) for root in roots))


def _text(data: bytes, name: str) -> None:
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise InvalidSubmissionError(f"Non-UTF8 source: {name}") from error
    if b"\0" in data:
        raise InvalidSubmissionError(f"Binary source: {name}")


def _git_patch(base: Mapping[str, bytes], candidate: Mapping[str, bytes], modes: Mapping[str, int],
               roots: Sequence[str]) -> str:
    """Build a source-only patch using a trusted temporary Git index."""
    with tempfile.TemporaryDirectory(prefix="hyper-swe-patch-") as directory:
        root = Path(directory)
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                    "GIT_ATTR_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"})

        def git(*args: str) -> str:
            """Run only trusted Git plumbing against controller-owned paths and index."""
            try:
                result = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "core.attributesFile=/dev/null",
                                         "-c", "core.autocrlf=false", *args], cwd=root, env=env,
                                        capture_output=True, check=False, timeout=30)
            except (OSError, subprocess.TimeoutExpired) as error:
                raise RuntimeError("Trusted patch generation could not run Git") from error
            if result.returncode:
                raise RuntimeError(f"Trusted patch generation failed: {result.stderr.decode(errors='replace')[:1000]}")
            return result.stdout.decode("utf-8")

        git("init", "--quiet", "--template=")
        for name, data in base.items():
            if _source(name, roots):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                target.chmod(modes[name])
        git("add", "--all", "--", ".")
        for name in set(base) | set(candidate):
            if not _source(name, roots):
                continue
            target = root / name
            if name not in candidate:
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(candidate[name])
                target.chmod(modes.get(name, 0o644))
        additions = sorted(name for name in candidate if name not in base and _source(name, roots))
        if additions:
            git("add", "--intent-to-add", "--", *additions)
        return git("diff", "--no-ext-diff", "--no-textconv", "--no-color",
                   "--src-prefix=a/", "--dst-prefix=b/", "--", ".")


def build_submission(candidate_archive: Path, baseline_archive: Path, *,
                     allowed_roots: Sequence[str] = (".",),
                     protected_paths: Sequence[str] = (),
                     max_bytes: int = 128 * 1024 * 1024, max_files: int = 65536) -> dict[str, Any]:
    """Build a strict textual patch without trusting candidate Git metadata.

    Baseline and candidate must be uncompressed snapshot tars. Only plain UTF-8
    Python source under trusted roots can change; executable-bit changes reject.
    New non-Python scratch files are validated but not submitted. Existing files
    and declared tests remain protected; build products are never replayed.
    Infrastructure failures propagate instead of becoming a zero-reward patch.
    """
    for name in (*allowed_roots, *protected_paths):
        path = PurePosixPath(name)
        if not name or path.is_absolute() or ".." in path.parts or str(path) != name or "\\" in name:
            raise ValueError("Source roots and protected paths must be canonical relative paths")
    base, base_modes = _read(Path(baseline_archive), max_bytes, max_files)
    files, modes = _read(Path(candidate_archive), max_bytes, max_files)
    files = {name: data for name, data in files.items()
             if name in base or name in protected_paths or PurePosixPath(name).suffix in {".py", ".pyi"}}
    modes = {name: mode for name, mode in modes.items() if name in files}
    changes = {"added": [], "modified": [], "deleted": []}
    for name in sorted(set(base) | set(files)):
        same_mode = base_modes.get(name) == modes.get(name)
        if base.get(name) == files.get(name) and same_mode:
            continue
        if not _source(name, allowed_roots) or name in protected_paths:
            raise InvalidSubmissionError(f"Protected path changed: {name}")
        if name in base and name in files and not same_mode:
            raise InvalidSubmissionError(f"Executable mode changed: {name}")
        if name not in base and modes[name] != 0o644:
            raise InvalidSubmissionError(f"Executable source added: {name}")
        for data in (base.get(name), files.get(name)):
            if data is not None:
                _text(data, name)
        operation = "added" if name not in base else "deleted" if name not in files else "modified"
        changes[operation].append(name)
    patch = _git_patch(base, files, base_modes, allowed_roots)
    return {"patch": patch, "base_hash": _hash(base, base_modes), "content_hash": _hash(files, modes),
            "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(), "changes": changes,
            "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())},
            "modes": modes}
