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
"""Frozen SWE-Smith artifact safety and exact textual patch contracts."""

import io
from pathlib import Path
import tarfile
import shutil
import subprocess
import tempfile
import unittest

from rl.agentic.envs.docker_workspace import InvalidSubmissionError

from examples.swe_smith.artifacts import build_submission


def _tar(path: Path, files: dict[str, bytes], executable: str = "") -> None:
    with tarfile.open(path, "w") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            member.mode = 0o755 if name == executable else 0o644
            archive.addfile(member, io.BytesIO(content))


class TestSubmissionArtifacts(unittest.TestCase):
    """Check meaningful edits and reject changes outside supported source semantics."""

    def setUp(self) -> None:
        """Create a trusted public baseline including executable source and protected tests."""
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory)
        self.base = Path(self.directory) / "base.tar"
        self.candidate = Path(self.directory) / "candidate.tar"
        self.files = {"src/_pytest/edit.py": b"old", "src/_pytest/delete.py": b"delete\n",
                      "src/_pytest/cacheprovider.py": b"cache\n", "testing/test_case.py": b"test\n",
                      "src/_pytest/_version.py": b"version\n", "pyproject.toml": b"[project]\nname='fixture'\n"}
        _tar(self.base, self.files, "src/_pytest/cacheprovider.py")

    def test_add_modify_delete_and_git_metadata_is_not_authoritative(self) -> None:
        """Include new files, deleted source and missing final newline despite fake Git."""
        candidate = dict(self.files)
        candidate["src/_pytest/edit.py"] = b"new"
        candidate["src/pytest/new.py"] = b"added\n"
        candidate.pop("src/_pytest/delete.py")
        candidate[".git/config"] = b"malicious metadata"
        candidate["src/pytest.egg-info/PKG-INFO"] = b"build data"
        candidate["src/_pytest/edit.py.bak"] = b"old source backup"
        candidate["sample.jsonl"] = b'{"value":1}\n'
        _tar(self.candidate, candidate, "src/_pytest/cacheprovider.py")
        result = build_submission(self.candidate, self.base, protected_paths=("src/_pytest/_version.py",))
        self.assertEqual(result["changes"], {"added": ["src/pytest/new.py"],
                                           "modified": ["src/_pytest/edit.py"],
                                           "deleted": ["src/_pytest/delete.py"]})
        self.assertIn("deleted file mode 100644", result["patch"])
        self.assertIn("new file mode 100644", result["patch"])
        self.assertIn("\\ No newline at end of file", result["patch"])
        self.assertNotIn(".git/config", result["files"])
        self.assertNotIn("PKG-INFO", result["patch"])
        self.assertNotIn("edit.py.bak", result["files"])
        self.assertNotIn("sample.jsonl", result["files"])
        replay = Path(self.directory) / "replay"
        for name, data in self.files.items():
            target = replay / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        applied = subprocess.run(["git", "apply", "--no-index", "--check", "-"], input=result["patch"], text=True,
                                 cwd=replay, capture_output=True, check=False)
        self.assertEqual(applied.returncode, 0, applied.stderr)
        subprocess.run(["git", "apply", "--no-index", "-"], input=result["patch"], text=True, cwd=replay, check=True)
        self.assertEqual((replay / "src/_pytest/edit.py").read_bytes(), b"new")
        self.assertEqual((replay / "src/pytest/new.py").read_bytes(), b"added\n")
        self.assertFalse((replay / "src/_pytest/delete.py").exists())

    def test_unchanged_executable_and_build_artifacts_produce_no_patch(self) -> None:
        """Preserve the trusted executable bit and ignore generated cache bytes."""
        candidate = {**self.files, ".pytest_cache/cache": b"cache", "src/_pytest/__pycache__/x.pyc": b"\0"}
        _tar(self.candidate, candidate, "src/_pytest/cacheprovider.py")
        result = build_submission(self.candidate, self.base, protected_paths=("src/_pytest/_version.py",))
        self.assertEqual(result["patch"], "")
        self.assertEqual(result["base_hash"], result["content_hash"])

    def test_protected_binary_and_mode_changes_reject(self) -> None:
        """Reject test/config changes, generated version changes, binary code and executable drift."""
        for name, data in (("testing/test_case.py", b"changed"), ("conftest.py", b"new"),
                           ("pyproject.toml", b"changed"),
                           ("src/_pytest/conftest.py", b"new"),
                           ("src/_pytest/_version.py", b"changed"), ("src/_pytest/edit.py", b"\0"),
                           ("src/_pytest/edit.py", b"\xff")):
            with self.subTest(name=name, data=data):
                _tar(self.candidate, {**self.files, name: data}, "src/_pytest/cacheprovider.py")
                with self.assertRaises(InvalidSubmissionError):
                    build_submission(self.candidate, self.base, protected_paths=("src/_pytest/_version.py",))
        _tar(self.candidate, self.files)
        with self.assertRaisesRegex(InvalidSubmissionError, "mode"):
            build_submission(self.candidate, self.base, protected_paths=("src/_pytest/_version.py",))
