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
"""Multiple task images, explicit download boundaries and manifest reuse."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from examples.swe_smith import prepare_images as images


class TestPrepareImages(unittest.TestCase):
    """No test starts Docker or accesses a registry."""

    def test_remote_architecture_comes_from_target_daemon(self) -> None:
        """An ARM controller can prepare native x86 tasks on a remote daemon."""
        endpoint = "ssh://user@127.0.0.1:2233"
        with patch.object(images.platform, "machine", return_value="aarch64"), \
                patch.object(images, "docker", return_value="x86_64\n") as docker:
            self.assertEqual(images.host_architecture(endpoint), "amd64")
        docker.assert_called_once_with("info", "--format", "{{.Architecture}}", docker_host=endpoint)

    def setUp(self) -> None:
        """Create trusted instance metadata without requiring a daemon."""
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.instances = self.root / "instances.jsonl"
        self.instances.write_text('\n'.join(json.dumps({"image_name": name}) for name in ["one", "one", "two"]))

    def test_plan_deduplicates_images_without_download_build_or_manifest_write(self) -> None:
        with patch.object(images, "inspect_local", return_value=None), patch.object(images, "docker") as docker:
            result = images.prepare_images(self.instances, self.root / "images.json", plan_only=True)
        self.assertEqual(list(result["images"]), ["one", "two"])
        self.assertFalse(result["complete"])
        docker.assert_not_called()
        self.assertFalse((self.root / "images.json").exists())

    def test_new_images_require_explicit_cli_and_pull(self) -> None:
        for cli, pull, error, message in [(None, True, ValueError, "requires --cli-image"),
                                          ("cli-runtime", False, RuntimeError, "before --pull")]:
            with self.subTest(cli=cli, pull=pull), \
                    patch.object(images, "inspect_local", return_value=None), patch.object(images, "docker") as docker:
                with self.assertRaisesRegex(error, message):
                    images.prepare_images(self.instances, self.root / "images.json", cli_image=cli, pull=pull)
                docker.assert_not_called()

    def test_remote_platform_mismatch_never_pulls_layers(self) -> None:
        """An incompatible remote image fails platform inspection before downloading layers."""
        metadata = {"Descriptor": {"platform": {"architecture": "amd64"}}}
        cli = {"Id": "sha256:" + "a" * 64, "Architecture": "arm64"}
        with patch.object(images, "host_architecture", return_value="arm64"), \
                patch.object(images, "inspect_local", side_effect=lambda name, **_kwargs:
                             cli if name == "cli-runtime" else None), \
                patch.object(images, "docker", return_value=json.dumps(metadata)) as docker:
            with self.assertRaisesRegex(RuntimeError, "no layers pulled"):
                images.prepare_images(self.instances, self.root / "images.json", pull=True, cli_image="cli-runtime")
        self.assertEqual(docker.call_count, 1)
        self.assertEqual(docker.call_args.args[:2], ("manifest", "inspect"))

    def test_local_native_overrides_and_successful_manifest_reuse(self) -> None:
        """Native overrides retain provenance and verified manifests reuse prepared images."""
        image = {"Id": "sha256:" + "a" * 64, "Architecture": "arm64", "Size": 100, "RepoDigests": []}
        output = self.root / "images.json"
        probe = 'codex-cli 0.152.1\n{"python":"/env/bin/python","pytest":"8.3.5"}\n'
        with patch.object(images, "host_architecture", return_value="arm64"), \
                patch.object(images, "inspect_local", return_value=image), \
                patch.object(images, "docker", return_value=probe) as docker:
            override = {"one": {"image": "native-one", "test_path_map": {"../dev/test_x.py": "tests/test_x.py"}}}
            result = images.prepare_images(self.instances, output, overrides=override, cli_image="cli-runtime")
            self.assertTrue(result["complete"])
            self.assertEqual(result["images"]["one"]["source"], "native-one")
            self.assertEqual(result["images"]["one"]["python"], "/env/bin/python")
            self.assertEqual(result["images"]["one"]["test_path_map"], override["one"]["test_path_map"])
            self.assertEqual(docker.call_count, 2)
            docker.reset_mock()
            images.prepare_images(self.instances, output, overrides=override)
            docker.assert_not_called()
