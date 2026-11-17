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
"""Prepare only selected SWE-Smith images, with explicit pull and platform checks."""

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping

from examples.swe_smith.prepare_data import iter_instances
from examples.swe_smith.workspace import test_paths


def docker(*args: str, timeout: float = 60, docker_host: str | None = None) -> str:
    """Run explicit Docker operations; no command implicitly enables a pull."""
    prefix = ["docker", *(["--host", docker_host] if docker_host is not None else [])]
    result = subprocess.run([*prefix, *args], capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"Docker {args[0]} failed: {result.stderr[:2000]}")
    return result.stdout


def inspect_local(image: str, *, docker_host: str | None = None) -> dict[str, Any] | None:
    """Read the selected daemon's image metadata without downloading layers."""
    prefix = ["docker", *(["--host", docker_host] if docker_host is not None else [])]
    result = subprocess.run([*prefix, "image", "inspect", image],
                            capture_output=True, text=True, timeout=30, check=False)
    if result.returncode:
        if "No such image" in result.stderr or "No such object" in result.stderr:
            return None
        raise RuntimeError(f"Docker image inspection failed: {result.stderr[:2000]}")
    return json.loads(result.stdout)[0]


def image_sources(instances: Path) -> list[str]:
    """Deduplicate actual image names while preserving task order."""
    return list(dict.fromkeys(row["image_name"] for row in iter_instances(instances)))


def host_architecture(docker_host: str | None = None) -> str:
    """Resolve the target architecture, querying the remote daemon when selected."""
    name = (platform.machine() if docker_host is None else
            docker("info", "--format", "{{.Architecture}}", docker_host=docker_host)).strip().lower()
    return {"aarch64": "arm64", "x86_64": "amd64"}.get(name, name)


def _check_remote_platform(image: str, architecture: str, *, docker_host: str | None = None) -> None:
    metadata = json.loads(docker("manifest", "inspect", "--verbose", image, docker_host=docker_host))
    records = metadata if isinstance(metadata, list) else [metadata]
    platforms = {row.get("Descriptor", {}).get("platform", {}).get("architecture") for row in records}
    if architecture not in platforms:
        raise RuntimeError(f"Image {image} has platforms {sorted(str(item) for item in platforms)}, "
                           f"but the target daemon requires {architecture}; no layers pulled")


def _image_profile(override: Any) -> tuple[str, dict[str, str]]:
    """Validate an image override and its trusted test-path mapping."""
    if not isinstance(override, (str, Mapping)):
        raise ValueError("Image overrides must be image strings or profile mappings")
    profile = {"image": override} if isinstance(override, str) else dict(override)
    source_name = profile.get("image")
    if (set(profile) - {"image", "test_path_map"} or not isinstance(source_name, str)
            or not source_name or source_name.startswith("-") or any(char.isspace() for char in source_name)):
        raise ValueError("Image overrides require image and optional test_path_map")
    path_map = profile.get("test_path_map", {})
    if (not isinstance(path_map, Mapping)
            or any(not isinstance(path, str) or not isinstance(target, str) for path, target in path_map.items())):
        raise ValueError("Image test_path_map must map dataset paths to trusted repository paths")
    test_paths(list(path_map.values()))
    return source_name, path_map


def _image_report(output: Path, plan_only: bool, names: list[str]) -> dict[str, Any]:
    """Reuse only selected entries from a compatible existing image manifest."""
    report: dict[str, Any] = {"schema_version": 1, "complete": False, "images": {}}
    if output.exists() and not plan_only:
        previous = json.loads(output.read_text())
        if previous.get("schema_version") != 1:
            raise ValueError("Unsupported image manifest schema")
        report["images"] = {name: value for name, value in previous.get("images", {}).items() if name in names}
    return report


def prepare_images(instances: Path, output: Path, *, pull: bool = False, plan_only: bool = False,
                   overrides: Mapping[str, str | Mapping[str, Any]] | None = None,
                   cli_image: str | None = None, docker_host: str | None = None) -> dict[str, Any]:
    """Resolve each task image independently and reuse verified prepared images.

    Overrides map dataset image names to locally prepared equivalent task images.
    Their provenance is retained; overrides do not alter dataset identities.
    """
    inspect = partial(inspect_local, docker_host=docker_host)
    names = image_sources(instances)
    if not names:
        raise ValueError("No task images selected")
    overrides = dict(overrides or {})
    if set(overrides) - set(names):
        raise ValueError("Image overrides contain unselected dataset images")
    architecture = host_architecture(docker_host)
    report = _image_report(output, plan_only, names)
    for name in names:
        override = overrides.get(name, name)
        source_name, path_map = _image_profile(override)
        source = inspect(source_name)
        if plan_only:
            report["images"][name] = {"source": source_name, "status": "local" if source else "missing",
                                      "bytes": None if source is None else source["Size"],
                                      "architecture": None if source is None else source["Architecture"]}
            continue
        if source is not None and source["Architecture"] != architecture:
            raise RuntimeError(f"Image {source_name} must support native {architecture}")
        existing = report["images"].get(name)
        if (existing and source is not None and existing.get("source_id") == source["Id"]
                and inspect(existing["image"]) is not None):
            if name in overrides:
                existing["test_path_map"] = dict(path_map)
            continue
        report["images"][name] = _prepare_task_image(
            source_name, source, path_map, architecture=architecture, cli_image=cli_image,
            pull=pull, docker_host=docker_host,
        )
        _write_manifest(output, report)
    if not plan_only:
        report["complete"] = True
        _write_manifest(output, report)
    return report


def _build_and_probe(source: Mapping[str, Any], cli: Mapping[str, Any], docker_host: str | None) -> tuple[dict, dict]:
    """Build offline from local image IDs and check the pinned CLI and Python runtime."""
    run = partial(docker, docker_host=docker_host)
    inspect = partial(inspect_local, docker_host=docker_host)
    key = hashlib.sha256((source["Id"] + cli["Id"]).encode()).hexdigest()[:20]
    target = "hyper-parallel/swe-smith:" + key
    if inspect(target) is None:
        run("build", "--pull=false", "--network=none", "--tag", target,
            "--build-arg", "TASK_IMAGE=" + source["Id"],
            "--build-arg", "CLI_IMAGE=" + cli["Id"], str(Path(__file__).parent), timeout=1800)
    built = inspect(target)
    probe = run("run", "--rm", "--network=none", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                "--entrypoint", "/bin/bash", built["Id"], "-lc",
                "codex --version && python -c 'import json,sys,pytest; "
                "print(json.dumps({\"python\":sys.executable,\"pytest\":pytest.__version__}))'")
    lines = probe.strip().splitlines()
    if len(lines) < 2 or lines[-2] != "codex-cli 0.152.1":
        raise RuntimeError("Prepared task image failed its pinned Codex probe")
    runtime = json.loads(lines[-1])
    return built, runtime


def _prepare_task_image(source_name: str, source: dict | None, path_map: Mapping[str, str], *,
                        architecture: str, cli_image: str | None, pull: bool,
                        docker_host: str | None) -> dict[str, Any]:
    """Resolve an explicitly permitted source and retain the prepared image provenance."""
    run = partial(docker, docker_host=docker_host)
    inspect = partial(inspect_local, docker_host=docker_host)
    if not cli_image:
        raise ValueError("Preparing new task images requires --cli-image with a local CLI runtime image")
    if source is None and not pull:
        raise RuntimeError(f"Required image {source_name} is missing; inspect size/platform before --pull")
    cli = inspect(cli_image)
    if cli is None or cli["Architecture"] != architecture:
        raise RuntimeError("A local CLI runtime image for the task architecture is required")
    if source is None:
        _check_remote_platform(source_name, architecture, docker_host=docker_host)
        run("pull", "--platform", "linux/" + architecture, source_name, timeout=1800)
        source = inspect(source_name)
    if source is None or source["Architecture"] != architecture:
        raise RuntimeError(f"Image {source_name} must support native {architecture}")
    built, runtime = _build_and_probe(source, cli, docker_host)
    return {"image": built["Id"], "source": source_name, "source_id": source["Id"],
            "source_digests": source.get("RepoDigests", []), "architecture": architecture,
            "size_bytes": built["Size"], "cli_image": cli["Id"],
            "test_path_map": dict(path_map), **runtime}


def _write_manifest(output: Path, report: Mapping[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def main() -> None:
    """Plan first or explicitly prepare only the images selected by the data stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--overrides", type=Path)
    parser.add_argument("--cli-image", help="Local CLI runtime image, required when preparing new task images")
    parser.add_argument("--docker-host", help="Explicit target Docker daemon, e.g. ssh://user@host:port")
    args = parser.parse_args()
    overrides = None if args.overrides is None else json.loads(args.overrides.read_text())
    report = prepare_images(args.instances, args.output, pull=args.pull, plan_only=args.plan_only,
                            overrides=overrides, cli_image=args.cli_image, docker_host=args.docker_host)
    print(json.dumps(report, indent=2))
    if not args.plan_only:
        print("images_sha256=" + hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
