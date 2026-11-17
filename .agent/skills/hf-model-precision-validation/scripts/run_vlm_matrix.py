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
"""Run a declared VLM Trainer matrix and verify every measured step exists."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Any


def _read_metrics(path: Path) -> list[int]:
    """Return structured global steps from one runtime evidence directory."""
    metrics = path / "cases" / "runtime" / "metrics.jsonl"
    if not metrics.is_file():
        return []
    return [json.loads(line)["step"] for line in metrics.read_text(encoding="utf-8").splitlines() if line]


def _write_result(path: Path, results: dict[str, Any]) -> None:
    """Persist progress atomically so a long matrix can be resumed."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _occupied_npu_devices() -> set[int]:
    """Read NPU process occupancy before starting a new eight-rank job."""
    result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True)
    occupied = set()
    for line in result.stdout.splitlines():
        match = re.match(r"\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|", line)
        if match:
            card, chip, _ = (int(value) for value in match.groups())
            occupied.add(card * 2 + chip)
    return occupied


def _source_sha256(repository: Path, cases: list[dict[str, Any]]) -> str:
    """Fingerprint training code and every case config before reusing evidence."""
    files = list((repository / "hyper_parallel").rglob("*.py"))
    files.append(repository / "scripts" / "train_vl.py")
    files.extend(Path(case["config"]).resolve() for case in cases)
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(repository)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _changed_assets(payload: dict[str, Any]) -> list[str]:
    """Find external model/data assets that changed after ledger creation."""
    changed = []
    for name, expected in payload.get("asset_sha256", {}).items():
        path = Path(payload["asset_paths"][name])
        if not path.exists() or _asset_sha256(path) != expected:
            changed.append(name)
    return changed


def _asset_sha256(path: Path) -> str:
    """Hash a file or each named file in a checkpoint directory."""
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
    for item in files:
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
        with item.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def _run_case(case: dict[str, Any], *, device_ids: list[int], port: int, timeout: int,
              repository: Path, source_sha256: str, ledger_sha256: str,
              previous: dict[str, Any]) -> dict[str, Any]:
    """Run one isolated Trainer process group and inspect its structured output."""
    evidence = Path(case["evidence_dir"]).resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    expected = case["measured_steps"]
    observed = _read_metrics(evidence)
    if observed == expected:
        if (previous.get("status") == "PASS" and previous.get("source_sha256") == source_sha256
                and previous.get("ledger_sha256") == ledger_sha256):
            return {"status": "PASS", "source": "existing", "steps": observed, "source_sha256": source_sha256}
        return {"status": "BLOCKED", "reason": "complete evidence has no matching source fingerprint",
                "steps": observed}
    if observed:
        return {"status": "BLOCKED", "reason": "partial evidence requires a fresh evidence directory",
                "steps": observed}
    if case["status"] in ("existing", "running"):
        return {"status": "BLOCKED", "reason": "declared existing run lacks the complete step range",
                "steps": observed}

    config = Path(case["config"]).resolve()
    if not config.is_file():
        return {"status": "BLOCKED", "reason": f"config is missing: {config}"}
    occupied = _occupied_npu_devices().intersection(device_ids)
    if occupied:
        return {"status": "BLOCKED", "reason": f"selected NPU devices are occupied: {sorted(occupied)}"}
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(value) for value in device_ids)
    environment["HYPER_PARALLEL_MODEL_INTEGRATION_OUTPUT_DIR"] = str(evidence)
    environment["HCCL_IF_BASE_PORT"] = str(port)
    command = ["torchrun", "--standalone", f"--nproc_per_node={len(device_ids)}",
               "scripts/train_vl.py", str(config)]
    log = evidence.parent / f"{case['case_id']}.log"
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as output, subprocess.Popen(
            command, cwd=repository, env=environment, stdout=output,
            stderr=subprocess.STDOUT, start_new_session=True,
    ) as process:
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return {"status": "BLOCKED", "reason": f"timeout after {timeout}s",
                    "log": str(log), "elapsed_seconds": round(time.monotonic() - started, 1),
                    "source_sha256": source_sha256}
    observed = _read_metrics(evidence)
    return {"status": "PASS" if returncode == 0 and observed == expected else "FAIL",
            "returncode": returncode, "steps": observed, "expected_steps": expected,
            "log": str(log), "elapsed_seconds": round(time.monotonic() - started, 1),
            "source_sha256": source_sha256}


def _validate_ledger(parser: argparse.ArgumentParser, payload: dict[str, Any],
                     selected: set[str] | None) -> None:
    """Reject a ledger whose devices, fingerprints, or case IDs are inconsistent."""
    device_ids = payload.get("device_ids")
    if (not isinstance(device_ids, list) or len(device_ids) != payload["devices"]
            or len(set(device_ids)) != len(device_ids)
            or any(isinstance(device_id, bool) or not isinstance(device_id, int)
                   or device_id < 0 for device_id in device_ids)):
        parser.error("ledger must declare exactly one distinct physical NPU ID per rank")
    repository = Path(payload["source_checkout"]).resolve()
    if payload.get("source_sha256") != _source_sha256(repository, payload["cases"]):
        parser.error("source or matrix configs changed after the ledger was frozen")
    if set(payload.get("asset_sha256", {})) != set(payload.get("asset_paths", {})):
        parser.error("asset_sha256 and asset_paths must name the same assets")
    if _changed_assets(payload):
        parser.error(f"matrix assets changed after the ledger was frozen: {_changed_assets(payload)}")
    if selected is not None:
        unknown = selected - {case["case_id"] for case in payload["cases"]}
        if unknown:
            parser.error(f"unknown case IDs: {sorted(unknown)}")


def _execute_cases(parser: argparse.ArgumentParser, arguments: argparse.Namespace,
                   payload: dict[str, Any], ledger_path: Path,
                   source_sha256: str, ledger_sha256: str,
                   selected: set[str] | None) -> dict[str, Any]:
    """Run selected cases under an exclusive device reservation lock."""
    cases = payload["cases"]
    repository = Path(payload["source_checkout"]).resolve()
    device_ids = payload["device_ids"]
    result_path = ledger_path.with_name("execution.json")
    results = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    with ledger_path.with_name("device-reservation.lock").open("w") as reservation:
        try:
            fcntl.flock(reservation, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("another matrix runner holds this device reservation") from error
        for index, case in enumerate(cases):
            name = case["case_id"]
            if selected is not None and name not in selected:
                continue
            if arguments.dry_run:
                print(f"{name}: {case['status']} -> {case['config']}", flush=True)
                continue
            port = arguments.base_port + index * 16
            if port > 65535:
                parser.error("case port range exceeds 65535")
            print(f"START {name} port={port} devices={device_ids}", flush=True)
            if _source_sha256(repository, cases) != source_sha256 or _changed_assets(payload):
                result = {"status": "BLOCKED", "reason": "source, config, or asset changed during matrix"}
            else:
                result = _run_case(case, device_ids=device_ids, port=port,
                                   timeout=arguments.timeout_seconds, repository=repository,
                                   source_sha256=source_sha256, ledger_sha256=ledger_sha256,
                                   previous=results.get(name, {}))
            result["source_sha256"] = source_sha256
            result["ledger_sha256"] = ledger_sha256
            results[name] = result
            _write_result(result_path, results)
            print(f"END {name} status={result['status']} steps={len(result.get('steps', []))}", flush=True)
    return results


def main() -> int:
    """Execute selected planned cases and record each result without hiding failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--select", nargs="*", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--base-port", type=int, default=28000)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    ledger_path = arguments.ledger.resolve()
    ledger_bytes = ledger_path.read_bytes()
    ledger_sha256 = hashlib.sha256(ledger_bytes).hexdigest()
    payload = json.loads(ledger_bytes)
    selected = set(arguments.select) if arguments.select is not None else None
    _validate_ledger(parser, payload, selected)
    source_sha256 = _source_sha256(Path(payload["source_checkout"]).resolve(), payload["cases"])
    results = _execute_cases(parser, arguments, payload, ledger_path,
                             source_sha256, ledger_sha256, selected)
    return 0 if all(row["status"] == "PASS" for row in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
