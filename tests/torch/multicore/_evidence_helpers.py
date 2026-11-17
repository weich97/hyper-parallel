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
"""Framework-free source identity and rank-log preservation for multicore ST."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.common.port_utils import allocate_port

EXTENDED_LEVEL1 = os.getenv("HP_MEGA_MOE_LEVEL1_EXTENDED", "0") == "1"


def source_revision(root: Path) -> dict[str, str | None]:
    """Record optional Git metadata without hiding other evidence failures."""
    revision, error = None, None
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.PIPE,
        ).strip()
    except subprocess.CalledProcessError as exc:
        error = f"git rev-parse HEAD exited with status {exc.returncode}: {(exc.stderr or '').strip()}"
    except OSError as exc:
        error = f"git rev-parse HEAD could not run: {exc}"
    return {"source_sha": revision, "source_sha_error": error}


def run_acceptance_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str, cards: int, launch: Callable[[Path], None],
    heap_bytes: int = 64 * 1024 * 1024,
    variant: str | None = None,
) -> dict:
    """Isolate one acceptance invocation and require its fresh rank evidence."""
    mode = os.getenv("HP_MEGA_MOE_DISPATCH_MODE", "push")
    variant = variant or os.getenv("HP_MEGA_MOE_GROWTH_CASE", os.getenv("TASK_QUEUE_ENABLE", "default"))
    result_dir = Path(os.getenv("HP_MEGA_MOE_EVIDENCE_DIR", str(tmp_path))) / case / mode / variant
    result_path = result_dir / f"{case}.json"
    result_path.unlink(missing_ok=True)
    monkeypatch.setenv("HP_DYNAMIC_DISPATCH", mode)
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(cards))
    monkeypatch.setenv("HP_MEGA_MOE_LEVEL1_RESULT", str(result_path))
    monkeypatch.setenv("HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{allocate_port()}")
    if cards == 2:
        monkeypatch.setenv("HYPER_PARALLEL_SHMEM_HEAP_SIZE", str(heap_bytes))
    else:
        monkeypatch.delenv("HYPER_PARALLEL_SHMEM_HEAP_SIZE", raising=False)
    launch(result_dir)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert len(result["ranks"]) == cards, f"expected {cards} rank records, got {len(result['ranks'])}."
    return result


@contextmanager
def capture_worker_logs(worker: str, case: str, destination: Path) -> Iterator[None]:
    """Archive this invocation's rank logs and show bounded tails on failure."""
    # Match the existing torchrun launcher path, including absolute worker names.
    log_root = Path(f"./logs/{worker}/{case}")
    previous = {path: (path.stat().st_mtime_ns, path.stat().st_size) for path in log_root.rglob("*.log")}
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        try:
            for path in sorted(log_root.rglob("*.log")):
                if path.name not in ("stdout.log", "stderr.log"):
                    continue
                stat = path.stat()
                if previous.get(path) == (stat.st_mtime_ns, stat.st_size):
                    continue
                target = destination / path.relative_to(log_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                if failed:
                    with path.open("rb") as stream:
                        stream.seek(max(0, path.stat().st_size - 16384))
                        tail = stream.read().decode("utf-8", errors="replace").splitlines()[-60:]
                    print(f"MegaMoe worker log: {target}\n" + "\n".join(tail), file=sys.stderr, flush=True)
        except OSError as exc:
            # Diagnostics must preserve the original distributed failure.
            print(f"MegaMoe log archive unavailable: {exc}", file=sys.stderr, flush=True)
