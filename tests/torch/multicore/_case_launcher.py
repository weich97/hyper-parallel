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
"""Framework-free device isolation for multicore system-test workers."""

import fcntl
import os
from contextlib import ExitStack
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from tests.common.distributed_launcher import torchrun_case
from tests.torch.multicore._test_env import INHERITED_RANK_VARIABLES

def run_multicore_case(
    file_name: str, case_name: str, num_proc: int = 1, timeout: float | None = 420,
) -> None:
    """Reserve physical NPUs before starting the bounded worker process group.

    Device indices follow the inherited ASCEND_RT_VISIBLE_DEVICES allocation.
    Only multicore launchers share these locks; unrelated tests are unaffected.
    Lock waiting is excluded from the worker deadline.
    """
    if num_proc < 1:
        raise ValueError("num_proc must be positive")
    env = os.environ.copy()
    cpu = env.get("HYPER_PARALLEL_TEST_DEVICE_TYPE", "").strip().lower() == "cpu"
    allocated = None if cpu else env.get("ASCEND_RT_VISIBLE_DEVICES")
    devices = list(map(int, allocated.split(","))) if allocated else list(range(num_proc))
    if cpu:
        devices = []
    if not cpu and (len(devices) < num_proc or min(devices) < 0 or len(set(devices)) != len(devices)):
        raise ValueError("invalid or insufficient ASCEND_RT_VISIBLE_DEVICES allocation")
    devices = devices[:num_proc]
    if cpu:
        env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    else:
        env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    for variable in INHERITED_RANK_VARIABLES:
        env.pop(variable, None)
    # CI may invoke pytest from tests/ without placing the archive root in its environment.
    root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (root, env.get("PYTHONPATH"))))
    command = [sys.executable, "-m", "tests.torch.multicore._case_launcher", file_name, case_name, str(num_proc)]
    started = time.monotonic()
    print(f"Multicore {case_name}: waiting for devices={devices}", flush=True)
    with ExitStack() as locks:
        lock_files = []
        # A fixed order prevents deadlock when two allocations partly overlap.
        for device in sorted(devices):
            path = Path(tempfile.gettempdir()) / f"hp_multicore_npu_{os.getuid()}_{device}.lock"
            lock = locks.enter_context(path.open("a", encoding="utf-8"))
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            lock_files.append(lock)
        print(
            f"Multicore {case_name}: acquired devices={devices}, waited={time.monotonic() - started:.3f}s", flush=True,
        )
        # The supervisor retains leases if its pytest parent is interrupted.
        with subprocess.Popen(
            command, env=env, start_new_session=True, pass_fds=tuple(lock.fileno() for lock in lock_files),
        ) as process:
            try:
                returncode = process.wait(timeout=timeout)
            finally:
                # Release devices only after workers and any surviving descendants stop.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)


if __name__ == "__main__":
    torchrun_case(sys.argv[1], sys.argv[2], num_proc=int(sys.argv[3]))
