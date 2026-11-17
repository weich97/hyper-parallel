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
"""Shared step actions and capture-window scheduling for diagnostic tools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class DiagnosticAction(Enum):
    """Action selected for one user-visible training step."""

    NONE = 0
    WARMUP = 1
    RECORD = 2
    RECORD_AND_SAVE = 3


@dataclass(frozen=True)
class _DiagnosticSchedule:
    """Callable schedule with Torch-profiler-compatible window arguments."""

    skip_first: int
    wait: int
    warmup: int
    active: int
    repeat: int

    def __call__(self, step: int) -> DiagnosticAction:
        """Return the capture action for one zero-based training step."""
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError(f"step must be a non-negative integer, got {step!r}")
        if step < self.skip_first:
            return DiagnosticAction.NONE

        relative_step = step - self.skip_first
        cycle_length = self.wait + self.warmup + self.active
        cycle_index, cycle_step = divmod(relative_step, cycle_length)
        if self.repeat and cycle_index >= self.repeat:
            return DiagnosticAction.NONE
        if cycle_step < self.wait:
            return DiagnosticAction.NONE
        if cycle_step < self.wait + self.warmup:
            return DiagnosticAction.WARMUP
        if cycle_step == cycle_length - 1:
            return DiagnosticAction.RECORD_AND_SAVE
        return DiagnosticAction.RECORD


def schedule(
    *,
    wait: int,
    warmup: int,
    active: int,
    repeat: int = 0,
    skip_first: int = 0,
) -> Callable[[int], DiagnosticAction]:
    """Create a repeatable MegaKernel capture schedule.

    Args:
        wait: Unrecorded steps between capture windows.
        warmup: Preparatory steps before each active window. Profiling records
            and discards cycles; Tensor dump does not copy values in these steps.
        active: Steps retained in every active window.
        repeat: Number of cycles. Zero repeats indefinitely.
        skip_first: Initial unrecorded steps before the first cycle.

    Returns:
        Callable mapping a zero-based step number to DiagnosticAction.
    """
    values = {
        "wait": wait,
        "warmup": warmup,
        "active": active,
        "repeat": repeat,
        "skip_first": skip_first,
    }
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    if active <= 0:
        raise ValueError(f"active must be positive, got {active}")
    return _DiagnosticSchedule(
        skip_first=skip_first,
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=repeat,
    )
