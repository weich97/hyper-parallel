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
"""Lightweight launchers for host-sharded visual input and memory validation."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


_WORKER = str(Path(__file__).with_name("_test_vision_memory.py"))


@arg_mark(["cpu_linux"], "level1", "allcards", "essential")
def test_cpu_transport():
    """Feature: host-sharded vision input.

    Description: Launch four Gloo workers across multiple TP/CP layouts.
    Expectation: Features and gradients match the previous device input path.
    """
    torchrun_case(_WORKER, "test_cpu_transport", num_proc=4)


@arg_mark(["platform_ascend910b"], "level1", "allcards", "essential")
def test_npu_transport():
    """Feature: host-sharded vision input on NPU.

    Description: Launch four HCCL workers across multiple TP/CP layouts.
    Expectation: Features and gradients match the previous device input path.
    """
    torchrun_case(_WORKER, "test_npu_transport", num_proc=4)


@arg_mark(["platform_ascend910b"], "level1", "allcards", "essential")
def test_npu_input_memory():
    """Feature: Encoder-CP input memory reduction.

    Description: Launch two HCCL workers with a 96 MiB CPU image payload.
    Expectation: Local-shard input peak saves at least one third of the payload.
    """
    torchrun_case(_WORKER, "test_npu_input_memory", num_proc=2)
