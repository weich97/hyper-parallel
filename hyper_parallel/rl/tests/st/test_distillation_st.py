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
"""Explicit real OPD/MOPD launcher; importing it loads no training framework."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distillation_training() -> None:
    """Require real teacher supervision, nonzero updates and published policy versions."""
    config = os.environ.get("RL_OPD_ST_CONFIG")
    if not config:
        pytest.skip("Set RL_OPD_ST_CONFIG to a prepared real-teacher training recipe")
    worker = Path(__file__).with_name("_distillation_st.py")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            os.environ.get("RL_OPD_ST_NPROC", "1"),
            str(worker),
            config,
        ],
        check=True,
    )
