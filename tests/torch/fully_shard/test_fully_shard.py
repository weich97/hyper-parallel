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
"""test fully_shard api"""

import os

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_TEST_FULLY_SHARD = "_test_fully_shard.py"
_TEST_HOOK_STATE = os.path.join(os.path.dirname(__file__), "_test_fully_shard_hook_state_machine.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_fully_shard_group1():
    """
    Feature: parallel run case in fully_shard
    Description:
        1.test_fully_shard_01
        2.test_fully_shard_02
        3.FSDP backward state and gradient accumulation on CPU/Gloo
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_FULLY_SHARD, "test_fully_shard_01", 12342, 4),
        TorchCase(_TEST_FULLY_SHARD, "test_fully_shard_02", 12400, 4),
    ])
    parallel_run([
        TorchCase(_TEST_HOOK_STATE, "test_torch_fully_shard_hook_state_machine", num_proc=2),
    ], global_num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_fully_shard_group2():
    """
    Feature: parallel run case in fully_shard
    Description:
        1.test_fully_shard_03
        2.test_fully_shard_04
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_FULLY_SHARD, "test_fully_shard_03", 12345, 4),
        TorchCase(_TEST_FULLY_SHARD, "test_fully_shard_04", 12346, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_fully_shard_group3():
    """
    Feature: parallel run case in fully_shard
    Description:
        1.test_fully_shard_from_group_mesh
        2.test_fully_shard_none_mesh
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_FULLY_SHARD, "test_fully_shard_from_group_mesh", 12343, 4),
        TorchCase(_TEST_FULLY_SHARD, "test_fully_shard_none_mesh", 12344, 4),
    ])
