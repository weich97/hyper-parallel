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
"""Verify DSpark stage ownership at the real FSDP unit-selection boundary."""
# pylint: disable=protected-access

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from torch import nn

from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.models.build_options import FSDP2Config
from hyper_parallel.models.deepseek_v41.adapter.policies.sharding import (
    get_fsdp_execution_order,
    get_fsdp_excluded_subtrees,
    get_fsdp_wrap_modules,
)
from tests.common.mark_utils import arg_mark


class TestDSparkSharding(unittest.TestCase):
    """Keep dense stage ownership independent of expert ownership and the root."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_stage_weights_reshard_independently(self):
        """Feature: DSpark FSDP. Description: select nested units. Expectation: dense and EP owners differ."""
        model = nn.Module()
        model.model = nn.Module()
        model.model.gradient_checkpointing = False
        model.model.layers = nn.ModuleList([nn.Linear(4, 4)])
        model.dspark = nn.Module()
        model.dspark.main_proj = nn.Linear(4, 4)
        model.dspark.stages = nn.ModuleList([nn.Module() for _ in range(3)])
        metadata = {}
        for stage in model.dspark.stages:
            stage.attention = nn.Linear(4, 4)
            stage.mlp = nn.Module()
            stage.mlp.experts = nn.Linear(4, 4, bias=False)
            stage.mlp.shared_experts = nn.Linear(4, 4)
            metadata[stage.mlp.experts.weight] = SimpleNamespace(
                mesh=SimpleNamespace(mesh_dim_names=("edp_shard", "ep")))
        manager = FSDP2Manager(FSDP2Config(), SimpleNamespace(fsdp_moe_mesh=object(), tp_size=1))
        adapter = SimpleNamespace(fsdp_wrap_modules=get_fsdp_wrap_modules,
                                  fsdp_excluded_subtrees=get_fsdp_excluded_subtrees)
        with patch.object(FSDP2Manager, "_get_model_adapter_spec", return_value=adapter):
            units = manager._find_wrap_modules(model, metadata)
        owners = manager._resolve_parameter_owners(model, units)
        self.assertIs(owners[model.dspark.main_proj.weight], model)
        for stage in model.dspark.stages:
            self.assertIs(owners[stage.attention.weight], stage)
            self.assertIs(owners[stage.mlp.shared_experts.weight], stage)
            self.assertIs(owners[stage.mlp.experts.weight], stage.mlp.experts)
        ordered = get_fsdp_execution_order(model, tuple(unit.fqn for unit in units))
        self.assertEqual(ordered, ("model.layers.0", "dspark.stages.0", "dspark.stages.0.mlp.experts",
                                   "dspark.stages.1", "dspark.stages.1.mlp.experts",
                                   "dspark.stages.2", "dspark.stages.2.mlp.experts"))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_text_only_model_does_not_add_draft_units(self):
        """Feature: optional drafter. Description: omit DSpark. Expectation: no nonexistent FSDP units."""
        model = nn.Module()
        model.model = nn.Module()
        model.model.layers = nn.ModuleList([nn.Linear(4, 4)])
        self.assertEqual(get_fsdp_wrap_modules(model), ())
        self.assertEqual(get_fsdp_execution_order(model, ("model.layers.0",)), ("model.layers.0",))
