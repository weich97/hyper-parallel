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
"""Regressions for per-instance forward identity in source emission."""

import importlib.util
import sys
from types import SimpleNamespace

from hyper_parallel.codegen.emit.recorded import materialize_recorded_forwards
from hyper_parallel.codegen.emit.snapshot_patches import emission_patches_from_snapshot_payload
from hyper_parallel.codegen.plan.observation import observe_emission_plan
from tests.common.mark_utils import arg_mark


_SOURCE = '''from types import MethodType
from torch import nn

class SharedMoe(nn.Module):
    def forward(self, value):
        return value

def backbone_forward(self, value):
    return value + 1

def draft_forward(self, value):
    return value + 2

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.mlp = SharedMoe()
        self.backbone.mlp.forward = MethodType(backbone_forward, self.backbone.mlp)
        self.draft = nn.Module()
        mlp = SharedMoe()
        mlp.forward = MethodType(draft_forward, mlp)
        self.draft.mlp = mlp
        self.plain = SharedMoe()
'''


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_instance_forwards_survive_observation_and_materialize_separate_contracts(tmp_path, monkeypatch):
    """Feature: Source-owned forward specialization for instances sharing a class.

    Description: Observe and emit two MoE paths with different bindings and distributed compute contracts.
    Expectation: Each instance keeps its own forward and receives a separate implementation wrapper.
    """
    source = tmp_path / "modeling_bound_toy.py"
    source.write_text(_SOURCE, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("modeling_bound_toy", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    model = module.ToyModel()
    plan = SimpleNamespace(modules={
        "backbone.mlp": SimpleNamespace(local_compute_fn={"_target_": "demo.backend.backbone"}, is_boundary=True),
        "draft.mlp": SimpleNamespace(local_compute_fn={"_target_": "demo.backend.draft"}, is_boundary=True),
        "plain": SimpleNamespace(is_boundary=True),
    })
    observed = observe_emission_plan(plan, model).to_dict()
    assert observed["boundary_forwards"] == {
        "backbone.mlp": "modeling_bound_toy:backbone_forward",
        "draft.mlp": "modeling_bound_toy:draft_forward",
    }
    payload = {
        "source": {"module_name": spec.name, "file_path": str(source)},
        "replacements": {"applications": []}, "plan": observed,
    }
    patches = emission_patches_from_snapshot_payload(payload)["local_computes"]
    records = []
    for item in patches:
        function_name = item["original_forward"].split(":")[1]
        records.append(SimpleNamespace(
            module_fqn=item["module_fqn"], module_type=module.SharedMoe,
            original_forward=getattr(module, function_name), factory=item["factory"],
        ))
    meta = SimpleNamespace(**observed, external_state_classes=[])
    rendered = materialize_recorded_forwards(
        _SOURCE, meta, patch_plan=SimpleNamespace(local_computes=records), source_path=str(source))
    assert "def backbone_forward_hp_impl(" in rendered
    assert "def draft_forward_hp_impl(" in rendered
    assert "self.backbone.mlp._forward_impl = MethodType(backbone_forward_hp_impl" in rendered
    assert "mlp._forward_impl = MethodType(draft_forward_hp_impl" in rendered
    compile(rendered, str(source), "exec")
