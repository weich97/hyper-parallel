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
"""Check expert staging ownership independently of the dense shard domain."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe import DeepseekV41MegaMoeExperts
from hyper_parallel.trainer.runtime.expert_grad_staging import build_expert_gradient_staging


class TestExpertGradientStagingSelection(unittest.TestCase):
    """Select only an owned expert shard that can safely expose its first partial."""

    @staticmethod
    def _base(dense_shards: int, expert_shards: int = 1) -> SimpleNamespace:
        experts = Mock(spec=DeepseekV41MegaMoeExperts)
        experts.gate_up_proj = torch.nn.Parameter(torch.zeros(2, 3))
        experts.down_proj = torch.nn.Parameter(torch.zeros(3, 2))
        return SimpleNamespace(
            config=SimpleNamespace(
                accelerator=SimpleNamespace(ep_size=128, cp_size=1, tp_size=1),
                fsdp_config=SimpleNamespace(
                    edp_shard_size=expert_shards, dp_shard_size=dense_shards,
                    comm_fusion=False, requires_grad_sync=True,
                ),
                optimizer=SimpleNamespace(fp32_main_params=True),
            ),
            num_micro_batches=2, model_config=SimpleNamespace(model_type="deepseek_v41"),
            model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(experts=experts))])),
            device=torch.device("cpu"), global_rank=1,
        )

    def test_dense_hybrid_layout_retains_local_expert_staging(self):
        """Changing only dense sharding must not disable lossless expert staging."""
        for dense_shards in (64, 128):
            with self.subTest(dense_shards=dense_shards):
                staging = build_expert_gradient_staging(self._base(dense_shards))
                self.assertIsNotNone(staging)
                self.assertEqual(len(staging.parameters), 2)
                for parameter in staging.parameters:
                    self.assertEqual(
                        getattr(parameter, "_hp_stage_local_main_grad", False), dense_shards == 64,
                    )

    def test_expert_sharding_remains_outside_the_local_staging_contract(self):
        """A sharded expert gradient cannot use a single-owner shortcut."""
        self.assertIsNone(build_expert_gradient_staging(self._base(64, expert_shards=2)))
