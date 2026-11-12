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
from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2
from hyper_parallel.trainer.runtime.expert_grad_staging import ExpertGradientStaging, build_expert_gradient_staging
from tests.common.mark_utils import arg_mark


class TestExpertGradientStagingSelection(unittest.TestCase):
    """Select only an owned expert shard that can safely expose its first partial."""

    @staticmethod
    def _base(dense_shards: int, expert_shards: int = 1) -> SimpleNamespace:
        experts = Mock(spec=DeepseekV41MegaMoeExperts)
        experts.gate_up_proj = torch.nn.Parameter(torch.zeros(2, 3))
        experts.down_proj = torch.nn.Parameter(torch.zeros(3, 2))
        model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(experts=experts))]))
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
            model=model,
            device=torch.device("cpu"), global_rank=1,
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dense_hybrid_layout_retains_local_expert_staging(self):
        """Feature: Expert gradient ownership.

        Description: Change only the dense shard domain.
        Expectation: Supported layouts retain the lossless local-expert staging path.
        """
        for dense_shards in (32, 64, 128):
            with self.subTest(dense_shards=dense_shards):
                staging = build_expert_gradient_staging(self._base(dense_shards))
                self.assertIsNotNone(staging)
                self.assertEqual(len(staging.parameters), 2)
                for parameter in staging.parameters:
                    self.assertEqual(
                        getattr(parameter, "_hp_stage_local_main_grad", False), dense_shards != 128,
                    )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_expert_sharding_remains_outside_the_local_staging_contract(self):
        """Feature: Shared expert gradient ownership.

        Description: Request staging with multiple owners per expert.
        Expectation: The unsupported layout does not install a local-owner shortcut.
        """
        for dense_shards in (32, 64, 128):
            with self.subTest(dense_shards=dense_shards):
                self.assertIsNone(build_expert_gradient_staging(self._base(dense_shards, expert_shards=2)))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_multiple_partials_require_a_local_hybrid_reduction_boundary(self) -> None:
        """Feature: Multi-microbatch staging admission.

        Description: Vary dense shard count and accumulation count.
        Expectation: Only layouts exposing every owned expert partial enable extended staging.
        """
        for shards in (32, 64, 128):
            for count in (1, 2, 4, 8):
                with self.subTest(shards=shards, count=count):
                    base = self._base(shards)
                    base.num_micro_batches = count
                    enabled = count == 2 or (count > 2 and shards in (32, 64))
                    self.assertEqual(build_expert_gradient_staging(base) is not None, enabled)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_staged_accumulation_preserves_fp32_sum_and_update(self) -> None:
        """Feature: Lossless cumulative gradient transport.

        Description: Stage four or eight partials through the real HSDP consumer.
        Expectation: The sum and parameter update equal ordinary FP32 accumulation exactly.
        """
        for count in (2, 4, 8):
            with self.subTest(count=count):
                parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
                owner = SimpleNamespace(
                    apply_grad_on_fp32_main_grad=True, sharded_param=parameter,
                    sharded_size=parameter.shape, offload_to_cpu=False,
                    to_sharded_dtensor=lambda value: value, unsharded_accumulated_grad_data=None,
                    unsharded_param=SimpleNamespace(grad=None),
                )
                staging = ExpertGradientStaging((parameter,))
                expected = torch.zeros_like(parameter)
                for micro_step in range(count):
                    partial = (torch.tensor([0.5, -2.0]) if micro_step == 0
                               else torch.tensor([0.00123, 3.14159]) * micro_step)
                    expected.add_(partial)
                    HSDPParamV2.apply_reduced_grad(owner, partial.clone())
                    if micro_step:
                        staging.finish()
                    if micro_step + 1 < count:
                        staging.pack(first_partial=micro_step == 0)
                        staged, _ = parameter._hp_staged_main_grad
                        expected_dtype = torch.bfloat16 if micro_step == 0 else torch.float32
                        self.assertEqual(staged.dtype, expected_dtype)
                        self.assertIsNone(parameter.main_grad)
                torch.testing.assert_close(parameter.main_grad, expected, rtol=0, atol=0)
                self.assertFalse(staging.active)
                reference_update = torch.tensor([1.0, -1.0]) - 0.001 * expected
                with torch.no_grad():
                    actual_update = parameter - 0.001 * parameter.main_grad
                torch.testing.assert_close(actual_update, reference_update, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unrepresentable_first_partial_is_not_silently_rounded(self) -> None:
        """Feature: Exact first-partial compaction.

        Description: Supply FP32 values outside the BF16 representable set.
        Expectation: Compaction fails before the original gradient is released.
        """
        parameter = torch.nn.Parameter(torch.zeros(2))
        gradient = torch.tensor([0.00123, 3.14159])
        parameter.main_grad = gradient
        staging = ExpertGradientStaging((parameter,))
        with self.assertRaisesRegex(RuntimeError, "outside the BF16 format"):
            staging.pack()
        self.assertIs(parameter.main_grad, gradient)
        self.assertFalse(staging.active)
