# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Verify partial-gradient ownership across the actual text training loop."""

import importlib
import itertools
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from tests.common.mark_utils import arg_mark


def _load_text_trainer() -> type:
    """Isolate the unrelated optional stateful dataloader dependency."""
    package = ModuleType("torchdata")
    loader = ModuleType("torchdata.stateful_dataloader")
    loader.StatefulDataLoader = torch.utils.data.DataLoader
    with patch.dict(sys.modules, {"torchdata": package, "torchdata.stateful_dataloader": loader}):
        module = importlib.import_module("hyper_parallel.trainer.text_trainer")
    return module.TextTrainer


TextTrainer = _load_text_trainer()


class TestStreamingGradientStep(unittest.TestCase):
    """Exercise lifecycle orchestration without a distributed accelerator."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_begin_epoch_preserves_only_the_resumed_collective_cursor(self) -> None:
        """Feature: Loader epoch lifecycle.

        Description: Begin restored and subsequent epochs with both loader types.
        Expectation: Only the initial collective epoch keeps its restored cursor.
        """
        for balanced in (False, True):
            for epoch in (3, 4):
                with self.subTest(balanced=balanced, epoch=epoch):
                    trainer = TextTrainer.__new__(TextTrainer)
                    trainer.base = SimpleNamespace(state=SimpleNamespace(epoch=3))
                    trainer.on_epoch_begin = Mock()
                    loader = Mock()
                    trainer._begin_data_epoch(loader, balanced, epoch, 3)
                    self.assertEqual(loader.set_epoch.call_count, int(not balanced or epoch != 3))
                    self.assertEqual(trainer.base.state.epoch, epoch)
                    trainer.on_epoch_begin.assert_called_once_with()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_each_nonfinal_partial_is_consumed_before_restaging(self) -> None:
        """Feature: Multi-microbatch gradient staging lifecycle.

        Description: Execute two, four and eight backwards through TextTrainer.train_step.
        Expectation: Every staged partial is consumed before restaging or optimizer update.
        """
        for count in (2, 4, 8):
            with self.subTest(count=count):
                events = []
                micro_steps = itertools.count()
                trainer = TextTrainer.__new__(TextTrainer)
                trainer.base = Mock(num_micro_batches=count, global_rank=1,
                                    state=SimpleNamespace(global_step=2))
                trainer.base.prepare_optimizer_step.return_value = (float(count), 1.0)
                trainer.base.step_optimizers_and_schedulers.side_effect = lambda: events.append(("optimizer",))
                staging = Mock()
                staging.pack.side_effect = lambda **kwargs: events.append(("pack", kwargs["first_partial"]))
                staging.finish.side_effect = lambda: events.append(("finish",))
                trainer._expert_gradient_staging = staging
                trainer.on_step_begin = Mock()
                trainer.on_step_end = Mock()

                def _backward(_iterator):
                    """Record one complete backward without mocking the training loop."""
                    events.append(("backward", next(micro_steps)))
                    return torch.tensor(1.0), {"foundation_loss": torch.tensor(1.0)}

                trainer.forward_backward_step = _backward
                with patch.dict(trainer.train_step.__globals__, synchronize=Mock(), memory_profiler=Mock()):
                    result = trainer.train_step(iter(()))
                expected = []
                for micro_step in range(count):
                    expected.append(("backward", micro_step))
                    if micro_step:
                        expected.append(("finish",))
                    if micro_step + 1 < count:
                        expected.append(("pack", micro_step == 0))
                expected.append(("optimizer",))
                self.assertEqual(events, expected)
                self.assertEqual(result, {"loss": float(count), "grad_norm": 1.0})
                self.assertEqual(trainer.base.state.global_step, 3)
