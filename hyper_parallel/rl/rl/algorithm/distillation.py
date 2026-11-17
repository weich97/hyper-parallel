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
"""verl-style sampled-token PG distillation, composed with task RL."""

from typing import Any, NamedTuple, Optional

import torch

from rl.algorithm.advantage import TargetOutput
from rl.algorithm.loss import AlgorithmRequirements, DataRequirements, LossOutput, RoleRequirements
from rl.distillation.config import DistillationConfig


class PureDistillationAlgorithm:
    """Use the existing Actor lifecycle without task advantages or a Critic."""

    name = "opd"
    requirements = AlgorithmRequirements(RoleRequirements(), DataRequirements())

    @staticmethod
    def compute_advantages(rewards: Any, group_ids: Optional[Any] = None) -> Any:
        """Return zero task advantages without grouping student responses."""
        del group_ids
        return torch.zeros_like(rewards)

    @staticmethod
    def compute_critic_loss(current_values: Any, old_values: Any, returns: Any, action_mask: Any) -> Any:
        """Reject a Critic in pure distillation instead of training unused values."""
        raise RuntimeError("Pure OPD has no Critic objective")

    @staticmethod
    def build_targets(rewards: Any, action_mask: Any, **kwargs: Any) -> TargetOutput:
        """Supply zero task targets; only the separate OPD objective is optimized."""
        del rewards, kwargs
        return TargetOutput(advantages=torch.zeros_like(action_mask, dtype=torch.float32), returns=None)

    @staticmethod
    def compute_actor_loss(
        current_log_probs: Any,
        old_log_probs: Any,
        reference_log_probs: Optional[Any],
        advantages: Any,
        action_mask: Any,
    ) -> LossOutput:
        """Return zero task numerators with normal policy diagnostics."""
        del reference_log_probs, advantages
        mask = action_mask.bool()
        zero = current_log_probs.masked_fill(~mask, 0.0).sum() * 0.0
        return LossOutput(
            zero,
            zero,
            zero,
            mask.sum(),
            (old_log_probs - current_log_probs).masked_fill(~mask, 0.0).sum(),
            zero.detach(),
        )


def compose_training_algorithm(task: Any, distillation: DistillationConfig) -> Any:
    """Keep the selected task recipe or replace its requirements for pure OPD."""
    return PureDistillationAlgorithm() if distillation.enabled and not distillation.use_task_rewards else task


class OPDTerms(NamedTuple):
    """Per-token objective values shared by backward and detached monitoring."""

    loss: torch.Tensor
    log_ratio: torch.Tensor
    ratio: torch.Tensor
    signal: torch.Tensor
    clipped: torch.Tensor
    dual_clipped: torch.Tensor


def compute_opd_terms(
    current: torch.Tensor, old: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor, config: DistillationConfig
) -> OPDTerms:
    """Compute the shared sampled-k1 objective and its clipping decisions.

    Teacher scores and old-policy anchors are fixed. Recompute the k1 signal
    at every forward, then stop its gradient as in verl's PG OPD path.
    """
    valid = mask.bool()
    current = current.masked_fill(~valid, 0.0)
    old = old.detach().masked_fill(~valid, 0.0)
    teacher = teacher.detach().masked_fill(~valid, 0.0)
    signal = current - teacher
    advantage = -signal.detach()
    if config.loss_max_clamp is not None:
        advantage = advantage.clamp(-config.loss_max_clamp, config.loss_max_clamp)
    log_ratio = current - old
    ratio = log_ratio.clamp(-20.0, 20.0).exp()
    unclipped = -ratio * advantage
    clipped = -ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantage
    loss = torch.maximum(unclipped, clipped)
    dual_clipped = (advantage < 0) & (loss > -config.dual_clip * advantage)
    loss = torch.where(advantage < 0, torch.minimum(loss, -config.dual_clip * advantage), loss)
    return OPDTerms(loss.masked_fill(~valid, 0.0), log_ratio, ratio, signal, clipped > unclipped, dual_clipped)


def compute_opd_loss(
    current: torch.Tensor, old: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor, config: DistillationConfig
) -> torch.Tensor:
    """Sum the shared clipped objective over valid action tokens."""
    return compute_opd_terms(current, old, teacher, mask, config).loss.sum()
