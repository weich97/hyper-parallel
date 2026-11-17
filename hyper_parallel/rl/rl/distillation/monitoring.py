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
"""Pre-update probability acceptance and detached OPD diagnostics."""

from typing import Any

import torch
import torch.distributed as dist

from rl.algorithm.distillation import compute_opd_terms
from rl.distillation.config import DistillationConfig


def validate_probability_gate(
    experience: Any, config: DistillationConfig, *, group: Any, size: int,
) -> dict[str, float]:
    """Reject every DP rank before optimization when sampled-policy alignment fails.

    Only active loss tokens count. Empty padding ranks participate in every
    collective but do not affect maxima, means or probability-ratio tails.
    """
    valid = experience.loss_action_mask.bool()
    current = experience.old_log_probs.detach().float()[valid]
    sampled = experience.rollout_log_probs.detach().float()[valid]
    delta = current - sampled
    finite = torch.isfinite(current).all() & torch.isfinite(sampled).all()
    finite = finite & (current <= 0).all() & (sampled <= 0).all()
    ratio = delta.clamp(-20, 20).exp()
    outside = (ratio < 1 - config.clip_ratio) | (ratio > 1 + config.clip_ratio)
    totals = torch.stack((delta.abs().sum(), delta.new_tensor(delta.numel()), outside.float().sum()))
    maximum = delta.abs().amax() if delta.numel() else delta.new_zeros(())
    accepted = finite.to(torch.int32)
    if size > 1:
        dist.all_reduce(accepted, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(totals, group=group)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    absolute, tokens, tail = totals.tolist()
    mean = absolute / max(tokens, 1)
    fraction = tail / max(tokens, 1)
    peak = float(maximum.item())
    if (not bool(accepted.item()) or tokens == 0 or mean > config.max_mean_logprob_diff
            or peak > config.max_logprob_diff or fraction > config.max_ratio_outside_fraction):
        raise RuntimeError(
            f"OPD pre-update probability gate rejected batch: tokens={tokens}, finite={bool(accepted.item())}, "
            f"mean={mean} (limit {config.max_mean_logprob_diff}), max={peak} (limit {config.max_logprob_diff}), "
            f"ratio_outside_fraction={fraction} (limit {config.max_ratio_outside_fraction}); "
            "no optimizer update allowed"
        )
    return {"distillation/pre_update_logprob_diff_mean": mean,
            "distillation/pre_update_logprob_diff_max": peak,
            "distillation/pre_update_ratio_outside_fraction": fraction}


@torch.no_grad()
def opd_statistics(current: torch.Tensor, old: torch.Tensor, teacher: torch.Tensor,
                   mask: torch.Tensor, config: DistillationConfig) -> torch.Tensor:
    """Return additive, mask-aware diagnostics using the objective's exact clipping rules."""
    valid = mask.bool()
    terms = compute_opd_terms(current.detach().float(), old.detach().float(), teacher.detach().float(), mask, config)
    delta, ratio, raw_signal = terms.log_ratio[valid], terms.ratio[valid], terms.signal[valid]
    return torch.stack((terms.loss[valid].sum(), delta.new_tensor(delta.numel()), ratio.sum(), delta.abs().sum(),
                        ((ratio < 1 - config.clip_ratio) | (ratio > 1 + config.clip_ratio)).float().sum(),
                        terms.clipped[valid].float().sum(), terms.dual_clipped[valid].float().sum(),
                        (delta.abs() > 20).float().sum(), raw_signal.sum(), raw_signal.abs().sum()))


class DistillationMetricAccumulator:
    """Accumulate token-weighted OPD and teacher statistics across micro-batches and DP."""

    def __init__(self, zero: torch.Tensor, config: DistillationConfig) -> None:
        """Keep a fixed teacher ordering so all ranks perform the same collective."""
        self.config = config
        self.keys = tuple(key for key, _ in config.teachers)
        self.totals = zero.new_zeros((1 + len(self.keys), 10), dtype=torch.float32)
        self.minimum = zero.new_full((1 + len(self.keys),), float("inf"), dtype=torch.float32)
        self.maximum = -self.minimum.clone()

    def add(self, current: torch.Tensor, old: torch.Tensor, teacher: torch.Tensor,
            mask: torch.Tensor, teacher_keys: tuple[str, ...]) -> None:
        """Accumulate detached statistics without transferring per-token data to the host."""
        self.totals[0] += opd_statistics(current, old, teacher, mask, self.config)
        self._add_extrema(0, current, old, mask)
        for index, key in enumerate(self.keys, 1):
            rows = torch.tensor([candidate == key for candidate in teacher_keys], device=mask.device)
            self.totals[index] += opd_statistics(current, old, teacher, mask.bool() & rows[:, None], self.config)
            self._add_extrema(index, current, old, mask.bool() & rows[:, None])

    @torch.no_grad()
    def _add_extrema(self, index: int, current: torch.Tensor, old: torch.Tensor, mask: torch.Tensor) -> None:
        """Retain unclamped log-ratio extrema, including tails hidden by numerical clipping."""
        delta = (current.detach().float() - old.detach().float())[mask.bool()]
        if delta.numel():
            self.minimum[index] = torch.minimum(self.minimum[index], delta.amin())
            self.maximum[index] = torch.maximum(self.maximum[index], delta.amax())

    def finalize(self, *, group: Any, size: int) -> dict[str, float]:
        """Reduce additive statistics once, retaining exact teacher token contribution."""
        totals = self.totals.clone()
        minimum, maximum = self.minimum.clone(), self.maximum.clone()
        if size > 1:
            dist.all_reduce(totals, group=group)
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN, group=group)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        metrics = {}
        names = ("loss", "supervised_tokens", "ratio_mean", "log_ratio_abs_mean", "ratio_outside_fraction",
                 "clip_fraction", "dual_clip_fraction", "log_ratio_saturation_fraction", "sampled_k1_mean",
                 "sampled_k1_abs_mean")
        global_tokens = float(totals[0, 1].item())
        for row, key in enumerate((None, *self.keys)):
            values = totals[row].tolist()
            tokens = values[1]
            prefix = "distillation" if key is None else f"distillation/teacher/{key}"
            metrics[f"{prefix}/log_ratio_min"] = float(minimum[row].item()) if tokens else 0.0
            metrics[f"{prefix}/log_ratio_max"] = float(maximum[row].item()) if tokens else 0.0
            for index, name in enumerate(names):
                metrics[f"{prefix}/{name}"] = values[index] if index == 1 else values[index] / max(tokens, 1)
            if key is not None:
                metrics[f"{prefix}/token_fraction"] = tokens / max(global_tokens, 1)
                metrics[f"{prefix}/weighted_loss"] = self.config.coefficient * values[0] / max(global_tokens, 1)
        return metrics
