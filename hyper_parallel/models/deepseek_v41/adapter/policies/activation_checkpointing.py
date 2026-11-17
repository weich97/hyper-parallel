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
"""Declare the DeepSeek-V4.1 recompute regions used by normal training."""

from functools import partial
import logging
from typing import Any

from hyper_parallel.core.activation_memory.wrapper import CheckpointWrapper
from hyper_parallel.distributed.encoder_parallel.attention import encoder_attention_checkpoint_contexts
from hyper_parallel.models.adapter_spec import RecomputePolicy
from hyper_parallel.models.deepseek_v41.adapter.policies.shared_state import (
    attention_role,
    configure_shared_state,
)
from hyper_parallel.models.deepseek_v41.vision import DeepseekV41VisionTower

#: The CSA2 attention roles, producers first.
_ATTENTION_ROLES = ("full", "reindex", "reuse")

#: The attention roles that recompute with their layer, per tier.
_RECOMPUTED_ROLES = {
    "none": (),
    "reuse": ("reuse",),
    "reuse_reindex": ("reuse", "reindex"),
    "all": _ATTENTION_ROLES,
}

#: The tier the model ships with: the layers that only consume cross-layer state
#: recompute, their producers stay outside the region and keep their
#: activations. Higher tiers recompute more attention for less memory, lower
#: tiers recompute less attention for less step time.
ATTENTION_RECOMPUTE_TIER = "all"
_CHECKPOINT_SWAP_PREFIX = 32
logger = logging.getLogger(__name__)


def _attention_stays_resident(module_fqn: str, module: Any, *, resident: tuple[str, ...]) -> bool:
    """Keep the attention of a role that does not recompute out of the region."""
    del module_fqn
    return attention_role(module) in resident


def _configure_recompute(model: Any, *, roles: tuple[str, ...]) -> None:
    """Configure decoder shared state and complete visual-block replay."""
    configure_shared_state(model, roles)
    layers = getattr(getattr(model, "model", None), "layers", ())
    if len(layers) == 40 and isinstance(layers[0], CheckpointWrapper):
        if layers[0].checkpoint_kwargs.get("swap_inputs"):
            for layer in layers[_CHECKPOINT_SWAP_PREFIX:]:
                if isinstance(layer, CheckpointWrapper):
                    layer.checkpoint_kwargs["swap_inputs"] = False
            logger.info("V4.1 checkpoint input offload enabled for decoder layers 0 through %d",
                        _CHECKPOINT_SWAP_PREFIX - 1)
    for module in model.modules():
        if isinstance(module, DeepseekV41VisionTower):
            for block in module.blocks:
                if isinstance(block, CheckpointWrapper):
                    block.checkpoint_kwargs.update(
                        context_fn=encoder_attention_checkpoint_contexts,
                        # All blocks reuse one rotary table from the vision tower.
                        swap_inputs=False,
                        # All ranks must replay every Ulysses/FSDP collective.
                        early_stop=False,
                    )


def build_recompute_policy() -> RecomputePolicy:
    """Return the V4.1 recompute regions used by normal training.

    Every decoder layer and every visual block is one region. The attention
    roles listed for :data:`ATTENTION_RECOMPUTE_TIER` recompute with their layer; the remaining
    roles stay outside the region, which is what the tiers differ in. The
    model-owned hook then pins the discrete selection of the recomputed index
    sources, so recomputing one neither chooses a different key set nor
    publishes its state a second time; published cross-layer entries stay alive
    until the graph drops the state in backward.
    """
    recomputed = _RECOMPUTED_ROLES[ATTENTION_RECOMPUTE_TIER]
    resident = tuple(role for role in _ATTENTION_ROLES if role not in recomputed)
    return RecomputePolicy(
        region_patterns=("model.layers.*", "model.vision.blocks.*"),
        exclude_predicate=(
            partial(_attention_stays_resident, resident=resident) if resident else None
        ),
        on_applied=partial(_configure_recompute, roles=recomputed),
    )


__all__ = ["build_recompute_policy"]
