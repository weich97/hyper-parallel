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
"""DeepSeek-V4.1 adapter for model-integrated Chunk Loss training."""

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping
from typing import Any

# This model-family adapter follows the PyTorch-native V4.1 facade.
# pylint: disable=forbidden-backend-import
import torch
from torch import nn
from torch.nn import functional

from hyper_parallel.components.losses.chunked_cross_entropy import (
    ChunkedCausalLMLoss,
    ChunkedCausalLMOutput,
    chunked_cross_entropy,
)
from hyper_parallel.distributed._builder.forward_rewriter import (  # pylint: disable=protected-access
    _ForwardRewriteRequest,
    _commit_forward_rewrite,
)

logger = logging.getLogger(__name__)


class DeepseekV41ChunkedCausalLMLoss(ChunkedCausalLMLoss):
    """Use bounded vocabulary projection for V4.1 Omni training with DSpark.

    Omni batches contain unshifted labels. This adapter performs their causal
    alignment before delegating to the shared Chunk Loss protocol. Complete
    sequences are required; shifting a CP-local shard would lose its boundary
    target. Explicit ``shift_labels`` retain precedence for text batches.
    """

    def bind_model(self, model: nn.Module, distributed_setup: Any = None) -> None:
        """Bind a supported complete-sequence V4.1 training model.

        Args:
            model: Constructed V4.1 model with its final parameter structure.
            distributed_setup: Trainer setup declaring the active parallel layout.
        """
        if self._parallel_size(distributed_setup, "cp_size") != 1:
            raise NotImplementedError("V4.1 Omni Chunk Loss currently requires cp_size=1")
        super().bind_model(model, distributed_setup)
        logger.info("V4.1 chunked backbone CE enabled; chunk_size=%d; DSpark=%s",
                    self.chunk_size, getattr(model, "dspark", None) is not None)

    def prepare_model_inputs(
            self,
            model_inputs: Mapping[str, Any],
            loss_inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Preserve ignored visual/padding targets while shifting Omni labels.

        Args:
            model_inputs: Token, image, and packed-sequence inputs for the backbone.
            loss_inputs: Unshifted Omni labels or explicit pre-shifted text targets.

        Returns:
            Inputs using the common Chunk Loss target and mask protocol.
        """
        aligned = dict(loss_inputs)
        if aligned.get("shift_labels") is None:
            labels = aligned.get("labels", model_inputs.get("labels"))
            if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
                raise ValueError("V4.1 Omni Chunk Loss requires labels with shape [batch, sequence]")
            aligned["shift_labels"] = functional.pad(labels[..., 1:], (0, 1), value=self.ignore_index)
            mask = aligned.get("loss_mask")
            if mask is not None:
                if not isinstance(mask, torch.Tensor) or mask.shape != labels.shape:
                    raise ValueError("V4.1 Omni loss_mask must match unshifted labels")
                aligned["loss_mask"] = functional.pad(mask[..., 1:], (0, 1), value=0)
        return super().prepare_model_inputs(model_inputs, aligned)


def _chunk_loss_forward(
    model: nn.Module,
    *,
    chunk_loss_targets: torch.Tensor,
    chunk_loss_mask: torch.Tensor | None,
    chunk_loss_chunk_size: int,
    chunk_loss_ignore_index: int,
    **model_inputs: Any,
) -> ChunkedCausalLMOutput:
    """Run the V4.1 backbone and calculate CE without full-sequence logits."""
    if model_inputs.get("past_key_values") is not None or model_inputs.get("use_cache"):
        raise ValueError("DeepSeek-V4.1 Chunk Loss does not support KV cache")
    if model_inputs.pop("return_dict", True) is False:
        raise ValueError("DeepSeek-V4.1 Chunk Loss requires return_dict=True")
    logits_to_keep = model_inputs.pop("logits_to_keep", 0)
    if isinstance(logits_to_keep, torch.Tensor) or logits_to_keep != 0:
        raise ValueError("DeepSeek-V4.1 Chunk Loss requires logits_to_keep=0")
    if model_inputs.get("output_router_logits", getattr(model.config, "output_router_logits", False)):
        raise NotImplementedError("DeepSeek-V4.1 Chunk Loss does not support HF output_router_logits")

    model_inputs["use_cache"] = False
    outputs = model.model(**model_inputs)
    hidden_states = outputs.last_hidden_state
    if chunk_loss_targets.dim() != 2 or hidden_states.shape[:2] != chunk_loss_targets.shape:
        raise ValueError(
            "DeepSeek-V4.1 Chunk Loss targets must match final hidden batch/sequence "
            f"dimensions, got hidden={tuple(hidden_states.shape)} and "
            f"targets={tuple(chunk_loss_targets.shape)}"
        )
    if chunk_loss_mask is not None:
        if chunk_loss_mask.shape != chunk_loss_targets.shape:
            raise ValueError("DeepSeek-V4.1 Chunk Loss mask must match targets")
        chunk_loss_targets = chunk_loss_targets.masked_fill(
            ~chunk_loss_mask.to(torch.bool), chunk_loss_ignore_index,
        )

    loss_sum = chunked_cross_entropy(
        hidden_states,
        chunk_loss_targets,
        model.lm_head.weight,
        chunk_size=chunk_loss_chunk_size,
        ignore_index=chunk_loss_ignore_index,
    )
    result = ChunkedCausalLMOutput(
        loss_sum=loss_sum,
        valid_token_count=chunk_loss_targets.ne(chunk_loss_ignore_index).sum(),
        logits=None,
        past_key_values=getattr(outputs, "past_key_values", None),
        hidden_states=getattr(outputs, "hidden_states", None),
        attentions=getattr(outputs, "attentions", None),
    )
    if getattr(model, "dspark", None) is not None:
        input_ids = model_inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("V4.1 DSpark Chunk Loss requires input_ids")
        dspark_loss, metrics = model.compute_dspark_loss(input_ids, chunk_loss_targets)
        result.aux_loss = dspark_loss
        result.aux_loss_coef = 1.0
        result.extra_metrics = metrics
    return result


def bind_chunk_loss(model: nn.Module) -> None:
    """Atomically install the V4.1 Chunk Loss training forward.

    Args:
        model: V4.1 facade with a backbone and a bias-free output projection.
    """
    if getattr(model, "_hp_deepseek_v41_chunk_loss_bound", False):
        return
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    architectures = getattr(config, "architectures", None) or ()
    if model_type != "deepseek_v4" or "DeepseekV41ForCausalLM" not in architectures:
        raise TypeError(
            "DeepSeek-V4.1 Chunk Loss adapter requires a DeepseekV4Config "
            "with architecture='DeepseekV41ForCausalLM'"
        )
    if not isinstance(getattr(model, "model", None), nn.Module):
        raise TypeError("DeepSeek-V4.1 Chunk Loss adapter requires a backbone at model.model")
    lm_head = getattr(model, "lm_head", None)
    if not isinstance(lm_head, nn.Linear) or lm_head.bias is not None:
        raise TypeError("DeepSeek-V4.1 Chunk Loss adapter requires a bias-free nn.Linear lm_head")

    original_forward = model.forward

    @functools.wraps(original_forward)
    def _chunk_loss_aware_forward(*args: Any, **kwargs: Any) -> Any:
        if "chunk_loss_targets" not in kwargs:
            return original_forward(*args, **kwargs)
        if args:
            raise TypeError("DeepSeek-V4.1 Chunk Loss requires keyword model inputs")
        return _chunk_loss_forward(model, **kwargs)

    _commit_forward_rewrite(
        _ForwardRewriteRequest(
            model,
            _chunk_loss_aware_forward,
            companion_attrs={"_hp_deepseek_v41_chunk_loss_bound": True},
        )
    )


__all__ = ["DeepseekV41ChunkedCausalLMLoss", "bind_chunk_loss"]
