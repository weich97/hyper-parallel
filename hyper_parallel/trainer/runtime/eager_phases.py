# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sequential eager objectives with one optimizer-step normalization."""

from typing import Any

from hyper_parallel.components.functional.aux_loss import aux_loss_scale_context
from hyper_parallel.trainer.runtime.memory import empty_cache


def eager_forward_backward(trainer: Any, model_inputs: dict, labels: Any) -> tuple:
    """Finish each independent objective backward before constructing the next.

    The loss adapter owns the phase inputs and must preserve the original
    objective. Every phase uses the same microbatch token counts; the caller
    counts tokens and updates parameters once, after every phase completes.

    Args:
        trainer: BaseTrainer exposing model, loss normalization and contexts.
        model_inputs: Prepared inputs for the current microbatch.
        labels: Loss-only targets shared by all objective phases.
    """
    provider = getattr(trainer.loss_fn, "forward_backward_phases", None)
    phases = provider(model_inputs) if callable(provider) else (model_inputs,)
    total_loss = None
    metrics = {}
    for index, phase_inputs in enumerate(phases):
        with aux_loss_scale_context():
            context = trainer.model_fwd_context
            with context() if callable(context) else context:
                outputs = trainer.model(**phase_inputs, use_cache=False)
            loss, phase_metrics = trainer.postforward(
                outputs, labels, current_token_counts=trainer.current_token_counts,
            )
            del outputs
            if index == 0:
                prefetch = getattr(trainer.train_dataloader, "prefetch_plan", None)
                if callable(prefetch):
                    prefetch()
            if trainer.config.training.empty_cache_before_backward:
                empty_cache()
            with trainer.model_bwd_context:
                loss.backward()
        total_loss = loss if total_loss is None else total_loss.detach() + loss.detach()
        for name, value in phase_metrics.items():
            if name in trainer.loss_names and name in metrics:
                metrics[name] = metrics[name].detach() + value.detach()
            else:
                metrics[name] = value
    if total_loss is None:
        raise ValueError("A training microbatch must execute at least one objective phase")
    return total_loss, metrics
