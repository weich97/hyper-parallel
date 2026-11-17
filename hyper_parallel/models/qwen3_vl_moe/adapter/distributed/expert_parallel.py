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
"""Expert-parallel semantics for Qwen3-VL-MoE text sparse blocks."""

from collections.abc import Callable
from typing import Any

import torch

from hyper_parallel.distributed.expert_parallel.recipes import build_ep_compute
from hyper_parallel.distributed.recipe_spec import local_compute


def _qwen3_vl_moe_router(
    module: Any,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the model-native top-k indices and normalized routing weights."""
    _, routing_weights, selected_experts = module.gate(hidden_states)
    return selected_experts, routing_weights


@local_compute
def qwen3_vl_moe_ep_compute_fn(
    *,
    module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    use_grouped_gemm: bool = False,
) -> Callable:
    """Build the complete routed-only EP path for Qwen3-VL-MoE."""
    del mesh, tp_mesh, cp_mesh
    return build_ep_compute(
        module,
        ep_mesh,
        router_fn=_qwen3_vl_moe_router,
        archetype_key="qwen3_vl_moe_topk_router",
        expected_attrs=["gate", "experts"],
        combine=lambda module, hidden_states, routed: routed,
        use_grouped_gemm=use_grouped_gemm,
    )


__all__ = ["qwen3_vl_moe_ep_compute_fn"]
