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
"""Fingerprint fresh full-model state before a paired VLM training run."""

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.models.external_state import get_model_external_state
from hyper_parallel.tools.model_integration.optimizer_layout import optimizer_leaves
from hyper_parallel.trainer.vlm_trainer import VLMTrainer


def append_tensor(digest: Any, name: str, tensor: torch.Tensor) -> None:
    """Hash logical metadata and every local byte in bounded host chunks."""
    local = tensor.to_local() if callable(getattr(tensor, "to_local", None)) else tensor
    if local.device.type == "meta":
        raise ValueError(f"Cannot fingerprint unmaterialized state: {name}")
    metadata = [name, str(tensor.dtype), list(tensor.shape), list(local.shape)]
    digest.update(json.dumps(metadata, separators=(",", ":")).encode("utf-8") + b"\0")
    elements_per_chunk = (64 * 2**20) // local.element_size()
    for chunk in local.detach().reshape(-1).split(elements_per_chunk):
        raw = chunk.cpu().contiguous().view(torch.uint8)
        digest.update(memoryview(raw.numpy()))


def install() -> None:
    """Attach an identity observer without changing the training arithmetic."""
    original = VLMTrainer.on_train_begin

    def record(trainer: VLMTrainer) -> None:
        original(trainer)
        base = trainer.base
        if base.state.global_step != 0:
            raise RuntimeError("Fresh paired runs must begin before the first update")
        leaves = optimizer_leaves(base.optimizer)
        for optimizer in leaves:
            state = optimizer.state
            if not isinstance(state, Mapping) or state:
                raise RuntimeError("Fresh optimizer leaves must have empty materialized state")
        parameters = dict(base.model.named_parameters())
        external = get_model_external_state(base.model)
        if external is None:
            raise RuntimeError("Full V4.1 fingerprinting requires its Host Engram state")
        external.check_identities()
        for name, parameter in external.parameters.items():
            if parameters.get(name) is not parameter:
                raise RuntimeError(f"External state is absent from the model fingerprint: {name}")
        if len(external.parameters) != 2:
            raise RuntimeError("The released full model must retain both Host Engram tables")
        digest = hashlib.sha256()
        main_count = 0
        for name, parameter in sorted(parameters.items()):
            append_tensor(digest, name, parameter)
            main = getattr(parameter, "main_param", None)
            if main is not None and main is not parameter:
                append_tensor(digest, name + ".main_param", main)
                main_count += 1
        buffers = dict(base.model.named_buffers())
        for name, buffer in sorted(buffers.items()):
            append_tensor(digest, name, buffer)
        rng = hashlib.sha256()
        append_tensor(rng, "cpu_rng", torch.get_rng_state())
        append_tensor(rng, "npu_rng", torch.npu.get_rng_state())
        rank = dist.get_rank()
        receipt = {
            "rank": rank, "step": 0, "named_parameters": len(parameters), "main_parameters": main_count,
            "named_buffers": len(buffers), "host_engram_parameters": sorted(external.parameters),
            "model_state_sha256": digest.hexdigest(), "rng_sha256": rng.hexdigest(),
            "optimizer_leaves": [type(optimizer).__name__ for optimizer in leaves],
            "optimizer_state_entries": 0,
        }
        directory = Path(os.environ["DSV41_FULL_RUN_DIR"]) / "initial-state"
        directory.mkdir(exist_ok=True)
        output = directory / f"rank-{rank:03}.json"
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
        print(f"FULL_V41_INITIAL_STATE_READY rank={rank}", flush=True)

    VLMTrainer.on_train_begin = record
