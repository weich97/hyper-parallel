"""Preserve the recovered execution path with lightweight benchmark receipts."""

from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import runpy

import torch

from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.vlm_trainer import VLMTrainer


def install():
    """Keep the Host-cache policy while removing training probes and profiler calls."""
    root = Path("/xw-tree/full-v41")
    parent = root / "3b23ef5d-observe-full-v41-fused-indexer.py"
    digest = "3b23ef5dd91c069329d2bcf5c0ab851925923d5c2764530b05e54c5f7c2ae7c6"
    if hashlib.sha256(parent.read_bytes()).hexdigest() != digest:
        raise ValueError("The pinned lightweight Indexer observer changed")
    runpy.run_path(str(parent))["install"]()
    begin, end = VLMTrainer.on_train_begin, VLMTrainer.on_train_end
    optimizer_step = BaseTrainer.step_optimizers_and_schedulers
    iterations = 0

    @wraps(begin)
    def checked_begin(trainer):
        result = begin(trainer)
        drafter = trainer.base.model.get_submodule("dspark")
        if len(drafter.stages) != 3 or not drafter.stage_checkpoint or not drafter.stage_checkpoint_offload or not all(
                not stage.activation_offload for stage in drafter.stages):
            raise ValueError("The full benchmark requires three complete recomputed DSpark stages")
        return result

    @wraps(optimizer_step)
    def step(trainer):
        nonlocal iterations
        # The recovery recipe releases inactive pinned cache at this phase boundary.
        torch.npu.synchronize()
        torch.npu.host_empty_cache()
        result = optimizer_step(trainer)
        iterations += 1
        return result

    @wraps(end)
    def checked_end(trainer):
        result = end(trainer)
        if iterations != trainer.base.state.global_step:
            raise ValueError("The cache policy must run at every optimizer update")
        rank = int(os.environ["RANK"])
        out = Path(os.environ["DSV41_FULL_RUN_DIR"]) / "dspark-stage-replay"
        out.mkdir(exist_ok=True)
        path = out / f"rank-{rank:03d}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"rank": rank, "completed_steps": iterations,
                                         "recomputed_drafter_stages": 3, "checkpoint_input_offload": True, "host_cache_policy": "flush-before-update"})
                             + "\n", encoding="utf-8")
        temporary.replace(path)
        return result

    VLMTrainer.on_train_begin, VLMTrainer.on_train_end = checked_begin, checked_end
    BaseTrainer.step_optimizers_and_schedulers = step
