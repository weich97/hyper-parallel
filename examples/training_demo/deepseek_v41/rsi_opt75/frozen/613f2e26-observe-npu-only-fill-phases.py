"""Attest separate objective phases and bounded checkpoint input offload."""
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import runpy
import time

import torch
import torch_npu
from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.core.activation_memory.wrapper import CheckpointWrapper
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.models.deepseek_v41.adapter.chunk_loss import DeepseekV41ChunkedCausalLMLoss
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.vlm_trainer import VLMTrainer


def write_receipt(category, value):
    rank = int(os.environ["RANK"])
    target = Path(os.environ["DSV41_FULL_RUN_DIR"]) / category
    target.mkdir(exist_ok=True)
    path = target / f"rank-{rank:03d}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(value, rank=rank)) + "\n")
    temporary.replace(path)


def install():
    parent = Path("/xw-tree/full-v41/4681135a-observe-full-stage-replay-mhc.py")
    if hashlib.sha256(parent.read_bytes()).hexdigest() != "4681135a87cc6ec0b8968b88793d5b3a812a489f98d77a0d77686f38d0c610b8":
        raise ValueError("Parent execution policy changed")
    runpy.run_path(str(parent))["install"]()
    begin, end = VLMTrainer.on_train_begin, VLMTrainer.on_train_end
    original_phases = DeepseekV41ChunkedCausalLMLoss.forward_backward_phases
    original_clip = BaseTrainer.clip_grad_norm
    events = []

    @wraps(begin)
    def started(trainer):
        result = begin(trainer)
        base = trainer.base
        if not base.loss_fn.requires_sequential_backward or not base.config.fsdp_config.requires_grad_sync:
            raise ValueError("This validated schedule requires sequential objectives and synchronized gradients")
        layers = base.model.get_submodule("model").layers
        if len(layers) != 40 or not all(isinstance(layer, CheckpointWrapper) for layer in layers):
            raise ValueError("Expected forty checkpointed decoder layers")
        for index, layer in enumerate(layers):
            layer.checkpoint_kwargs["swap_inputs"] = index < 32
        if not torch.are_deterministic_algorithms_enabled():
            raise ValueError("Retain deterministic kernels")
        if not torch.utils.deterministic.fill_uninitialized_memory:
            raise ValueError("Keep CPU deterministic allocation initialization enabled")
        torch_npu._C._npu_set_fill_uninitialized_memory(False)
        if torch_npu._C._npu_get_fill_uninitialized_memory():
            raise RuntimeError("NPU allocation filling was not disabled")
        print("SEQUENTIAL_DSPARK=True CHECKPOINT_INPUT_OFFLOAD_LAYERS=32/40 CPU_MEMORY_FILL=True NPU_MEMORY_FILL=False", flush=True)
        return result

    @wraps(original_phases)
    def phases(loss, inputs):
        for item in original_phases(loss, inputs):
            phase = item["chunk_loss_phase"]
            started_at = time.monotonic()
            yield item
            events.append({"phase": phase, "host_elapsed_seconds": time.monotonic() - started_at,
                           "allocated_bytes": torch.npu.memory_allocated(),
                           "reserved_bytes": torch.npu.memory_reserved(),
                           "peak_allocated_bytes": torch.npu.max_memory_allocated()})

    @wraps(original_clip)
    def clip(base, *args, **kwargs):
        try:
            return original_clip(base, *args, **kwargs)
        except RuntimeError as error:
            if "Non-finite" in str(error):
                bad = []
                with SkipDTensorDispatch():
                    for name, parameter in base.model.named_parameters():
                        grad = getattr(parameter, "main_grad", None)
                        if grad is None:
                            grad = parameter.grad
                        if grad is None:
                            continue
                        if isinstance(grad, DTensor):
                            grad = grad.to_local()
                        if grad.is_sparse:
                            grad = grad.coalesce().values()
                        invalid = 0
                        for part in grad.detach().reshape(-1).split(1048576):
                            invalid += int((~part.isfinite()).sum().item())
                        if invalid:
                            bad.append({"parameter": name, "nonfinite_elements": invalid, "numel": grad.numel()})
                write_receipt("nonfinite-gradients", {"error": str(error), "parameters": bad, "phase_events": events})
            raise

    @wraps(end)
    def finished(trainer):
        result = end(trainer)
        steps = trainer.base.state.global_step
        if [row["phase"] for row in events] != ["backbone", "dspark"] * steps:
            raise RuntimeError("Incomplete or misordered objective phases")
        write_receipt("sequential-dspark", {"completed_steps": steps, "checkpoint_swap_prefix": 32,
                                             "memory_fill": torch.utils.deterministic.fill_uninitialized_memory,
                                             "npu_memory_fill": torch_npu._C._npu_get_fill_uninitialized_memory(), "events": events})
        return result

    VLMTrainer.on_train_begin, VLMTrainer.on_train_end = started, finished
    BaseTrainer.clip_grad_norm = clip
    DeepseekV41ChunkedCausalLMLoss.forward_backward_phases = phases
