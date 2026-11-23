"""Run the native HP VLM Trainer with a full-model smoke receipt."""

from datetime import timedelta
import json
import math
import os
from pathlib import Path
import runpy
import sys
import traceback

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from hyper_parallel.trainer.vlm_trainer import VLMTrainer
from scripts.train_vl import main as train_vlm


def install_smoke_receipt(rank, output, expected_steps, mode):
    """Record real multimodal branch execution and completed optimizer updates."""
    watched = ("model.vision", "model.aligner", "dspark", "model.layers.1.engram", "model.layers.14.engram")
    counts = dict.fromkeys(watched, 0)
    draft_loss = None
    steps = []
    original_begin = VLMTrainer.on_train_begin
    original_step = VLMTrainer.train_step

    def observe(name):
        def record(_module, _args, value):
            nonlocal draft_loss
            counts[name] += 1
            if name == "dspark":
                draft_loss = value[0].detach()
        return record

    def begin(trainer):
        original_begin(trainer)
        modules = dict(trainer.base.model.named_modules())
        for name in watched:
            if name not in modules:
                raise ValueError(f"Missing full-model training branch: {name}")
            modules[name].register_forward_hook(observe(name))
        print(f"FULL_V41_MODEL_READY rank={rank} ep={trainer.base.mesh.ep_size}", flush=True)

    def step(trainer, iterator):
        metrics = original_step(trainer, iterator)
        record = {
            "step": trainer.base.state.global_step,
            "loss": float(metrics["loss"]),
            "grad_norm": float(metrics["grad_norm"]),
        }
        if mode == "gate":
            if draft_loss is None or any(value == 0 for value in counts.values()):
                raise ValueError(f"Missing full-model execution evidence on rank {rank}: {counts}")
            record["dspark_loss"] = float(draft_loss)
        if not all(math.isfinite(value) for value in record.values()) or record["grad_norm"] <= 0:
            raise ValueError(f"Invalid full-model update on rank {rank}: {record}")
        steps.append(record)
        if mode == "gate":
            print(f"FULL_V41_UPDATE rank={rank} {json.dumps(record)}", flush=True)
        return metrics

    if mode == "gate":
        VLMTrainer.on_train_begin = begin
    VLMTrainer.train_step = step

    def write():
        if [row["step"] for row in steps] != list(range(1, expected_steps + 1)):
            raise ValueError(f"Rank {rank} did not finish {expected_steps} optimizer updates")
        result = {
            "rank": rank,
            "completed_steps": len(steps),
            "mode": mode,
            "steps": steps,
            "peak_npu_allocated_bytes": torch.npu.max_memory_allocated(),
            "peak_npu_reserved_bytes": torch.npu.max_memory_reserved(),
        }
        if mode == "gate":
            result["branch_forward_calls"] = counts
        receipt = output / f"rank-{rank:03}.json"
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(receipt)
    return write


def install_full_model_profile(rank, output):
    """Add multimodal regions to HP's existing stream-event profiling adapter."""
    from hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe import DeepseekV41MegaMoeExperts
    from hyper_parallel.models.deepseek_v41.adapter.engram.host_table import HostEngramTable
    from hyper_parallel.models.deepseek_v41.dspark import DeepseekV41DSpark, DSparkDraftAttention, DSparkStage
    from hyper_parallel.models.deepseek_v41.vision import DeepseekV41VisionTower

    profiler = Path(os.environ["DSV41_FULL_PROFILE_SCRIPT"])
    install = runpy.run_path(str(profiler))["install"]
    regions = (
        (DeepseekV41VisionTower, "forward", "vision_tower"),
        (DeepseekV41DSpark, "forward", "dspark"),
        (DSparkStage, "forward", "dspark_stage"),
        (DSparkDraftAttention, "forward", "dspark_attention"),
        (HostEngramTable, "forward", "engram_host_lookup"),
        (DeepseekV41MegaMoeExperts, "forward", "backbone_expert_dispatch_compute_combine"),
    )
    install(rank, output / f"stage-timings-rank-{rank:03}.json", VLMTrainer, (4, 5), regions)


def main():
    """Initialize the verified shared FileStore and launch the VLM training entry point."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != int(os.environ["DSV41_FULL_WORLD"]) or torch.npu.device_count() != 8:
        raise ValueError("The full model recipe must use its specified 8-die nodes")
    if os.environ.get('HCCL_DETERMINISTIC') != 'false':
        raise ValueError('HCCL policy must be disabled before process-group initialization')
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    print('NONDETERMINISTIC_PROCESS_BOOT rank=' + str(rank), flush=True)
    torch.npu.set_device(local_rank)
    output = Path(os.environ["DSV41_FULL_RUN_DIR"])
    timeout = timedelta(minutes=9)
    store = dist.FileStore(str(output / "rendezvous.store"), world)
    store.set_timeout(timeout)
    dist.init_process_group(backend="hccl", store=store, rank=rank, world_size=world, timeout=timeout)
    mode = os.environ.get("DSV41_FULL_MODE", "gate")
    if mode not in {"gate", "benchmark", "profile"}:
        raise ValueError(f"Unknown full-model run mode {mode}")
    write = install_smoke_receipt(rank, output / "receipts", int(os.environ["DSV41_FULL_STEPS"]), mode)
    if "DSV41_FULL_DIAGNOSTICS" in os.environ:
        runpy.run_path(os.environ["DSV41_FULL_DIAGNOSTICS"])["install"]()
    profile_requested = mode == "profile" or (mode == "gate" and "DSV41_FULL_PROFILE_SCRIPT" in os.environ)
    if profile_requested and rank in (0, 127, 128, 255):
        install_full_model_profile(rank, output)
    try:
        train_vlm()
        torch.npu.synchronize()
        write()
    except Exception as error:
        failure = output / f"failed-node-{os.environ['NODE_RANK']}-rank-{rank}"
        failure.write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # The launcher observes the marker on every node. A failed collective
        # cannot safely enter another collective during group destruction.
        os._exit(1)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
