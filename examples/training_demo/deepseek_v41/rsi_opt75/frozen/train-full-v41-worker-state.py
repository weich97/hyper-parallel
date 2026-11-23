"""Start HyperParallel's text entry point with the verified C3 rendezvous."""

from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from examples.training_demo.train_text import main as train_text
from hyper_parallel.trainer.text_trainer import TextTrainer


def append_tensor(digest, name, tensor) -> None:
    """Hash local values in bounded CPU chunks, including type and layout identity."""
    local = tensor.to_local() if callable(getattr(tensor, "to_local", None)) else tensor
    metadata = [name, str(tensor.dtype), list(tensor.shape), list(local.shape)]
    digest.update(json.dumps(metadata, separators=(",", ":")).encode("utf-8") + b"\0")
    elements_per_chunk = (64 * 2**20) // local.element_size()
    for chunk in local.detach().reshape(-1).split(elements_per_chunk):
        raw = chunk.cpu().contiguous().view(torch.uint8)
        digest.update(memoryview(raw.numpy()))


def install_initial_state_receipt() -> None:
    """Record scratch state immediately before the first training forward."""
    original = TextTrainer.on_train_begin

    def record(trainer):
        original(trainer)
        base = trainer.base
        if base.state.global_step != 0:
            raise RuntimeError("Initial-state fingerprinting requires a fresh optimizer run")
        leaves = base.optimizer.chained_optimizers
        if any(optimizer.state for optimizer in leaves):
            raise RuntimeError("The scratch optimizer unexpectedly holds materialized state")
        digest = hashlib.sha256()
        parameter_count = 0
        for name, parameter in sorted(base.model.named_parameters()):
            append_tensor(digest, name, parameter)
            parameter_count += 1
            main_parameter = getattr(parameter, "main_param", None)
            if main_parameter is not None and main_parameter is not parameter:
                append_tensor(digest, name + ".main_param", main_parameter)
        for name, buffer in sorted(base.model.named_buffers()):
            append_tensor(digest, name, buffer)
        rng = hashlib.sha256()
        append_tensor(rng, "cpu_rng", torch.get_rng_state())
        append_tensor(rng, "npu_rng", torch.npu.get_rng_state())
        rank = dist.get_rank()
        receipt = {"rank": rank, "step": 0, "named_parameters": parameter_count,
                   "model_state_sha256": digest.hexdigest(), "rng_sha256": rng.hexdigest(),
                   "optimizer_leaves": [type(optimizer).__name__ for optimizer in leaves], "optimizer_state_entries": 0}
        output = Path(os.environ["DSV41_RUN_DIR"]) / "initial-state"
        output.mkdir(exist_ok=True)
        (output / f"rank-{rank:03}.json").write_text(json.dumps(receipt) + "\n", encoding="utf-8")
        if int(os.environ["LOCAL_RANK"]) == 0:
            print(f"DSV41_INITIAL_STATE_READY rank={rank}", flush=True)

    TextTrainer.on_train_begin = record


def main() -> None:
    """Attach to the allocation and use the upstream Trainer to run the recipe."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 128 or torch.npu.device_count() != 8:
        raise RuntimeError("The full experiment requires 16 nodes with 8 visible dies each")
    torch.npu.set_device(local_rank)
    store = dist.FileStore(os.environ["DSV41_STORE_FILE"], world_size)
    store.set_timeout(timedelta(minutes=8))
    dist.init_process_group(backend="hccl", store=store, rank=rank, world_size=world_size,
                            timeout=timedelta(minutes=8))
    if os.environ.get("DSV41_INITIAL_STATE_RECEIPT") == "1":
        install_initial_state_receipt()
    if local_rank == 0:
        print(f"DSV41_PROCESS_GROUP_READY rank={rank}", flush=True)
    try:
        train_text()
        torch.npu.synchronize()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    output = Path(os.environ["DSV41_RUN_DIR"]) / "completed-ranks"
    (output / f"{rank:03}.txt").write_text("TRAINER_RETURNED\n", encoding="utf-8")
    sys.stdout.flush()
    sys.stderr.flush()
    # The worker has synchronized and destroyed its process groups. Avoid the
    # known interpreter teardown hang in this training image.
    os._exit(0)


if __name__ == "__main__":
    main()
