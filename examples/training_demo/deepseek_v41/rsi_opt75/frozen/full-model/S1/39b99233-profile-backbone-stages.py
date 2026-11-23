"""Measure rank-local training stage spans in a dedicated C3 profiling run."""

from collections import defaultdict
from functools import wraps
import json
import os
from pathlib import Path
import runpy
from time import perf_counter

import torch
import torch_npu  # noqa: F401

from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.text_trainer import TextTrainer


def install(rank: int, output: Path, trainer_type=TextTrainer, profile_steps=(6, 7), extra_regions=()) -> None:
    """Record NPU events on the Trainer stream without fencing each phase."""
    events = defaultdict(list)
    active_step = None

    def time_call(name, method, *args, **kwargs):
        if active_step is None:
            return method(*args, **kwargs)
        start_event, end_event = (torch.npu.Event(enable_timing=True) for _ in range(2))
        start_event.record()
        start = perf_counter()
        result = method(*args, **kwargs)
        host_ms = (perf_counter() - start) * 1000
        end_event.record()
        events[active_step].append((name, start_event, end_event, host_ms))
        return result

    def wrap(owner, method_name, label):
        original = getattr(owner, method_name)

        @wraps(original)
        def measured(*args, **kwargs):
            return time_call(label, original, *args, **kwargs)

        setattr(owner, method_name, measured)

    wrap(trainer_type, "forward_backward_step", "input_and_forward_backward")
    wrap(BaseTrainer, "forward_backward_step", "forward_backward")
    wrap(BaseTrainer, "prepare_optimizer_step", "gradient_preparation")
    wrap(BaseTrainer, "step_optimizers_and_schedulers", "optimizer_and_scheduler")
    wrap(torch.Tensor, "backward", "backward")
    for owner, method_name, label in extra_regions:
        wrap(owner, method_name, label)
    original_step, original_end = trainer_type.train_step, trainer_type.on_train_end

    @wraps(original_step)
    def train_step(trainer, *args, **kwargs):
        nonlocal active_step
        step = trainer.base.state.global_step + 1
        active_step = step if step in profile_steps else None
        try:
            return time_call("optimizer_step", original_step, trainer, *args, **kwargs)
        finally:
            active_step = None

    @wraps(original_end)
    def train_end(trainer):
        original_end(trainer)
        torch.npu.synchronize()
        report = {}
        for step, records in sorted(events.items()):
            durations = defaultdict(lambda: {"stream_span_ms": 0.0, "host_call_ms": 0.0, "calls": 0})
            for name, start, end, host_ms in records:
                durations[name]["stream_span_ms"] += start.elapsed_time(end)
                durations[name]["host_call_ms"] += host_ms
                durations[name]["calls"] += 1
            report[step] = dict(durations)
        output.write_text(json.dumps({"rank": rank, "steps": report}, indent=2) + "\n", encoding="utf-8")

    trainer_type.train_step = train_step
    trainer_type.on_train_end = train_end


if __name__ == "__main__":
    worker = Path(__file__).with_name("train-full-v41-worker-state.py")
    rank = int(os.environ["RANK"])
    ranks = {int(value) for value in os.environ.get("DSV41_STAGE_PROFILE_RANKS", "0,63").split(",")}
    if rank in ranks:
        install(rank, Path(os.environ["DSV41_RUN_DIR"]) / f"stage-timings-rank-{rank:03}.json")
    runpy.run_path(str(worker))["main"]()
