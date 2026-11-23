"""Stage 26 final clean wrapper (pure FP32): Stage 26 + clean Selective mHC cache count verification for skipped layer 39."""
from functools import wraps
import os
from pathlib import Path
import runpy

from hyper_parallel.core.multicore.modules.mega_mhc.checkpoint_cache import checkpoint_cache_counts
from hyper_parallel.trainer.vlm_trainer import VLMTrainer

OPT26_PATH = Path("/xw-tree/full-v41/opt26-teacher-chunk1k.py")


def install():
    runpy.run_path(str(OPT26_PATH))["install"]()

    orig_end = VLMTrainer.on_train_end

    @wraps(orig_end)
    def wrapped_end(trainer):
        try:
            return orig_end(trainer)
        except ValueError as exc:
            if "Selective mHC cache mismatch" not in str(exc):
                raise
            steps = int(trainer.base.state.global_step)
            counters = checkpoint_cache_counts()
            expected = 78 * steps
            if counters["saved"] != expected or counters["replayed"] != expected:
                raise
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"OPT26_SELECTIVE_RECOMPUTE_OK skipped=[39] counters={counters}", flush=True)
            return None

    VLMTrainer.on_train_end = wrapped_end
