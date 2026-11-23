"""Stage 10: Chain Stage 9 + skip activation_checkpoint on layer 39 only."""
from functools import wraps
import hashlib
import os
from pathlib import Path
import runpy

from hyper_parallel.core.activation_memory import api as act_mem_api
from hyper_parallel.core.multicore.modules.mega_mhc.checkpoint_cache import checkpoint_cache_counts
from hyper_parallel.trainer.vlm_trainer import VLMTrainer

PARENT = Path('/xw-tree/full-v41/opt9-observe-engram-recompute-cache.py')
PARENT_SHA = 'c8831047ea034e5b8c402effd21cfb70da00e265b139afbe4b9b19e7ba8cb6a4'
SKIP_FROM_LAYER = 39


def install():
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError('Stage 9 observer changed')
    runpy.run_path(str(PARENT))['install']()

    orig_checkpoint = act_mem_api.checkpoint
    skipped_layers = set()

    @wraps(orig_checkpoint)
    def selective_checkpoint(function, *args, **kwargs):
        layer_idx = getattr(function, 'layer_idx', None)
        if isinstance(layer_idx, int) and layer_idx >= SKIP_FROM_LAYER:
            skipped_layers.add(layer_idx)
            call_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in ('swap_inputs', 'policy_fn', 'context_fn', 'group_swap', 'cpu_pool', 'early_stop', 'use_reentrant', 'preserve_rng_state')
            }
            return function(*args, **call_kwargs)
        return orig_checkpoint(function, *args, **kwargs)

    act_mem_api.checkpoint = selective_checkpoint

    orig_end = VLMTrainer.on_train_end

    @wraps(orig_end)
    def wrapped_end(trainer):
        try:
            return orig_end(trainer)
        except ValueError as exc:
            if 'mHC cache execution count differs' not in str(exc):
                raise
            steps = int(trainer.base.state.global_step)
            counters = checkpoint_cache_counts()
            expected = (80 - 2 * len(skipped_layers)) * steps
            if counters['saved'] != expected or counters['replayed'] != expected:
                raise ValueError(f'Selective mHC cache mismatch: {counters}; expected {expected}') from exc
            if int(os.environ.get('RANK', '0')) == 0:
                print(f'SELECTIVE_RECOMPUTE_OK skipped={sorted(skipped_layers)} counters={counters}', flush=True)
            return None

    VLMTrainer.on_train_end = wrapped_end
