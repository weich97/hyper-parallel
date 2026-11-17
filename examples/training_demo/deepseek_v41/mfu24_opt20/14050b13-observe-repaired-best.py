"""Record compact-cache execution without adding tensor probes to timing."""
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import runpy

from hyper_parallel.core.multicore.modules.mega_mhc.checkpoint_cache import checkpoint_cache_counts
from hyper_parallel.models.deepseek_v41.adapter.distributed.mega_mhc import DeepseekV41HyperMegaMhc
from hyper_parallel.trainer.vlm_trainer import VLMTrainer


def install():
    parent = Path('/xw-tree/full-v41/f42e4024-observe-rolling-gradient-guard.py')
    if hashlib.sha256(parent.read_bytes()).hexdigest() != 'f42e4024fc8741b99f265ea585dbfd0f010de99325f7e0dc0f0b77f2cf3d3a30':
        raise ValueError('Accepted observer changed')
    runpy.run_path(str(parent))['install']()
    begin, end = VLMTrainer.on_train_begin, VLMTrainer.on_train_end
    configured = []

    @wraps(begin)
    def started(trainer):
        result = begin(trainer)
        for name, module in trainer.base.model.named_modules():
            if isinstance(module, DeepseekV41HyperMegaMhc):
                if module.checkpoint_cache != 'mappings':
                    raise ValueError(f'Unexpected cache mode for {name}')
                configured.append(name)
        if len(configured) != 80:
            raise ValueError(f'Expected 80 mHC boundaries, got {len(configured)}')
        return result

    @wraps(end)
    def finished(trainer):
        result = end(trainer)
        steps = int(trainer.base.state.global_step)
        counters = checkpoint_cache_counts()
        expected = 80 * steps
        passed = counters['saved'] == expected and counters['replayed'] == expected
        root = Path(os.environ['DSV41_FULL_RUN_DIR']) / 'mhc-cache'
        root.mkdir(exist_ok=True)
        rank = int(os.environ['RANK'])
        (root / f'rank-{rank:03d}.json').write_text(json.dumps(dict(
            rank=rank, updates=steps, modules=configured, counts=counters,
            expected_calls=expected, passed=passed, cache_mode='mappings')) + '\n')
        if not passed:
            raise ValueError(f'mHC cache execution count differs: {counters}; expected {expected}')
        return result

    VLMTrainer.on_train_begin, VLMTrainer.on_train_end = started, finished
