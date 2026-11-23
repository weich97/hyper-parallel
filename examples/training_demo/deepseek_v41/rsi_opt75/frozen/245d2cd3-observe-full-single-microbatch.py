"""Audit real microbatch execution while retaining the established runtime policy."""
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import runpy

import torch

from hyper_parallel.core.multicore.modules.mega_mhc import function as forward_module
from hyper_parallel.core.multicore.modules.mega_mhc_grad import function as backward_module
from hyper_parallel.trainer.vlm_trainer import VLMTrainer

MICROBATCHES = 1
ORDERED_PROJECTION = True
POLICY = 'ordered_projection_parallel_vector_reductions'
SWAP_PREFIX = 32


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Unexpected observer contract: ' + before)
    return text.replace(before, after)


def install():
    root = Path('/xw-tree/full-v41')
    parent = root / '30d1b883-observe-dspark-anchor-sampling.py'
    run_path = runpy.run_path
    adapted = []
    specs = {
        '30d1b883-observe-dspark-anchor-sampling.py': (
            '30d1b88349e17b0623d75c523e2c6207e0fe77986552fbfca5c3555190475612',
            [('dict(update=len(events) + 1, positions=positions, population=selection.population,',
              f'dict(update=len(events) // {MICROBATCHES} + 1, micro_step=len(events) % {MICROBATCHES} + 1, '
              'positions=positions, population=selection.population,'),
             ('if len(events) != trainer.base.state.global_step:',
              f'if len(events) != trainer.base.state.global_step * {MICROBATCHES}:')]),
        '613f2e26-observe-npu-only-fill-phases.py': (
            '613f2e26bbd52f8146c85412c1a191af86d54f15ae7e0b35b3182a5e661843b8',
            [('["backbone", "dspark"] * steps:', f'["backbone", "dspark"] * steps * {MICROBATCHES}:'),
             ('layer.checkpoint_kwargs["swap_inputs"] = index < 32',
              f'layer.checkpoint_kwargs["swap_inputs"] = index < {SWAP_PREFIX}'),
             ('CHECKPOINT_INPUT_OFFLOAD_LAYERS=32/40', f'CHECKPOINT_INPUT_OFFLOAD_LAYERS={SWAP_PREFIX}/40'),
             ('"checkpoint_swap_prefix": 32,', f'"checkpoint_swap_prefix": {SWAP_PREFIX},')]),
        '4681135a-observe-full-stage-replay-mhc.py': (
            '4681135a87cc6ec0b8968b88793d5b3a812a489f98d77a0d77686f38d0c610b8',
            [('if native_calls != 80 * steps:', f'if native_calls != 80 * steps * {MICROBATCHES}:'),
             ('"parameter_reduction": "original_order"', f'"parameter_reduction": "{POLICY}"'),
             ('ordered_reductions_parallel_activation_output', POLICY)]),
    }

    def load(path, *args, **kwargs):
        observed = Path(path)
        if observed.name == '2fd2889b-diagnose-full-v41-mhc-reduction-order.py':
            if hashlib.sha256(observed.read_bytes()).hexdigest() != '2fd2889b7f5bd7a0a0140aa1208d93fa8e3d98ec8f96de95cef30570c4f5f4aa':
                raise ValueError('Superseded diagnostic changed')
            initialized = root / 'df307082-verify-full-v41-dspark-initialization.py'
            if hashlib.sha256(initialized.read_bytes()).hexdigest() != 'df307082da2cd4ef416cf5ad58590b8f79014962636da766bb704e3d3db27262':
                raise ValueError('Initialization observer changed')
            return run_path(str(initialized))
        if observed.name not in specs:
            return run_path(path, *args, **kwargs)
        expected, replacements = specs[observed.name]
        payload = observed.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('Frozen observer changed: ' + observed.name)
        text = payload.decode()
        for before, after in replacements:
            text = replace_once(text, before, after)
        namespace = {'__name__': 'audited_multibatch_' + observed.stem, '__file__': str(observed)}
        exec(compile(text, str(observed), 'exec'), namespace)
        adapted.append(observed.name)
        return namespace

    runpy.run_path = load
    try:
        load(str(parent))['install']()
    finally:
        runpy.run_path = run_path
    if sorted(adapted) != sorted(specs):
        raise ValueError('Not every microbatch accounting contract was adapted')
    original = backward_module.hyper_mega_mhc_grad
    if forward_module.hyper_mega_mhc_grad is not original:
        raise ValueError('Inconsistent mHC bindings')
    begin, end = VLMTrainer.on_train_begin, VLMTrainer.on_train_end
    calls = 0

    @wraps(original)
    def backward(*args, **kwargs):
        nonlocal calls
        if any(key in kwargs for key in ('allow_parallel_reduction', 'ordered_projection')):
            raise ValueError('Conflicting reduction policy')
        result = original(*args, **kwargs, allow_parallel_reduction=True, ordered_projection=ORDERED_PROJECTION)
        calls += 1
        return result

    @wraps(begin)
    def started(trainer):
        config = trainer.base.config
        if config.training.global_batch_size != 256 * MICROBATCHES or config.training.micro_batch_size != 1:
            raise ValueError('The declared microbatch count differs from the recipe')
        if not config.fsdp_config.requires_grad_sync or not config.fsdp_config.reshard_after_backward:
            raise ValueError('Retain per-microbatch reduce-scatter and parameter resharding')
        return begin(trainer)

    @wraps(end)
    def finished(trainer):
        result = end(trainer)
        steps = trainer.base.state.global_step
        if calls != 80 * steps * MICROBATCHES or not torch.are_deterministic_algorithms_enabled():
            raise ValueError('Incomplete mHC microbatch coverage or changed global policy')
        target = Path(os.environ['DSV41_FULL_RUN_DIR']) / 'multibatch-mhc'
        target.mkdir(exist_ok=True)
        rank = int(os.environ['RANK'])
        row = dict(rank=rank, updates=steps, microbatches_per_update=MICROBATCHES,
                   calls=calls, parameter_reduction=POLICY, checkpoint_swap_prefix=SWAP_PREFIX,
                   bitwise_repeatability_guaranteed=False, full_model_promotion=False,
                   global_determinism=True, attention_backend='flash_v3', adapted_observers=adapted)
        (target / f'rank-{rank:03d}.json').write_text(json.dumps(row) + '\n')
        return result

    backward_module.hyper_mega_mhc_grad = backward
    forward_module.hyper_mega_mhc_grad = backward
    VLMTrainer.on_train_begin, VLMTrainer.on_train_end = started, finished
    print(f'FULL_V41_MICROBATCHES={MICROBATCHES} MHC_POLICY={POLICY}', flush=True)
