"""Reuse native Host cache without modifying tensor or optimizer semantics."""
import hashlib
import json
import os
from pathlib import Path
import runpy

def install():
    root = Path('/xw-tree/full-v41')
    parent = root / '3a487fab-observe-csa-router-parallel-mhc.py'
    if hashlib.sha256(parent.read_bytes()).hexdigest() != '3a487fabe502768afe7a5dd4f89120f4bea0f10ce4fb3d01fcee10b0c4457344':
        raise ValueError('Parent changed')
    original = runpy.run_path
    intercepted = []

    def load(path, *args, **kwargs):
        source = Path(path)
        if source.name != '2e4edeec-observe-full-stage-replay-storage.py':
            return original(path, *args, **kwargs)
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != '2e4edeeccb71090d550f961d2299b8fca076f92fdfc2c133078ac06a32f44560':
            raise ValueError('Recovery policy changed')
        text = data.decode()
        replacements = [
            ('not drafter.stage_checkpoint or not drafter.stage_checkpoint_offload', 'drafter.stage_checkpoint or drafter.stage_checkpoint_offload'),
            ('three complete recomputed DSpark stages', 'three complete DSpark stages with local checkpoints only'),
            ('"recomputed_drafter_stages": 3, "checkpoint_input_offload": True', '"recomputed_drafter_stages": 0, "checkpoint_input_offload": False, "local_prefix_checkpoint": True'),
            ('    iterations = 0\n', '    iterations = 0\n    memory_samples = []\n'),
            ('        torch.npu.host_empty_cache()\n',
             '        if iterations < 2:\n            torch.npu.host_empty_cache()\n'),
            ('        iterations += 1\n',
             '        iterations += 1\n'
             '        status = {}\n'
             '        for line in Path("/proc/self/status").read_text().splitlines():\n'
             '            key, _, value = line.partition(":")\n'
             '            if key in ("VmRSS", "VmHWM", "VmLck", "VmPin"):\n'
             '                status[key + "_kib"] = int(value.split()[0])\n'
             '        memory_samples.append(dict(update=iterations, flushed=iterations <= 2, **status))\n'),
            ('"host_cache_policy": "flush-before-update"',
             '"host_cache_policy": "flush-first-two-updates-then-reuse", "host_memory": memory_samples'),
        ]
        for old, new in replacements:
            if text.count(old) != 1:
                raise ValueError('Unknown recovery wrapper boundary: ' + old[:60])
            text = text.replace(old, new)
        namespace = dict(__name__='pinned_cache_reuse_policy', __file__=str(source))
        exec(compile(text, str(source), 'exec'), namespace)
        intercepted.append(source.name)
        return namespace

    runpy.run_path = load
    try:
        original(str(parent))['install']()
    finally:
        runpy.run_path = original
    if intercepted != ['2e4edeec-observe-full-stage-replay-storage.py']:
        raise ValueError('Recovery policy was not replaced exactly once')

    from functools import wraps
    from hyper_parallel.models.deepseek_v41 import dspark as module
    from hyper_parallel.trainer.vlm_trainer import VLMTrainer
    previous_stage = module.DSparkStage.forward
    previous_checkpoint = module._checkpoint_stage
    previous_end = VLMTrainer.on_train_end
    calls = dict(stage_forward=0, whole_stage_checkpoint=0)

    @wraps(previous_stage)
    def stage_forward(self, *args, **kwargs):
        if kwargs.get('use_checkpoint') is not True:
            raise ValueError('The local DSpark prefix checkpoint must remain enabled')
        calls['stage_forward'] += 1
        return previous_stage(self, *args, **kwargs)

    @wraps(previous_checkpoint)
    def stage_checkpoint(*args, **kwargs):
        calls['whole_stage_checkpoint'] += 1
        raise ValueError('Whole-stage replay was requested in the selective-checkpoint experiment')

    @wraps(previous_end)
    def finished(trainer):
        result = previous_end(trainer)
        steps = trainer.base.state.global_step
        if calls != dict(stage_forward=3 * steps, whole_stage_checkpoint=0):
            raise ValueError('Unexpected DSpark execution count: ' + repr(calls))
        target = Path(os.environ['DSV41_FULL_RUN_DIR']) / 'dspark-checkpoint-scope'
        target.mkdir(exist_ok=True)
        rank = int(os.environ['RANK'])
        (target / f'rank-{rank:03d}.json').write_text(json.dumps(dict(
            rank=rank, updates=steps, local_prefix_checkpoint=True,
            architecture_and_objective_unchanged=True, **calls)) + '\n')
        return result

    module.DSparkStage.forward = stage_forward
    module._checkpoint_stage = stage_checkpoint
    VLMTrainer.on_train_end = finished
