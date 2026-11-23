"""Stage 2: Chain Stage 1 (EDP balance) + Vision FSDP comm_fusion (4714ab8e)."""
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import runpy

from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.distributed._builder import fsdp_adapter
from hyper_parallel.trainer.vlm_trainer import VLMTrainer

PARENT = Path('/xw-tree/full-v41/opt1-observe-edp-balance.py')
PARENT_SHA = '8a15af39c469572727b6deac8213006b60772f10ef1bac10fd7d8d212868f3bf'
PREFIX = 'model.vision.'


def install():
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError('Stage 1 observer changed')
    runpy.run_path(str(PARENT))['install']()

    fused_units = []
    original_units = fsdp_adapter.FSDP2Manager._parallelize_child_units

    @wraps(original_units)
    def parallelize_child_units(self, wrap_modules, *args, **kwargs):
        vision_ids = {id(w.module): w.fqn for w in wrap_modules if w.fqn.startswith(PREFIX)}
        original_shard = fsdp_adapter.fully_shard

        def shard(module, *shard_args, **shard_kwargs):
            if id(module) in vision_ids:
                shard_kwargs = dict(shard_kwargs, comm_fusion=True, comm_fusion_zero_copy=False)
                fused_units.append(vision_ids[id(module)])
            return original_shard(module, *shard_args, **shard_kwargs)

        fsdp_adapter.fully_shard = shard
        try:
            return original_units(self, wrap_modules, *args, **kwargs)
        finally:
            fsdp_adapter.fully_shard = original_shard

    fsdp_adapter.FSDP2Manager._parallelize_child_units = parallelize_child_units

    begin, end = VLMTrainer.on_train_begin, VLMTrainer.on_train_end
    observed = {}

    @wraps(begin)
    def started(trainer):
        result = begin(trainer)
        fused, unfused_vision, fused_other = [], [], []
        for name, module in trainer.base.model.named_modules():
            if not isinstance(module, HSDPModule):
                continue
            enabled = bool(module.hsdp_scheduler.comm_fusion_policy.enable_comm_fusion)
            is_vision = name.startswith(PREFIX)
            if enabled and is_vision:
                fused.append(name)
            elif is_vision:
                unfused_vision.append(name)
            elif enabled:
                fused_other.append(name)
        if not fused or unfused_vision or fused_other:
            raise ValueError(f'Vision fusion scope mismatch: fused={len(fused)} '
                             f'unfused_vision={unfused_vision} fused_other={fused_other}')
        if sorted(fused) != sorted(fused_units):
            raise ValueError('Fused vision units differ from wrapped units')
        observed.update(fused=fused)
        return result

    @wraps(end)
    def finished(trainer):
        result = end(trainer)
        root = Path(os.environ['DSV41_FULL_RUN_DIR']) / 'vision-comm-fusion'
        root.mkdir(exist_ok=True)
        rank = int(os.environ['RANK'])
        (root / f'rank-{rank:03d}.json').write_text(json.dumps(dict(
            rank=rank, updates=int(trainer.base.state.global_step),
            fused_units=observed.get('fused', []), fused_count=len(observed.get('fused', [])),
            zero_copy=False, passed=bool(observed.get('fused')))) + '\n')
        return result

    VLMTrainer.on_train_begin, VLMTrainer.on_train_end = started, finished
