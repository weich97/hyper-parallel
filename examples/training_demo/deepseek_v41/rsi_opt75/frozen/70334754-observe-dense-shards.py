"""Verify actual HSDP mesh sizes without adding timed tensor probes."""
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import runpy
from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.trainer.vlm_trainer import VLMTrainer

def install():
    parent = Path(os.environ['DSV41_SHARD_PARENT'])
    if hashlib.sha256(parent.read_bytes()).hexdigest() != os.environ['DSV41_SHARD_PARENT_SHA']:
        raise ValueError('Parent changed')
    runpy.run_path(str(parent))['install']()
    begin, end = VLMTrainer.on_train_begin, VLMTrainer.on_train_end
    rows = []
    @wraps(begin)
    def started(trainer):
        result = begin(trainer)
        expected = int(os.environ['DSV41_EXPECT_DENSE_SHARDS'])
        if trainer.base.config.fsdp_config.dp_shard_size != expected:
            raise ValueError('Recipe mismatch')
        for name, module in trainer.base.model.named_modules():
            if not isinstance(module, HSDPModule):
                continue
            state = module.hsdp_scheduler.hsdp_state
            mesh = state.mesh
            dims = tuple(mesh.mesh_dim_names or ())
            sizes = tuple(int(mesh.size(index)) for index in range(mesh.ndim))
            if 'fsdp_shard' in dims and sizes[dims.index('fsdp_shard')] != expected:
                raise ValueError('Dense mesh differs from plan')
            rows.append(dict(module=name, dimensions=dims, sizes=sizes))
        if not rows or not any('fsdp_shard' in row['dimensions'] for row in rows):
            raise ValueError('No dense FSDP units found')
        return result
    @wraps(end)
    def finished(trainer):
        result = end(trainer)
        if trainer.base.state.global_step != 8:
            raise ValueError('Incomplete screen')
        root = Path(os.environ['DSV41_FULL_RUN_DIR']) / 'dense-shards'
        root.mkdir(exist_ok=True)
        rank = int(os.environ['RANK'])
        (root / f'rank-{rank:03d}.json').write_text(json.dumps(dict(rank=rank, updates=8, units=rows))+'\n')
        return result
    VLMTrainer.on_train_begin, VLMTrainer.on_train_end = started, finished
