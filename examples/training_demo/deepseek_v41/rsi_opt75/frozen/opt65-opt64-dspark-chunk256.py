"""Stage 65 (32-node / 40L, ZERO precision compromise):
  opt64 stack (19.9803s / 25.4659% MFU)
  + DSpark loss_chunk_size=256 (1280 tokens/chunk, 4 chunks vs 16 chunks;
    single-NPU 246.338 -> 124.018 ms, ~-122 ms/step, chunking-invariant loss).
"""
from functools import wraps
from pathlib import Path
import runpy

import torch.distributed as dist

import hyper_parallel.models.deepseek_v41.dspark as dspark_mod

OPT64_PATH = Path("/xw-tree/full-v41/opt64-opt63-grouped-proj-bwd.py")


def install():
    runpy.run_path(str(OPT64_PATH))["install"]()

    orig_init = dspark_mod.DeepseekV41DSpark.__init__

    @wraps(orig_init)
    def wrapped_init(self, config, mlp_factory):
        orig_init(self, config, mlp_factory)
        self.loss_chunk_size = 256

    dspark_mod.DeepseekV41DSpark.__init__ = wrapped_init

    if not dist.is_initialized() or dist.get_rank() == 0:
        print("OPT65_DSPARK_CHUNK256_INSTALLED=1", flush=True)
