"""Stage 66 (32-node / 40L, ZERO precision compromise):
  opt65 stack (19.9339s / 25.5252% MFU)
  + DSpark loss_chunk_size=512 (2560 tokens/chunk, 2 chunks vs 4 chunks;
    single-NPU 123.942 -> 104.443 ms and lower peak alloc 9.359 -> 8.753 GB).
"""
from functools import wraps
from pathlib import Path
import runpy

import torch.distributed as dist

import hyper_parallel.models.deepseek_v41.dspark as dspark_mod

OPT65_PATH = Path("/xw-tree/full-v41/opt65-opt64-dspark-chunk256.py")


def install():
    runpy.run_path(str(OPT65_PATH))["install"]()

    orig_init = dspark_mod.DeepseekV41DSpark.__init__

    @wraps(orig_init)
    def wrapped_init(self, config, mlp_factory):
        orig_init(self, config, mlp_factory)
        self.loss_chunk_size = 512

    dspark_mod.DeepseekV41DSpark.__init__ = wrapped_init

    if not dist.is_initialized() or dist.get_rank() == 0:
        print("OPT66_DSPARK_CHUNK512_INSTALLED=1", flush=True)
