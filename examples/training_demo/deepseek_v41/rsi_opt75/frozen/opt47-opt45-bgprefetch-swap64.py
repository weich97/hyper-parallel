"""Stage 47 (32-node / 40L, ZERO precision compromise): opt45 + background dataloader prefetcher + swap_times=64."""
from functools import wraps
from pathlib import Path
import runpy

import hyper_parallel.core.optimizer.swap_optimizer_base as swap_base_mod
import hyper_parallel.trainer.runtime.data_iterator as data_iter_mod

OPT45_PATH = Path("/xw-tree/full-v41/opt45-opt43-kl-swap32-attn35.py")


def install():
    runpy.run_path(str(OPT45_PATH))["install"]()

    # 1. Enable background dataloader prefetcher (overlaps next-batch CPU/H2D with NPU compute)
    orig_iter_init = data_iter_mod.HyperIter.__init__
    iter_logged = {"done": False}

    @wraps(orig_iter_init)
    def bg_iter_init(self, dataloader, use_background_prefetcher=False, maxsize=1):
        if not iter_logged["done"]:
            iter_logged["done"] = True
            print(f"OPT47_BG_PREFETCHER orig={use_background_prefetcher} -> True", flush=True)
        return orig_iter_init(self, dataloader, use_background_prefetcher=True, maxsize=max(2, maxsize))

    data_iter_mod.HyperIter.__init__ = bg_iter_init

    # 2. Finer SwapOptimizer pipeline (swap_times=64)
    orig_swap_init = swap_base_mod.SwapOptimizer.__init__
    swap_logged = {"done": False}

    @wraps(orig_swap_init)
    def fast_swap_init(self, optimizer, config, *args, **kwargs):
        object.__setattr__(config, "swap_times", 64)
        if not swap_logged["done"]:
            swap_logged["done"] = True
            print("OPT47_SWAP_TIMES=64", flush=True)
        return orig_swap_init(self, optimizer, config, *args, **kwargs)

    swap_base_mod.SwapOptimizer.__init__ = fast_swap_init
