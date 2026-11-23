"""Stage 52 (32-node / 40L, ZERO precision compromise): opt50 (layer 35 dense no-reshard, rock-solid 50.45 GB) + swap_times=96."""
from functools import wraps
from pathlib import Path
import runpy

import hyper_parallel.core.optimizer.swap_optimizer_base as swap_base_mod

OPT50_PATH = Path("/xw-tree/full-v41/opt50-opt48-nr35dense-muonclean.py")


def install():
    runpy.run_path(str(OPT50_PATH))["install"]()

    # Finer SwapOptimizer pipeline (swap_times=96) without layer 34 no-reshard
    orig_swap_init = swap_base_mod.SwapOptimizer.__init__
    swap_logged = {"done": False}

    @wraps(orig_swap_init)
    def fast_swap_init(self, optimizer, config, *args, **kwargs):
        object.__setattr__(config, "swap_times", 96)
        if not swap_logged["done"]:
            swap_logged["done"] = True
            print("OPT52_SWAP_TIMES=96", flush=True)
        return orig_swap_init(self, optimizer, config, *args, **kwargs)

    swap_base_mod.SwapOptimizer.__init__ = fast_swap_init
