"""Stage 48 (32-node / 40L, ZERO precision compromise): opt47 + VLMTrainer._train_epoch BackgroundPrefetcher."""
from functools import wraps
from pathlib import Path
import runpy

import torch

from hyper_parallel.trainer.runtime.data_iterator import HyperIter
from hyper_parallel.trainer.vlm_trainer import VLMTrainer

OPT47_PATH = Path("/xw-tree/full-v41/opt47-opt45-bgprefetch-swap64.py")


def install():
    runpy.run_path(str(OPT47_PATH))["install"]()

    # Wrap VLMTrainer._train_epoch to use HyperIter(use_background_prefetcher=True, maxsize=2)
    orig_train_epoch = VLMTrainer._train_epoch
    logged = {"done": False}

    @wraps(orig_train_epoch)
    def bg_train_epoch(self, epoch: int, balanced_loader: bool) -> bool:
        orig_dl = self.base.train_dataloader
        if orig_dl is None:
            return orig_train_epoch(self, epoch, balanced_loader)
        wrapped_dl = HyperIter(orig_dl, use_background_prefetcher=True, maxsize=2)
        if not logged["done"]:
            logged["done"] = True
            if torch.distributed.get_rank() == 0:
                print("OPT48_VLM_BG_PREFETCHER=True maxsize=2", flush=True)
        self.base.train_dataloader = wrapped_dl
        try:
            return orig_train_epoch(self, epoch, balanced_loader)
        finally:
            wrapped_dl.stop()
            self.base.train_dataloader = orig_dl

    VLMTrainer._train_epoch = bg_train_epoch
