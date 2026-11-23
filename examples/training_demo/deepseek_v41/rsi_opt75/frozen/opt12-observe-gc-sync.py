"""Stage 12: Chain Stage 10 + disable unsynchronized Python cyclic GC during steps."""
from functools import wraps
import gc
import hashlib
from pathlib import Path
import runpy

from hyper_parallel.trainer.vlm_trainer import VLMTrainer

PARENT = Path('/xw-tree/full-v41/opt10-observe-skip1.py')
PARENT_SHA = 'ca2c0af32df7ec184ce9520b34ecc9530105e5e045bb2c5411df37641f3f2c6a'


def install():
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError('Stage 10 observer changed')
    runpy.run_path(str(PARENT))['install']()

    orig_begin = VLMTrainer.on_train_begin
    orig_end = VLMTrainer.on_train_end

    @wraps(orig_begin)
    def wrapped_begin(trainer):
        res = orig_begin(trainer)
        gc.collect()
        gc.disable()
        return res

    @wraps(orig_end)
    def wrapped_end(trainer):
        gc.enable()
        gc.collect()
        return orig_end(trainer)

    VLMTrainer.on_train_begin = wrapped_begin
    VLMTrainer.on_train_end = wrapped_end
