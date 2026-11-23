"""Stage 69 (32-node / 40L, ZERO precision compromise):
  opt68 stack (19.6848s / 25.8482% MFU, peak 19.6682s / 25.8700% MFU, max_alloc=50.6927 GB)
  + 1. Replace modeling_deepseek_v41._hc_pre with dspark._collapse_streams (_FusedStreamCollapse Triton kernel),
       eliminating the 2.50 GiB FP32 allocation spike (1.25 GiB persistent across chunk_loss and DSpark) at
       `self.norm(_hc_pre(hidden_states, pre_mix))`.
  + 2. Skip redundant dspark._checkpoint on `_collapse` (which already uses _FusedStreamCollapse with no FP32 buffer).
  + 3. Set DSparkDraftAttention.attention_chunk_size = 1024 (1 chunk of 5120 query tokens per stage vs 2 chunks of 2560,
       halving FlashAttentionScore / FlashAttentionScoreGrad / RoPE / wo_a launches in DSpark, -12.2 ms/step).
"""
from functools import wraps
from pathlib import Path
import runpy

import torch.distributed as dist

import hyper_parallel.models.deepseek_v41.dspark as dspark_mod
from hyper_parallel.models.deepseek_v41.dspark import DSparkDraftAttention
import hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 as modeling_mod

OPT68_PATH = Path("/xw-tree/full-v41/opt68-opt66-dspark-attn-rope.py")


def install():
    runpy.run_path(str(OPT68_PATH))["install"]()

    # 1. Replace modeling_deepseek_v41._hc_pre with fused Triton _collapse_streams (_FusedStreamCollapse)
    modeling_mod._hc_pre = dspark_mod._collapse_streams

    # 2. Skip redundant _checkpoint on _collapse in dspark.py
    orig_dspark_ckpt = dspark_mod._checkpoint

    @wraps(orig_dspark_ckpt)
    def fast_dspark_ckpt(function, *args):
        if getattr(function, "__name__", None) == "_collapse":
            return function(*args)
        return orig_dspark_ckpt(function, *args)

    dspark_mod._checkpoint = fast_dspark_ckpt

    # 3. Set DSparkDraftAttention.attention_chunk_size = 1024
    orig_attn_init = DSparkDraftAttention.__init__

    @wraps(orig_attn_init)
    def fast_attn_init(self, config, *args, **kwargs):
        orig_attn_init(self, config, *args, **kwargs)
        if 0 < self.attention_chunk_size < 1024:
            self.attention_chunk_size = 1024

    DSparkDraftAttention.__init__ = fast_attn_init

    if not dist.is_initialized() or dist.get_rank() == 0:
        print("OPT69_HCPRE_DSPARK1024_INSTALLED=1", flush=True)
