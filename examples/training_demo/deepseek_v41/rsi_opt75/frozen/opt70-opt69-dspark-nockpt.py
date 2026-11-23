"""Stage 70 (32-node / 40L, ZERO precision compromise):
  opt69 stack (19.6607s / 25.8799% MFU, peak 19.6501s / 25.8939% MFU, max_alloc=50.0677 GB)
  + 1. Eliminate DSparkStage._pre_ffn recomputation in dspark._checkpoint (while keeping use_checkpoint=True
       on DSparkStage.forward for 0fe83cff compliance), utilizing the ~5.9 GiB free HBM headroom (50.07 / 56.00 GiB)
       to save all 3 stages' _pre_ffn activations (~3.4 GiB during DSpark only) and eliminate 3x full DSparkDraftAttention
       + mHC forward recomputations (~130 ms/step).
  + 2. Set mlp.recompute_shared_experts = False on DSpark stages (only 45 MiB/stage = 135 MiB total across 3 stages,
       eliminating 3x shared_experts forward recomputations).
  + 3. Set set_reshard_after_forward(False, recurse=False) on dspark.stages[0..2] non-expert FSDP units (~768 MiB total),
       eliminating 3x 256-rank cross-superpod FSDP AllGathers during DSpark backward.
"""
from functools import wraps
from pathlib import Path
import runpy

import torch.distributed as dist

import hyper_parallel.models.deepseek_v41.dspark as dspark_mod
from hyper_parallel.models.deepseek_v41.dspark import DeepseekV41DSpark
import hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 as modeling_mod

OPT69_PATH = Path("/xw-tree/full-v41/opt69-opt68-hcpre-lm2048-dspark1024.py")


def install():
    runpy.run_path(str(OPT69_PATH))["install"]()

    # 1. Skip _checkpoint on both _collapse and _pre_ffn inside dspark.py
    orig_dspark_ckpt = dspark_mod._checkpoint

    @wraps(orig_dspark_ckpt)
    def fast_dspark_ckpt(function, *args):
        if getattr(function, "__name__", None) in ("_collapse", "_pre_ffn"):
            return function(*args)
        return orig_dspark_ckpt(function, *args)

    dspark_mod._checkpoint = fast_dspark_ckpt

    # 2. Disable recompute_shared_experts on DSpark MLP modules (only 135 MiB across 3 stages)
    orig_build_dspark_mlp = modeling_mod._build_dspark_mlp

    @wraps(orig_build_dspark_mlp)
    def fast_build_dspark_mlp(config, *args, **kwargs):
        mlp = orig_build_dspark_mlp(config, *args, **kwargs)
        mlp.recompute_shared_experts = False
        return mlp

    modeling_mod._build_dspark_mlp = fast_build_dspark_mlp

    # 3. Also configure existing/constructed DSpark stages: recompute_shared_experts=False and set_reshard_after_forward(False, recurse=False)
    orig_run_stages = DeepseekV41DSpark._run_stages
    dspark_configured = {"done": False}

    @wraps(orig_run_stages)
    def fast_run_stages(self, *args, **kwargs):
        if not dspark_configured["done"]:
            dspark_configured["done"] = True
            for stage in self.stages:
                if hasattr(stage, "mlp") and hasattr(stage.mlp, "recompute_shared_experts"):
                    stage.mlp.recompute_shared_experts = False
                if hasattr(stage, "set_reshard_after_forward"):
                    stage.set_reshard_after_forward(False, recurse=False)
            if not dist.is_initialized() or dist.get_rank() == 0:
                print("OPT70_DSPARK_STAGES_NO_RECOMPUTE_NO_RESHARD=1", flush=True)
        return orig_run_stages(self, *args, **kwargs)

    DeepseekV41DSpark._run_stages = fast_run_stages

    if not dist.is_initialized() or dist.get_rank() == 0:
        print("OPT70_DSPARK_NOCKPT_INSTALLED=1", flush=True)
