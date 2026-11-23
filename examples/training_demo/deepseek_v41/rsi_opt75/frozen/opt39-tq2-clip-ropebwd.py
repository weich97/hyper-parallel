"""Stage 39 (32-node / 40L, ZERO precision compromise): opt26-final-s15 + TASK_QUEUE_ENABLE=2 + vec clip_grad + cached _repeat_cs + inplace vec RoPE bwd."""
from pathlib import Path
import runpy

import torch

import hyper_parallel.components.functional._triton.v41_partial_rope as rope_mod
import hyper_parallel.core.utils.clip_grad as clip_grad_mod

OPT26_PATH = Path("/xw-tree/full-v41/opt26-final-s15.py")
OPT20_PATH = Path("/xw-tree/full-v41/opt20-observe-vec-rope-skip1-clip-muon.py")


def install():
    runpy.run_path(str(OPT26_PATH))["install"]()

    # 1. Vectorized stack + mul in clip_grad._foreach_p_norms (saved ~41ms)
    def fast_vec_p_norms(grads, norm_type: float, device: torch.device) -> torch.Tensor:
        total = torch.tensor(0.0, device=device, dtype=torch.float32)
        grouped = clip_grad_mod._group_tensors_by_device_and_dtype([[g.detach() for g in grads]])
        for (dev, _), ([dev_grads], _) in grouped.items():
            if clip_grad_mod._has_foreach_support(dev_grads, dev):
                try:
                    per_norms = torch._foreach_norm(dev_grads, norm_type)
                except RuntimeError:
                    per_norms = None
                if per_norms is not None:
                    stacked = torch.stack(per_norms).to(device=device, dtype=torch.float32)
                    if norm_type == 2.0:
                        total.add_((stacked * stacked).sum())
                    else:
                        total.add_((stacked ** norm_type).sum())
                else:
                    clip_grad_mod._sum_p_norms(dev_grads, norm_type, device, total)
            else:
                clip_grad_mod._sum_p_norms(dev_grads, norm_type, device, total)
        return total

    clip_grad_mod._foreach_p_norms = fast_vec_p_norms

    # 2. Cache cos/sin repeat_interleave(2)
    rep_cache = {}

    def cached_repeat_cs(t: torch.Tensor) -> torch.Tensor:
        key = (t.data_ptr(), tuple(t.shape), t.dtype, str(t.device))
        cached = rep_cache.get(key)
        if cached is not None and cached[0] is t:
            return cached[1]
        if len(rep_cache) >= 8:
            rep_cache.clear()
        out = t.repeat_interleave(2, dim=-1)
        rep_cache[key] = (t, out)
        return out

    rope_mod._repeat_cs = cached_repeat_cs

    # 3. In _FastVecPartialRope.backward, use inplace 64-head vec kernel when grad_output is unaliased contiguous
    opt20_ns = runpy.run_path(str(OPT20_PATH))
    run_inplace_vec = opt20_ns["_run_inplace_vec_rope"]
    fast_cls = rope_mod._V41PartialRope

    @staticmethod
    def fast_bwd(ctx, grad_output):
        cos_rep, sin_rep = ctx.saved_tensors
        g = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
        if (g.shape[0] == 1 and g.shape[2] == 64 and g.shape[3] == 512
                and (not grad_output.is_contiguous() or grad_output._base is None)):
            return run_inplace_vec(g, cos_rep, sin_rep, backward=True), None, None
        return rope_mod._run_partial_rope(g, cos_rep, sin_rep, backward=True), None, None

    fast_cls.backward = fast_bwd
