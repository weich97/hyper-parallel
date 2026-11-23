"""Stage 43 (32-node / 40L, ZERO precision compromise): opt39 + 2D tiled Triton NativeV41RotaryTail (2.6x faster, diff=0.0) + step4 optimizer breakdown."""
from functools import wraps
from pathlib import Path
import runpy
import time

import torch
import triton
import triton.language as tl

import hyper_parallel.components.modules.shared_compressed_dsa_attention as csa_mod
import hyper_parallel.core.optimizer.adamw as adamw_mod
import hyper_parallel.core.optimizer.muon as muon_mod
import hyper_parallel.trainer.base as trainer_base_mod

OPT39_PATH = Path("/xw-tree/full-v41/opt39-tq2-clip-ropebwd.py")


@triton.jit
def _tail_2d_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    S, H: tl.constexpr,
    xs, xh, cs, ss, os, oh,
    BLOCK_S: tl.constexpr, bwd: tl.constexpr,
):
    lane = tl.arange(0, 64)
    s_offs = tl.arange(0, BLOCK_S)
    odd = lane % 2 == 1
    p_idx = tl.broadcast_to((lane ^ 1)[None, :], (BLOCK_S, 64))
    for s0 in range(tl.program_id(0) * BLOCK_S, S, tl.num_programs(0) * BLOCK_S):
        sv = s0 + s_offs
        sm = sv < S
        c = tl.load(cos_ptr + sv[:, None] * cs + lane[None, :], sm[:, None], other=0).to(tl.float32)
        s = tl.load(sin_ptr + sv[:, None] * ss + lane[None, :], sm[:, None], other=0).to(tl.float32)
        for h in range(H):
            x = tl.load(x_ptr + sv[:, None] * xs + h * xh + lane[None, :], sm[:, None], other=0).to(tl.float32)
            p = tl.gather(x, p_idx, axis=1)
            if bwd:
                d = (x * c).to(tl.bfloat16).to(tl.float32)
                cr = (p * s).to(tl.bfloat16).to(tl.float32)
                out = (d + tl.where(odd[None, :], -cr, cr)).to(tl.bfloat16)
            else:
                out = (x * c + tl.where(odd[None, :], p, -p) * s).to(tl.bfloat16)
            tl.store(out_ptr + sv[:, None] * os + h * oh + lane[None, :], out, sm[:, None])


def _run_tail_2d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, sequence_dim: int, backward: bool) -> torch.Tensor:
    out = torch.empty_like(x)
    h_dim = 2 if sequence_dim == 1 else 1
    S = x.shape[sequence_dim]
    H = x.shape[h_dim]
    _tail_2d_kernel[(48,)](
        x, cos, sin, out,
        S, H,
        x.stride(sequence_dim), x.stride(h_dim),
        cos.stride(sequence_dim), sin.stride(sequence_dim),
        out.stride(sequence_dim), out.stride(h_dim),
        BLOCK_S=64, bwd=backward,
        enable_fp_fusion=False,
    )
    return out


def install():
    runpy.run_path(str(OPT39_PATH))["install"]()

    # 1. Replace _NativeV41RotaryTail with 2D tiled Triton kernel
    orig_tail = csa_mod._NativeV41RotaryTail

    class _Triton2DNativeV41RotaryTail(torch.autograd.Function):
        @staticmethod
        def forward(ctx, rotary: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, sequence_dim: int) -> torch.Tensor:
            if rotary.stride(-1) == 1 and cos.stride(-1) == 1 and sin.stride(-1) == 1 and rotary.shape[-1] == 64:
                ctx.save_for_backward(cos, sin)
                ctx.sequence_dim = sequence_dim
                ctx.use_triton = True
                return _run_tail_2d(rotary, cos, sin, sequence_dim, backward=False)
            ctx.use_triton = False
            return orig_tail.forward(ctx, rotary, cos, sin, sequence_dim)

        @staticmethod
        def backward(ctx, grad: torch.Tensor):
            if getattr(ctx, "use_triton", False):
                cos, sin = ctx.saved_tensors
                g = grad if grad.stride(-1) == 1 else grad.contiguous()
                return _run_tail_2d(g, cos, sin, ctx.sequence_dim, backward=True), None, None, None
            return orig_tail.backward(ctx, grad)

    csa_mod._NativeV41RotaryTail = _Triton2DNativeV41RotaryTail

    # 2. Step-4 optimizer breakdown (rank 0 only)
    prof = {"step": 0, "muon": 0.0, "adamw": 0.0}
    orig_fwd_bwd = trainer_base_mod.BaseTrainer.forward_backward_step
    orig_step_opt = trainer_base_mod.BaseTrainer.step_optimizers_and_schedulers
    orig_muon_step = muon_mod.Muon.step
    orig_adamw_step = adamw_mod.AdamW.step

    @wraps(orig_fwd_bwd)
    def count_step(self, *args, **kwargs):
        prof["step"] += 1
        return orig_fwd_bwd(self, *args, **kwargs)

    @wraps(orig_muon_step)
    def timed_muon(self, *args, **kwargs):
        if prof["step"] == 4 and torch.distributed.get_rank() == 0:
            torch.npu.synchronize()
            t0 = time.perf_counter()
            r = orig_muon_step(self, *args, **kwargs)
            torch.npu.synchronize()
            prof["muon"] += time.perf_counter() - t0
            return r
        return orig_muon_step(self, *args, **kwargs)

    @wraps(orig_adamw_step)
    def timed_adamw(self, *args, **kwargs):
        if prof["step"] == 4 and torch.distributed.get_rank() == 0:
            torch.npu.synchronize()
            t0 = time.perf_counter()
            r = orig_adamw_step(self, *args, **kwargs)
            torch.npu.synchronize()
            prof["adamw"] += time.perf_counter() - t0
            return r
        return orig_adamw_step(self, *args, **kwargs)

    @wraps(orig_step_opt)
    def timed_opt_all(self, *args, **kwargs):
        if prof["step"] == 4 and torch.distributed.get_rank() == 0:
            prof["muon"] = 0.0
            prof["adamw"] = 0.0
            torch.npu.synchronize()
            t0 = time.perf_counter()
            r = orig_step_opt(self, *args, **kwargs)
            torch.npu.synchronize()
            tot = time.perf_counter() - t0
            other = tot - prof["muon"] - prof["adamw"]
            print(f"OPT43_STEP4_OPT_BREAKDOWN total={tot:.4f}s muon={prof['muon']:.4f}s adamw={prof['adamw']:.4f}s other={other:.4f}s", flush=True)
            return r
        return orig_step_opt(self, *args, **kwargs)

    trainer_base_mod.BaseTrainer.forward_backward_step = count_step
    trainer_base_mod.BaseTrainer.step_optimizers_and_schedulers = timed_opt_all
    muon_mod.Muon.step = timed_muon
    adamw_mod.AdamW.step = timed_adamw
