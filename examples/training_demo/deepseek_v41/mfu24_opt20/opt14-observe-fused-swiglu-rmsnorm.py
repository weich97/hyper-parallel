"""Stage 14: Chain Stage 12 + Fused Triton ClampedSwiGLU (6.08x) + npu_rms_norm on DeepseekV4RMSNorm (5.40x) + RoPE cos/sin repeat_interleave cache."""
from functools import wraps
import hashlib
from pathlib import Path
import runpy

import torch
import torch_npu
import triton
import triton.language as tl

from transformers.models.deepseek_v4 import modeling_deepseek_v4 as hf_v4_mod
from hyper_parallel.components.functional._triton import v41_partial_rope as rope_mod
from hyper_parallel.models.deepseek_v41 import modeling_deepseek_v41 as v41_mod

PARENT = Path('/xw-tree/full-v41/opt12-observe-gc-sync.py')
PARENT_SHA = 'e8055ca504006ef47f65e05678545a55bbaf14ea79cd70a1a13ef13945fe107a'


@triton.jit
def _clamped_swiglu_fwd_kernel(gate_ptr, up_ptr, out_ptr, n_elements, limit: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    step = num_progs * BLOCK
    for base in range(pid * BLOCK, n_elements, step):
        offs = base + tl.arange(0, BLOCK)
        mask = offs < n_elements
        g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g_c = tl.minimum(g, limit)
        u_c = tl.maximum(tl.minimum(u, limit), -limit)
        sig = 1.0 / (1.0 + tl.exp(-g_c))
        silu = g_c * sig
        out = (silu * u_c).to(tl.bfloat16)
        tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def _clamped_swiglu_bwd_kernel(dy_ptr, gate_ptr, up_ptr, dg_ptr, du_ptr, n_elements, limit: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    step = num_progs * BLOCK
    for base in range(pid * BLOCK, n_elements, step):
        offs = base + tl.arange(0, BLOCK)
        mask = offs < n_elements
        dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g_c = tl.minimum(g, limit)
        u_c = tl.maximum(tl.minimum(u, limit), -limit)
        sig = 1.0 / (1.0 + tl.exp(-g_c))
        silu = g_c * sig
        du = tl.where((u >= -limit) & (u <= limit), dy * silu, 0.0)
        dsilu = sig * (1.0 + g_c * (1.0 - sig))
        dg = tl.where(g <= limit, (dy * u_c) * dsilu, 0.0)
        tl.store(dg_ptr + offs, dg.to(tl.bfloat16), mask=mask)
        tl.store(du_ptr + offs, du.to(tl.bfloat16), mask=mask)


class _FusedClampedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up, limit: float):
        gate = gate.contiguous()
        up = up.contiguous()
        ctx.save_for_backward(gate, up)
        ctx.limit = float(limit)
        out = torch.empty_like(gate)
        n = gate.numel()
        _clamped_swiglu_fwd_kernel[(48,)](gate, up, out, n, limit=ctx.limit, BLOCK=4096)
        return out

    @staticmethod
    def backward(ctx, dy):
        gate, up = ctx.saved_tensors
        dy = dy.contiguous()
        dg = torch.empty_like(gate)
        du = torch.empty_like(up)
        n = gate.numel()
        _clamped_swiglu_bwd_kernel[(48,)](dy, gate, up, dg, du, n, limit=ctx.limit, BLOCK=4096)
        return dg, du, None


def install():
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError('Stage 12 observer changed')
    runpy.run_path(str(PARENT))['install']()

    # 1. Replace DeepseekV4RMSNorm.forward with torch_npu.npu_rms_norm (5.40x speedup)
    orig_rms_fwd = hf_v4_mod.DeepseekV4RMSNorm.forward

    @wraps(orig_rms_fwd)
    def fast_rms_fwd(self, hidden_states):
        if hidden_states.device.type == 'npu' and hidden_states.dtype == self.weight.dtype:
            return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=float(self.variance_epsilon))[0]
        return orig_rms_fwd(self, hidden_states)

    hf_v4_mod.DeepseekV4RMSNorm.forward = fast_rms_fwd

    # 2. Replace deepseek_v41_swiglu with Fused Triton ClampedSwiGLU (6.08x speedup)
    orig_swiglu = v41_mod.deepseek_v41_swiglu

    @wraps(orig_swiglu)
    def fast_swiglu(gate, up, limit: float):
        if gate.device.type == 'npu' and gate.dtype == torch.bfloat16 and up.dtype == torch.bfloat16 and limit > 0:
            return _FusedClampedSwiGLU.apply(gate, up, float(limit))
        return orig_swiglu(gate, up, limit)

    v41_mod.deepseek_v41_swiglu = fast_swiglu

    # 3. Cache cos/sin repeat_interleave in _V41PartialRope
    rope_cache = {}
    orig_run_rope = rope_mod._run_partial_rope

    class _CachedPartialRope(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, cos, sin):
            key = (cos.data_ptr(), sin.data_ptr(), cos.shape, cos._version, sin._version)
            cached = rope_cache.get(key)
            if cached is None:
                rope_cache.clear()
                cos_r = cos.repeat_interleave(2, dim=-1)
                sin_r = sin.repeat_interleave(2, dim=-1)
                rope_cache[key] = (cos_r, sin_r)
            else:
                cos_r, sin_r = cached
            ctx.save_for_backward(cos_r, sin_r)
            return orig_run_rope(value, cos_r, sin_r, backward=False)

        @staticmethod
        def backward(ctx, grad_output):
            cos_r, sin_r = ctx.saved_tensors
            return orig_run_rope(grad_output.contiguous(), cos_r, sin_r, backward=True), None, None

    rope_mod._V41PartialRope = _CachedPartialRope
