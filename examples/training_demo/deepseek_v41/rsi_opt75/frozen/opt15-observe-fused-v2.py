from __future__ import annotations

import hashlib
import pathlib
import runpy
from functools import wraps

import torch
import torch_npu
import triton
import triton.language as tl

import hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 as v41_mod
from hyper_parallel.components.modules.engram import EngramModule
import hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel as ep_mod
import hyper_parallel.models.deepseek_v41.adapter.clipped_swiglu as clipped_swiglu_mod
import hyper_parallel.models.deepseek_v41.adapter.aux_loss as aux_loss_mod
import hyper_parallel.models.deepseek_v41.dspark as dspark_mod
import hyper_parallel.components.functional._triton.v41_partial_rope as rope_mod
import hyper_parallel.core.multicore.modules.mega_moe.function as mega_moe_fn_mod
import hyper_parallel.core.optimizer.muon as muon_mod
import hyper_parallel.trainer.base as trainer_base_mod
from hyper_parallel.core.dtensor.dtensor import DTensor, SkipDTensorDispatch

PARENT = pathlib.Path("/xw-tree/full-v41/opt14-observe-fused-swiglu-rmsnorm.py")
PARENT_SHA = "1d12307394358ec9becd41c10df0065456c4e57753951572d9cc36f513fa13fb"


# --- 1. PackedSwiGLU fused Triton backward (4.66x speedup, supports any N) ---
@triton.jit
def _packed_swiglu_bwd_kernel(
    dy_ptr, packed_ptr, dpacked_ptr, M, N: tl.constexpr, limit: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    for row in range(pid, M, num_progs):
        for c_start in range(0, N, BLOCK_N):
            cols = c_start + tl.arange(0, BLOCK_N)
            mask = cols < N
            dy = tl.load(dy_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
            g = tl.load(packed_ptr + row * (2 * N) + cols, mask=mask, other=0.0).to(tl.float32)
            u = tl.load(packed_ptr + row * (2 * N) + N + cols, mask=mask, other=0.0).to(tl.float32)
            g_c = tl.minimum(g, limit)
            u_c = tl.maximum(tl.minimum(u, limit), -limit)
            sig = 1.0 / (1.0 + tl.exp(-g_c))
            silu = g_c * sig
            du = tl.where((u >= -limit) & (u <= limit), dy * silu, 0.0)
            dsilu = sig * (1.0 + g_c * (1.0 - sig))
            dg = tl.where(g <= limit, (dy * u_c) * dsilu, 0.0)
            tl.store(dpacked_ptr + row * (2 * N) + cols, dg.to(tl.bfloat16), mask=mask)
            tl.store(dpacked_ptr + row * (2 * N) + N + cols, du.to(tl.bfloat16), mask=mask)


# --- 2. DSpark _StreamCollapse fused Triton fwd + bwd (5.70x speedup) ---
@triton.jit
def _stream_collapse_fwd_kernel(rows_ptr, w_ptr, out_ptr, R, H: tl.constexpr, BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    for r in range(pid, R, num_progs):
        w0 = tl.load(w_ptr + r * 4 + 0).to(tl.float32)
        w1 = tl.load(w_ptr + r * 4 + 1).to(tl.float32)
        w2 = tl.load(w_ptr + r * 4 + 2).to(tl.float32)
        w3 = tl.load(w_ptr + r * 4 + 3).to(tl.float32)
        for c_start in range(0, H, BLOCK_H):
            cols = c_start + tl.arange(0, BLOCK_H)
            mask = cols < H
            base = r * (4 * H) + cols
            r0 = tl.load(rows_ptr + base + 0 * H, mask=mask, other=0.0).to(tl.float32)
            r1 = tl.load(rows_ptr + base + 1 * H, mask=mask, other=0.0).to(tl.float32)
            r2 = tl.load(rows_ptr + base + 2 * H, mask=mask, other=0.0).to(tl.float32)
            r3 = tl.load(rows_ptr + base + 3 * H, mask=mask, other=0.0).to(tl.float32)
            out = (w0 * r0 + w1 * r1 + w2 * r2 + w3 * r3).to(tl.bfloat16)
            tl.store(out_ptr + r * H + cols, out, mask=mask)


@triton.jit
def _stream_collapse_bwd_kernel(grad_ptr, rows_ptr, w_ptr, grows_ptr, gw_ptr, R, H: tl.constexpr, BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    for r in range(pid, R, num_progs):
        w0 = tl.load(w_ptr + r * 4 + 0).to(tl.float32)
        w1 = tl.load(w_ptr + r * 4 + 1).to(tl.float32)
        w2 = tl.load(w_ptr + r * 4 + 2).to(tl.float32)
        w3 = tl.load(w_ptr + r * 4 + 3).to(tl.float32)
        acc0 = tl.zeros((BLOCK_H,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_H,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_H,), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for c_start in range(0, H, BLOCK_H):
            cols = c_start + tl.arange(0, BLOCK_H)
            mask = cols < H
            dy = tl.load(grad_ptr + r * H + cols, mask=mask, other=0.0).to(tl.float32)
            base = r * (4 * H) + cols
            r0 = tl.load(rows_ptr + base + 0 * H, mask=mask, other=0.0).to(tl.float32)
            r1 = tl.load(rows_ptr + base + 1 * H, mask=mask, other=0.0).to(tl.float32)
            r2 = tl.load(rows_ptr + base + 2 * H, mask=mask, other=0.0).to(tl.float32)
            r3 = tl.load(rows_ptr + base + 3 * H, mask=mask, other=0.0).to(tl.float32)
            tl.store(grows_ptr + base + 0 * H, (dy * w0).to(tl.bfloat16), mask=mask)
            tl.store(grows_ptr + base + 1 * H, (dy * w1).to(tl.bfloat16), mask=mask)
            tl.store(grows_ptr + base + 2 * H, (dy * w2).to(tl.bfloat16), mask=mask)
            tl.store(grows_ptr + base + 3 * H, (dy * w3).to(tl.bfloat16), mask=mask)
            acc0 += dy * r0
            acc1 += dy * r1
            acc2 += dy * r2
            acc3 += dy * r3
        tl.store(gw_ptr + r * 4 + 0, tl.sum(acc0, axis=0))
        tl.store(gw_ptr + r * 4 + 1, tl.sum(acc1, axis=0))
        tl.store(gw_ptr + r * 4 + 2, tl.sum(acc2, axis=0))
        tl.store(gw_ptr + r * 4 + 3, tl.sum(acc3, axis=0))


class _FusedStreamCollapse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_streams: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        streams, hidden = hidden_streams.shape[-2:]
        if (
            hidden_streams.device.type == "npu"
            and hidden_streams.dtype == torch.bfloat16
            and streams == 4
            and hidden == 5120
        ):
            rows = hidden_streams.reshape(-1, streams, hidden).contiguous()
            weights = pre_mix.reshape(-1, streams).float().contiguous()
            output = rows.new_empty(rows.shape[0], hidden)
            if rows.shape[0] > 0:
                _stream_collapse_fwd_kernel[(48,)](rows, weights, output, rows.shape[0], H=5120, BLOCK_H=1024)
            ctx.save_for_backward(hidden_streams, pre_mix)
            return output.view(*hidden_streams.shape[:-2], hidden)
        return dspark_mod._orig_StreamCollapse.forward(ctx, hidden_streams, pre_mix)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden_streams, pre_mix = ctx.saved_tensors
        streams, hidden = hidden_streams.shape[-2:]
        if (
            hidden_streams.device.type == "npu"
            and hidden_streams.dtype == torch.bfloat16
            and streams == 4
            and hidden == 5120
        ):
            rows = hidden_streams.reshape(-1, streams, hidden).contiguous()
            weights = pre_mix.reshape(-1, streams).float().contiguous()
            grad = grad_output.reshape(-1, hidden).contiguous()
            grad_rows = torch.empty_like(rows)
            grad_weights_fp32 = torch.empty_like(weights)
            if rows.shape[0] > 0:
                _stream_collapse_bwd_kernel[(48,)](
                    grad, rows, weights, grad_rows, grad_weights_fp32, rows.shape[0], H=5120, BLOCK_H=1024
                )
            return grad_rows.view_as(hidden_streams), grad_weights_fp32.to(pre_mix.dtype).view_as(pre_mix)
        rows = hidden_streams.reshape(-1, streams, hidden)
        weights = pre_mix.reshape(-1, streams)
        grad = grad_output.reshape(-1, hidden)
        grad_rows, grad_weights = torch.empty_like(rows), torch.empty_like(weights)
        chunk = dspark_mod._MHC_ROW_CHUNK
        for start in range(0, rows.shape[0], chunk):
            stop = start + chunk
            upstream = grad[start:stop].float().unsqueeze(-2)
            grad_rows[start:stop] = upstream * weights[start:stop, :, None]
            grad_weights[start:stop] = (upstream * rows[start:stop].float()).sum(-1)
        return grad_rows.view_as(hidden_streams), grad_weights.view_as(pre_mix)


# --- 3. Engram Gating fused Triton fwd + bwd (8.42x speedup) ---
@triton.jit
def _engram_fwd_k(
    h_ptr, k_ptr, v_ptr, w_ptr, out_ptr,
    rstd_h_ptr, rstd_k_ptr, dot_ptr, gate_ptr,
    R, H: tl.constexpr, D: tl.constexpr, eps: tl.constexpr, clamp_val: tl.constexpr,
    inv_D: tl.constexpr, inv_sqrt_D: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    z = tl.arange(0, 1)
    for idx in range(pid, R * H, num_progs):
        r = idx // H
        h_idx = idx % H
        acc_h2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        acc_k2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        acc_dot = tl.zeros((BLOCK_D,), dtype=tl.float32)
        row_off = idx * D
        w_off = h_idx * D
        for c_start in range(0, D, BLOCK_D):
            cols = c_start + tl.arange(0, BLOCK_D)
            x = tl.load(h_ptr + row_off + cols).to(tl.float32)
            k_val = tl.load(k_ptr + row_off + cols).to(tl.float32)
            w_val = tl.load(w_ptr + w_off + cols).to(tl.float32)
            acc_h2 += x * x
            acc_k2 += k_val * k_val
            acc_dot += x * w_val * k_val
        rstd_h = tl.rsqrt(tl.sum(acc_h2, axis=0) * inv_D + eps)
        rstd_k = tl.rsqrt(tl.sum(acc_k2, axis=0) * inv_D + eps)
        dot = tl.sum(acc_dot, axis=0) * rstd_h * rstd_k * inv_sqrt_D
        abs_dot = tl.abs(dot)
        root = tl.sqrt(tl.maximum(abs_dot, clamp_val))
        signed_root = tl.where(dot < 0.0, -root, root)
        gate = 1.0 / (1.0 + tl.exp(-signed_root))
        tl.store(rstd_h_ptr + idx + z, rstd_h)
        tl.store(rstd_k_ptr + idx + z, rstd_k)
        tl.store(dot_ptr + idx + z, dot)
        tl.store(gate_ptr + idx + z, gate)
        v_off = r * D
        for c_start in range(0, D, BLOCK_D):
            cols = c_start + tl.arange(0, BLOCK_D)
            x = tl.load(h_ptr + row_off + cols).to(tl.float32)
            v_val = tl.load(v_ptr + v_off + cols).to(tl.float32)
            tl.store(out_ptr + row_off + cols, (x + gate * v_val).to(tl.bfloat16))


@triton.jit
def _engram_bwd_k(
    dy_ptr, h_ptr, k_ptr, v_ptr, w_ptr,
    rstd_h_ptr, rstd_k_ptr, dot_ptr, gate_ptr,
    dh_ptr, dk_ptr, dv_ptr, dw_part_ptr,
    R, H: tl.constexpr, D: tl.constexpr, clamp_val: tl.constexpr,
    inv_D: tl.constexpr, inv_sqrt_D: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    z = tl.arange(0, 1)
    for h_idx in range(H):
        for c_start in range(0, D, BLOCK_D):
            cols = c_start + tl.arange(0, BLOCK_D)
            tl.store(dw_part_ptr + (pid * H + h_idx) * D + cols, tl.zeros((BLOCK_D,), dtype=tl.float32))
    for r in range(pid, R, num_progs):
        v_off = r * D
        base_r = (r * H) * D
        g0 = tl.sum(tl.load(gate_ptr + r * H + 0 + z), axis=0)
        g1 = tl.sum(tl.load(gate_ptr + r * H + 1 + z), axis=0)
        g2 = tl.sum(tl.load(gate_ptr + r * H + 2 + z), axis=0)
        g3 = tl.sum(tl.load(gate_ptr + r * H + 3 + z), axis=0)
        for c_start in range(0, D, BLOCK_D):
            cols = c_start + tl.arange(0, BLOCK_D)
            dy0 = tl.load(dy_ptr + base_r + 0 * D + cols).to(tl.float32)
            dy1 = tl.load(dy_ptr + base_r + 1 * D + cols).to(tl.float32)
            dy2 = tl.load(dy_ptr + base_r + 2 * D + cols).to(tl.float32)
            dy3 = tl.load(dy_ptr + base_r + 3 * D + cols).to(tl.float32)
            dv = (dy0 * g0 + dy1 * g1 + dy2 * g2 + dy3 * g3).to(tl.bfloat16)
            tl.store(dv_ptr + v_off + cols, dv)
        for h_idx in range(H):
            idx = r * H + h_idx
            row_off = idx * D
            acc_dg = tl.zeros((BLOCK_D,), dtype=tl.float32)
            for c_start in range(0, D, BLOCK_D):
                cols = c_start + tl.arange(0, BLOCK_D)
                dy_v = tl.load(dy_ptr + row_off + cols).to(tl.float32)
                v_v = tl.load(v_ptr + v_off + cols).to(tl.float32)
                acc_dg += dy_v * v_v
            dgate = tl.sum(acc_dg, axis=0)
            gate = tl.sum(tl.load(gate_ptr + idx + z), axis=0)
            dot = tl.sum(tl.load(dot_ptr + idx + z), axis=0)
            rstd_h = tl.sum(tl.load(rstd_h_ptr + idx + z), axis=0)
            rstd_k = tl.sum(tl.load(rstd_k_ptr + idx + z), axis=0)
            abs_dot = tl.abs(dot)
            root = tl.sqrt(tl.maximum(abs_dot, clamp_val))
            dsigned = dgate * gate * (1.0 - gate)
            ddot = tl.where(abs_dot >= clamp_val, dsigned * 0.5 / root, 0.0)
            scale = ddot * rstd_h * rstd_k * inv_sqrt_D
            coeff_h = (ddot * dot) * (rstd_h * rstd_h) * inv_D
            coeff_k = (ddot * dot) * (rstd_k * rstd_k) * inv_D
            w_off = h_idx * D
            dw_off = (pid * H + h_idx) * D
            for c_start in range(0, D, BLOCK_D):
                cols = c_start + tl.arange(0, BLOCK_D)
                dy_v = tl.load(dy_ptr + row_off + cols).to(tl.float32)
                x = tl.load(h_ptr + row_off + cols).to(tl.float32)
                k_v = tl.load(k_ptr + row_off + cols).to(tl.float32)
                w_v = tl.load(w_ptr + w_off + cols).to(tl.float32)
                dh = dy_v + scale * w_v * k_v - x * coeff_h
                dk = scale * w_v * x - k_v * coeff_k
                tl.store(dh_ptr + row_off + cols, dh.to(tl.bfloat16))
                tl.store(dk_ptr + row_off + cols, dk.to(tl.bfloat16))
                prev_dw = tl.load(dw_part_ptr + dw_off + cols)
                tl.store(dw_part_ptr + dw_off + cols, prev_dw + scale * x * k_v)


class _TritonEngramGatingV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, key, value, q_weight, k_weight, eps: float, clamp_val: float):
        orig_h_shape = hidden_states.shape
        orig_k_shape = key.shape
        orig_v_shape = value.shape
        H, D = orig_h_shape[-2], orig_h_shape[-1]
        R = hidden_states.numel() // (H * D)
        h_c = hidden_states.reshape(R, H, D).contiguous()
        k_c = key.reshape(R, H, D).contiguous()
        v_c = value.reshape(R, D).contiguous()
        w_c = (q_weight.float() * k_weight.float()).contiguous()
        out = torch.empty_like(h_c)
        rstd_h = torch.empty((R * H,), dtype=torch.float32, device=h_c.device)
        rstd_k = torch.empty((R * H,), dtype=torch.float32, device=h_c.device)
        dot = torch.empty((R * H,), dtype=torch.float32, device=h_c.device)
        gate = torch.empty((R * H,), dtype=torch.float32, device=h_c.device)
        if R > 0:
            _engram_fwd_k[(48,)](
                h_c, k_c, v_c, w_c, out, rstd_h, rstd_k, dot, gate,
                R, H=H, D=D, eps=float(eps), clamp_val=float(clamp_val),
                inv_D=1.0 / D, inv_sqrt_D=float(D ** -0.5), BLOCK_D=1024
            )
        ctx.save_for_backward(h_c, k_c, v_c, q_weight, k_weight, w_c, rstd_h, rstd_k, dot, gate)
        ctx.clamp_val = float(clamp_val)
        ctx.shapes = (orig_h_shape, orig_k_shape, orig_v_shape)
        return out.view(orig_h_shape)

    @staticmethod
    def backward(ctx, dy):
        h_c, k_c, v_c, q_weight, k_weight, w_c, rstd_h, rstd_k, dot, gate = ctx.saved_tensors
        orig_h_shape, orig_k_shape, orig_v_shape = ctx.shapes
        R, H, D = h_c.shape
        dy_c = dy.reshape(R, H, D).contiguous()
        dh = torch.empty_like(h_c)
        dk = torch.empty_like(k_c)
        dv = torch.empty_like(v_c)
        num_progs = 48
        dw_part = torch.empty((num_progs, H, D), dtype=torch.float32, device=h_c.device)
        if R > 0:
            _engram_bwd_k[(num_progs,)](
                dy_c, h_c, k_c, v_c, w_c, rstd_h, rstd_k, dot, gate,
                dh, dk, dv, dw_part,
                R, H=H, D=D, clamp_val=ctx.clamp_val,
                inv_D=1.0 / D, inv_sqrt_D=float(D ** -0.5), BLOCK_D=1024
            )
            dw = dw_part.sum(dim=0)
        else:
            dw = torch.zeros((H, D), dtype=torch.float32, device=h_c.device)
        dqw = (dw * k_weight.float()).to(q_weight.dtype)
        dkw = (dw * q_weight.float()).to(k_weight.dtype)
        return dh.view(orig_h_shape), dk.view(orig_k_shape), dv.view(orig_v_shape), dqw, dkw, None, None


# --- 4. In-place tail-only Partial RoPE kernel (skips 87.5% prefix copy!) ---
@triton.jit
def _partial_rope_inplace_kernel(
    tensor_ptr, cosine, sine, tokens: tl.constexpr, backward: tl.constexpr
):
    head = tl.arange(0, 64)
    lane = tl.arange(0, 64)
    odd_lane = lane[None, None, :] % 2 == 1
    partner_index = tl.broadcast_to((lane ^ 1)[None, None, :], (1, 64, 64))
    for tile in range(tl.program_id(0), tokens, tl.num_programs(0)):
        token = tile + tl.arange(0, 1)
        valid = token < tokens
        table_offset = token[:, None] * 64 + lane[None, :]
        cos_value = tl.load(cosine + table_offset, valid[:, None], other=0).to(tl.float32)
        sin_value = tl.load(sine + table_offset, valid[:, None], other=0).to(tl.float32)
        row_base = (token[:, None, None] * 64 + head[None, :, None]) * 512
        rotary_offset = row_base + 448 + lane[None, None, :]
        value = tl.load(tensor_ptr + rotary_offset, valid[:, None, None], other=0).to(tl.float32)
        partner = tl.gather(value, partner_index, axis=2)
        if backward:
            direct = (value * cos_value[:, None, :]).to(tl.bfloat16).to(tl.float32)
            crossed = (partner * sin_value[:, None, :]).to(tl.bfloat16).to(tl.float32)
            rotary = direct + tl.where(odd_lane, -crossed, crossed)
        else:
            rotary = value * cos_value[:, None, :] + tl.where(odd_lane, partner, -partner) * sin_value[:, None, :]
        tl.store(tensor_ptr + rotary_offset, rotary, valid[:, None, None])


def _get_scaled_metadata(metadata, hidden_size: int):
    attr = f"_scaled_offsets_{hidden_size}"
    cached = getattr(metadata, attr, None)
    if cached is None:
        d_src = metadata.dispatch_src_off * hidden_size
        d_tgt = metadata.dispatch_target_off * hidden_size
        d_sz = metadata.dispatch_size * hidden_size
        c_src = metadata.combine_src_off * hidden_size
        c_tgt = metadata.combine_target_off * hidden_size
        c_sz = metadata.combine_size * hidden_size
        cached = (d_src, d_tgt, d_sz, c_src, c_tgt, c_sz)
        object.__setattr__(metadata, attr, cached)
        object.__setattr__(metadata, f"_st0_{hidden_size}", d_src)
        object.__setattr__(metadata, f"_st1_{hidden_size}", d_tgt)
        object.__setattr__(metadata, f"_st2_{hidden_size}", d_sz)
        object.__setattr__(metadata, f"_st3_{hidden_size}", c_src)
        object.__setattr__(metadata, f"_st4_{hidden_size}", c_tgt)
        object.__setattr__(metadata, f"_st5_{hidden_size}", c_sz)
    return cached


def install():
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError("Stage 14 observer changed")
    runpy.run_path(str(PARENT))["install"]()

    # 1. Propagate Stage 14 fast_swiglu to moe_engram_expert_parallel
    fast_swiglu = v41_mod.deepseek_v41_swiglu
    ep_mod.deepseek_v41_swiglu = fast_swiglu

    # 2. Replace _PackedSwiGLU.backward with 1-kernel Triton backward (never unpack saved_tensors twice!)
    @staticmethod
    def fast_packed_bwd(ctx, gradient: torch.Tensor):
        (packed,) = ctx.saved_tensors
        if (
            packed.device.type == "npu"
            and packed.dtype == torch.bfloat16
            and gradient.dtype == torch.bfloat16
            and packed.ndim == 2
            and packed.shape[1] % 2 == 0
            and ctx.limit > 0
        ):
            M, two_N = packed.shape
            N = two_N // 2
            dpacked = torch.empty_like(packed)
            if M > 0:
                _packed_swiglu_bwd_kernel[(48,)](
                    gradient.contiguous(), packed.contiguous(), dpacked, M, N=N, limit=float(ctx.limit), BLOCK_N=1024
                )
            return dpacked, None, None
        derivative = torch.empty_like(packed)
        for start in range(0, packed.shape[0], ctx.chunk_size):
            end = min(start + ctx.chunk_size, packed.shape[0])
            derivative[start:end].copy_(
                clipped_swiglu_mod._backward_chunk(gradient[start:end], packed[start:end], ctx.limit)
            )
        return derivative, None, None

    clipped_swiglu_mod._PackedSwiGLU.backward = fast_packed_bwd

    # 3. Replace DSpark _StreamCollapse with fused Triton kernels (5.70x speedup)
    if not hasattr(dspark_mod, "_orig_StreamCollapse"):
        dspark_mod._orig_StreamCollapse = dspark_mod._StreamCollapse
    dspark_mod._StreamCollapse = _FusedStreamCollapse

    # 4. Replace EngramModule._fuse gating with fused Triton kernel (8.42x speedup)
    orig_engram_fuse = EngramModule._fuse

    @wraps(orig_engram_fuse)
    def fast_engram_fuse(self, hidden_states: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        if (
            hidden_states.device.type == "npu"
            and hidden_states.dtype == torch.bfloat16
            and self.hc_mult == 4
            and self.hidden_size == 5120
        ):
            key_value = self.wkv(embeddings.flatten(start_dim=-2))
            key, value = key_value.split([self.hc_mult * self.hidden_size, self.hidden_size], dim=-1)
            return _TritonEngramGatingV2.apply(
                hidden_states,
                key,
                value,
                self.q_weight,
                self.k_weight,
                float(self.eps),
                float(self.clamp_value),
            )
        return orig_engram_fuse(self, hidden_states, embeddings)

    EngramModule._fuse = fast_engram_fuse

    # 5. Cache balanced fixed-router indices in moe_engram_expert_parallel._resolve_router
    balanced_cache = {}
    orig_resolve_router = ep_mod._resolve_router

    @wraps(orig_resolve_router)
    def fast_resolve_router(router_fn, ep_size: int, fix_router: bool = False):
        if not fix_router:
            return router_fn

        def balanced_router(module, hidden_states: torch.Tensor):
            indices, weights = router_fn(module, hidden_states)
            expert_count = module.gate.num_experts
            key = (tuple(indices.shape), indices.dtype, indices.device, int(expert_count), int(ep_size))
            cached = balanced_cache.get(key)
            if cached is None:
                if expert_count <= 0 or ep_size <= 0 or expert_count % ep_size != 0:
                    raise ValueError("The positive global expert count must be divisible by the positive EP size")
                local_count = expert_count // ep_size
                slots = torch.arange(indices.numel(), device=indices.device, dtype=torch.long)
                balanced = (slots % ep_size) * local_count + (slots // ep_size) % local_count
                cached = balanced.view_as(indices).to(indices.dtype).contiguous()
                balanced_cache[key] = cached
            return cached, weights

        return balanced_router

    ep_mod._resolve_router = fast_resolve_router

    # 6. Skip accumulate_router_load scatter_add during no_grad checkpointed forward
    orig_accum_load = aux_loss_mod.accumulate_router_load

    @wraps(orig_accum_load)
    def fast_accum_load(weights, selected_experts, image_mask, token_mask, accumulator):
        if not torch.is_grad_enabled():
            return weights
        return orig_accum_load(weights, selected_experts, image_mask, token_mask, accumulator)

    aux_loss_mod.accumulate_router_load = fast_accum_load
    v41_mod.accumulate_router_load = fast_accum_load

    # 7. Upgrade _V41PartialRope to use in-place tail rotation when safe (no_grad forward + non-contiguous bwd)
    rope_cache = {}
    orig_run_rope = rope_mod._run_partial_rope

    class _FastInplacePartialRope(torch.autograd.Function):
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
            if not ctx.needs_input_grad[0] and value.is_contiguous():
                _partial_rope_inplace_kernel[(48,)](
                    value, cos_r, sin_r, value.shape[1], False, enable_fp_fusion=False
                )
                return value
            ctx.save_for_backward(cos_r, sin_r)
            return orig_run_rope(value, cos_r, sin_r, backward=False)

        @staticmethod
        def backward(ctx, grad_output):
            cos_r, sin_r = ctx.saved_tensors
            if not grad_output.is_contiguous():
                grad_c = grad_output.contiguous()
                _partial_rope_inplace_kernel[(48,)](
                    grad_c, cos_r, sin_r, grad_c.shape[1], True, enable_fp_fusion=False
                )
                return grad_c, None, None
            return orig_run_rope(grad_output, cos_r, sin_r, backward=True), None, None

    rope_mod._V41PartialRope = _FastInplacePartialRope

    # 8. Cache * spec.hidden_size offsets in mega_moe/function.py, use torch.mm(..., out=output),
    #    and skip dead down_proj.copy_(dispatch) during no_grad forward
    multicore_ops = mega_moe_fn_mod.multicore_ops
    record_plan_stream = mega_moe_fn_mod.record_plan_stream
    _native_source = mega_moe_fn_mod._native_source
    _prepare_forward_execution = mega_moe_fn_mod._prepare_forward_execution
    load_torch_npu = mega_moe_fn_mod.load_torch_npu

    def fast_launch_forward_kernel(plan, metadata, routed_tokens, weight1, weight2, execution) -> None:
        spec = plan.spec
        d_src, d_tgt, d_sz, c_src, c_tgt, c_sz = _get_scaled_metadata(metadata, spec.hidden_size)
        kernel_parameters = (
            execution.dispatch,
            d_tgt,
            _native_source(routed_tokens),
            d_src,
            d_sz,
            weight1,
            metadata.group_list,
            execution.intermediates.up_proj,
            execution.intermediates.activation,
            weight2,
            metadata.group_list,
            execution.intermediates.down_proj,
            execution.combine,
            c_tgt,
            c_src,
            c_sz,
        )
        if not plan.fwd_diagnostics.is_active:
            record_plan_stream(plan, plan.fwd_runtime_config)
            multicore_ops.mega_moe(
                *kernel_parameters,
                execution.gmm_workspace,
                plan.up_proj_tiling,
                plan.swiglu_tiling,
                plan.down_proj_tiling,
                plan.fwd_runtime_config,
                execution.event_counters,
                spec.rank_id,
                spec.ep_size,
                spec.num_experts,
                spec.hidden_size,
                getattr(spec, "plan_tokens", spec.local_num_tokens),
            )
            return
        with plan.fwd_diagnostics.launch(
            kernel_parameters, direction="forward", fallback_event_counters=execution.event_counters
        ) as diagnostic_call:
            record_plan_stream(plan, diagnostic_call.runtime_config)
            multicore_ops.mega_moe_with_diagnostic_buffer(
                *kernel_parameters,
                execution.gmm_workspace,
                plan.up_proj_tiling,
                plan.swiglu_tiling,
                plan.down_proj_tiling,
                diagnostic_call.runtime_config,
                diagnostic_call.event_counters,
                diagnostic_call.diagnostic_buffer,
                spec.rank_id,
                spec.ep_size,
                spec.num_experts,
                spec.hidden_size,
                getattr(spec, "plan_tokens", spec.local_num_tokens),
            )

    def fast_save_forward_state(
        ctx, plan, workspace, saved_dispatch, up_proj, activation, weight1, weight2, metadata, permutation_inputs=()
    ) -> None:
        ctx.plan = plan
        ctx.workspace = workspace
        d_src, d_tgt, d_sz, c_src, c_tgt, c_sz = _get_scaled_metadata(metadata, plan.spec.hidden_size)
        ctx.save_for_backward(
            saved_dispatch,
            up_proj,
            activation,
            weight1,
            weight2,
            metadata.group_list,
            d_src,
            d_tgt,
            d_sz,
            c_src,
            c_tgt,
            c_sz,
            *permutation_inputs,
        )
        workspace.track_graph(ctx)

    def fast_launch_backward_kernel(plan, saved, grad_output, execution) -> None:
        spec = plan.spec
        gradients = execution.intermediates
        kernel_parameters = (
            execution.dispatch,
            saved.dispatch_target_off,
            _native_source(grad_output),
            saved.dispatch_src_off,
            saved.dispatch_size,
            saved.activation,
            gradients.grad_weight2,
            saved.weight2,
            gradients.act_grad,
            saved.up_proj,
            gradients.swiglu_grad,
            saved.weight1,
            gradients.gate_dx,
            execution.grad_x,
            saved.combine_target_off,
            saved.combine_src_off,
            saved.combine_size,
            saved.dispatch,
            gradients.grad_weight1,
            saved.group_list,
        )
        if not plan.bwd_diagnostics.is_active:
            record_plan_stream(plan, plan.bwd_runtime_config)
            multicore_ops.mega_moe_grad(
                *kernel_parameters,
                plan.act_grad_tiling,
                plan.gate_grad_tiling,
                plan.w1_grad_tiling,
                plan.w2_grad_tiling,
                plan.swiglu_grad_tiling,
                execution.gmm_workspace,
                execution.swiglu_workspace,
                plan.bwd_runtime_config,
                execution.event_counters,
                spec.rank_id,
                spec.ep_size,
                spec.num_experts,
                spec.hidden_size,
                getattr(spec, "plan_tokens", spec.local_num_tokens),
            )
            return
        with plan.bwd_diagnostics.launch(
            kernel_parameters, direction="backward", fallback_event_counters=execution.event_counters
        ) as diagnostic_call:
            record_plan_stream(plan, diagnostic_call.runtime_config)
            multicore_ops.mega_moe_grad_with_diagnostic_buffer(
                *kernel_parameters,
                plan.act_grad_tiling,
                plan.gate_grad_tiling,
                plan.w1_grad_tiling,
                plan.w2_grad_tiling,
                plan.swiglu_grad_tiling,
                execution.gmm_workspace,
                execution.swiglu_workspace,
                diagnostic_call.runtime_config,
                diagnostic_call.event_counters,
                diagnostic_call.diagnostic_buffer,
                spec.rank_id,
                spec.ep_size,
                spec.num_experts,
                spec.hidden_size,
                getattr(spec, "plan_tokens", spec.local_num_tokens),
            )

    def fast_write_weight_grad(output: torch.Tensor, inputs: torch.Tensor, gradient: torch.Tensor, output_first: bool) -> None:
        if output.is_contiguous():
            if output_first:
                torch.mm(gradient.T, inputs, out=output)
            else:
                torch.mm(inputs.T, gradient, out=output)
        else:
            if output_first:
                output.copy_(torch.mm(gradient.T, inputs))
            else:
                output.copy_(torch.mm(inputs.T, gradient))

    @staticmethod
    def fast_megamoe_forward(
        ctx, routed_tokens, weight1, weight2, route, plan, workspace, permutation, topk_weights, workspace_claimed
    ):
        spec = plan.spec
        ctx.weight_grad_offsets = None
        if spec.use_matmul_weight_grad:
            if route.expert_end_offsets is None or len(route.expert_end_offsets) != spec.local_experts:
                raise ValueError("Separate weight gradients require exact Host expert end offsets")
            ctx.weight_grad_offsets = route.expert_end_offsets
        if topk_weights is not None and (spec.dispatch_mode != "pull" or permutation is None):
            raise ValueError("Integrated output unpermutation requires pull dispatch and an input permutation.")
        metadata = route
        permutation_inputs = ()
        if permutation is not None:
            routed_tokens, _, unpermute_mapping = permutation
            if ctx.needs_input_grad[0] or topk_weights is not None:
                permutation_inputs = (unpermute_mapping,)
        ctx.source_rows = routed_tokens.shape[0]
        ctx.token_rows = ctx.source_rows // spec.top_k
        ctx.has_permutation = permutation is not None
        ctx.has_unpermute = topk_weights is not None
        if workspace_claimed:
            if not workspace.in_use:
                raise RuntimeError("MegaMoe forward requires the caller-held workspace lease.")
        else:
            if not spec.auto_tokens:
                workspace.ensure(spec, routed_tokens.dtype, routed_tokens.device)
            workspace.claim()
        try:
            capacity = metadata.expert_capacity
            execution = _prepare_forward_execution(workspace, plan, routed_tokens, capacity)
            fast_launch_forward_kernel(plan, metadata, execution.source, weight1, weight2, execution)
            output = execution.combine[:ctx.source_rows].clone()
            if spec.dispatch_mode == "pull":
                saved_dispatch = execution.dispatch
            else:
                if any(ctx.needs_input_grad):
                    execution.intermediates.down_proj.copy_(execution.dispatch[:capacity])
                saved_dispatch = execution.intermediates.down_proj
            if ctx.has_unpermute:
                probs = topk_weights.float().contiguous()
                ctx.topk_dtype = topk_weights.dtype
                permutation_inputs += (output, probs)
                output = (
                    load_torch_npu().npu_moe_token_unpermute(output, unpermute_mapping, probs=probs)
                    if ctx.source_rows
                    else output[:0]
                )
            if any(ctx.needs_input_grad):
                fast_save_forward_state(
                    ctx,
                    plan,
                    workspace,
                    saved_dispatch,
                    execution.intermediates.up_proj,
                    execution.intermediates.activation,
                    weight1,
                    weight2,
                    metadata,
                    permutation_inputs,
                )
            return output
        finally:
            if not workspace_claimed:
                workspace.release()

    mega_moe_fn_mod._launch_forward_kernel = fast_launch_forward_kernel
    mega_moe_fn_mod._save_forward_state = fast_save_forward_state
    mega_moe_fn_mod._launch_backward_kernel = fast_launch_backward_kernel
    mega_moe_fn_mod._write_weight_grad = fast_write_weight_grad
    mega_moe_fn_mod._MegaMoeFunction.forward = fast_megamoe_forward

    # 9. Batch BaseTrainer.loss_normalize grad.div_(denominator) with torch._foreach_div_
    @torch.no_grad()
    def fast_loss_normalize(self, loss: float, loss_dict: dict[str, float], *, denominator: float | None = None) -> float:
        if denominator is None:
            denominator = self._loss_denominator()
        external = trainer_base_mod.get_model_external_state(self.model)
        excluded = {id(param) for param in external.parameters.values()} if external is not None else set()
        denom_f = float(denominator)
        with SkipDTensorDispatch():
            grads_by_group = {}
            for parameter in self.model.parameters():
                if id(parameter) in excluded:
                    continue
                grad = getattr(parameter, "main_grad", None)
                if grad is None:
                    grad = parameter.grad
                if grad is not None:
                    if isinstance(grad, DTensor):
                        grad = grad.to_local()
                    grads_by_group.setdefault((grad.device, grad.dtype), []).append(grad)
            for g_list in grads_by_group.values():
                if g_list:
                    torch._foreach_div_(g_list, denom_f)
        for name in self.loss_names:
            loss_dict[name] /= denominator
        return loss / denominator

    trainer_base_mod.BaseTrainer.loss_normalize = fast_loss_normalize

    # 10. Fuse buf_a.mul_(coeff_b); buf_b.mul_(coeff_c); buf_a.add_(buf_b) in Muon Newton-Schulz
    orig_ns5 = muon_mod.zeropower_via_newtonschulz5

    @wraps(orig_ns5)
    def fast_ns5(
        ns_inputs: torch.Tensor,
        steps: int,
        ns_variant: str = "asym5",
        epsilon: float = 1e-10,
        ns_coefficients=None,
    ) -> torch.Tensor:
        mat_x = ns_inputs
        transposed = ns_inputs.size(-2) > ns_inputs.size(-1)
        if transposed:
            mat_x = mat_x.mT
        mat_x = mat_x / (mat_x.norm(dim=(-2, -1), keepdim=True) + epsilon)
        n_size = mat_x.size(-2)
        buf_a = torch.empty(mat_x.shape[:-2] + (n_size, n_size), dtype=mat_x.dtype, device=mat_x.device)
        buf_b = torch.empty_like(buf_a)
        buf_c = torch.empty(mat_x.shape, dtype=mat_x.dtype, device=mat_x.device)
        if ns_variant == "legacy":
            step_coeffs = [muon_mod._NS_LEGACY_COEFFS] * steps
        elif ns_variant == "asym5":
            step_coeffs = muon_mod._NS_ASYM5_COEFFS[:steps]
        elif ns_variant == "custom":
            if ns_coefficients is None or steps > len(ns_coefficients):
                return orig_ns5(ns_inputs, steps, ns_variant, epsilon, ns_coefficients)
            step_coeffs = tuple(ns_coefficients[:steps])
        else:
            return orig_ns5(ns_inputs, steps, ns_variant, epsilon, ns_coefficients)

        for coeff_a, coeff_b, coeff_c in step_coeffs:
            torch.matmul(mat_x, mat_x.mT, out=buf_a)
            torch.matmul(buf_a, buf_a, out=buf_b)
            buf_a.mul_(coeff_b).add_(buf_b, alpha=coeff_c)
            torch.matmul(buf_a, mat_x, out=buf_c)
            mat_x.mul_(coeff_a).add_(buf_c)

        del buf_a, buf_b, buf_c
        if transposed:
            mat_x = mat_x.mT
        return mat_x

    muon_mod.zeropower_via_newtonschulz5 = fast_ns5
