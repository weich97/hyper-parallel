"""Stage 26 (32-node / 40L): Stage 20 + Triton prefix & compressed teacher softmax-sum + chunk_size=1024."""
from functools import wraps
from pathlib import Path
import runpy

import torch
import triton
import triton.language as tl

import hyper_parallel.components.functional.sparse_lightning_indexer_kl_loss_grad as kl_mod

OPT20_PATH = Path("/xw-tree/full-v41/opt20-observe-vec-rope-skip1-clip-muon.py")


@triton.jit
def _sink_softmax_sum_kernel(
    scores_ptr, sinks_ptr, counts_ptr, out_ptr,
    H: tl.constexpr, Q, K_SRC: tl.constexpr, K_PAD: tl.constexpr,
    scale,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    ks = tl.arange(0, K_PAD)
    for q_idx in range(pid, Q, num_progs):
        cnt = tl.load(counts_ptr + q_idx)
        k_mask = ks < cnt
        acc = tl.zeros((K_PAD,), dtype=tl.float32)
        for h in range(H):
            s_val = tl.load(sinks_ptr + h)
            row_ptr = scores_ptr + (h * Q + q_idx) * K_SRC + ks
            sc = tl.load(row_ptr, mask=k_mask, other=-1.0e9) * scale
            sc = tl.where(k_mask, sc, -1.0e9)
            m = tl.maximum(tl.max(sc, axis=0), s_val)
            e = tl.exp(sc - m)
            e = tl.where(k_mask, e, 0.0)
            denom = tl.sum(e, axis=0) + tl.exp(s_val - m)
            acc += e / denom
        tl.store(out_ptr + q_idx * K_SRC + ks, acc, mask=k_mask)


@triton.jit
def _compressed_sink_softmax_sum_2d_kernel(
    scores_ptr, sinks_ptr, indices_ptr, out_ptr,
    H: tl.constexpr, Q, K: tl.constexpr,
    scale, normalize: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    hs = tl.arange(0, 16)
    ks = tl.arange(0, K)
    for q_idx in range(pid, Q, num_progs):
        v_mask = (tl.load(indices_ptr + q_idx * K + ks) >= 0)[None, :]
        acc = tl.zeros((K,), dtype=tl.float32)
        for h0 in range(0, H, 16):
            h_idx = h0 + hs
            h_mask = (h_idx < H)[:, None]
            s_vec = tl.load(sinks_ptr + h_idx, mask=h_idx < H, other=-1.0e9)[:, None]
            sc = tl.load(
                scores_ptr + q_idx * H * K + h_idx[:, None] * K + ks[None, :],
                mask=h_mask,
                other=-1.0e9,
            ) * scale
            sc = tl.where(v_mask & h_mask, sc, -1.0e9)
            m = tl.maximum(tl.max(sc, axis=1)[:, None], s_vec)
            e = tl.exp(sc - m)
            e = tl.where(v_mask & h_mask, e, 0.0)
            denom = tl.sum(e, axis=1)[:, None] + tl.exp(s_vec - m)
            acc += tl.sum(tl.where(h_mask, e / denom, 0.0), axis=0)
        if normalize:
            acc_sum = tl.sum(acc, axis=0)
            acc = tl.where(acc_sum > 0.0, acc / acc_sum, 0.0)
        tl.store(out_ptr + q_idx * K + ks, acc)


def install():
    runpy.run_path(str(OPT20_PATH))["install"]()

    orig_compressed = kl_mod.compressed_attention_teacher

    @wraps(orig_compressed)
    def fast_compressed_attention_teacher(query, key, indices, sinks, scale, reduce_sum=None):
        width = indices.shape[-1]
        if width == 0 or (width & (width - 1)) != 0 or width > 512:
            return orig_compressed(query, key, indices, sinks, scale, reduce_sum=reduce_sum)
        with torch.no_grad(), torch.autocast(device_type=query.device.type, enabled=False):
            B, H, Q, D = query.shape
            selected = kl_mod.gather_selected_keys_fp32(key, indices.clamp_min(0).long())
            scores = torch.matmul(query.float().transpose(1, 2), selected.transpose(-1, -2)).contiguous()
            del selected
            out = torch.empty((B, Q, width), device=query.device, dtype=torch.float32)
            sinks_f = sinks.float().contiguous()
            idx_c = indices.contiguous()
            do_norm = reduce_sum is None
            for b in range(B):
                _compressed_sink_softmax_sum_2d_kernel[(48,)](
                    scores[b], sinks_f, idx_c[b], out[b],
                    H, Q, width, float(scale), do_norm,
                )
            del scores
            if reduce_sum is not None:
                out = reduce_sum(out)
                out = out / out.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
            return out

    kl_mod.compressed_attention_teacher = fast_compressed_attention_teacher

    orig_teacher = kl_mod.causal_attention_teacher

    @wraps(orig_teacher)
    def fast_causal_attention_teacher(query, key, indices, sinks, scale, ratio, residual, chunk_size):
        chunk_size = max(chunk_size, 1024)
        with torch.no_grad(), torch.autocast(device_type=query.device.type, enabled=False):
            batch, heads, length, _ = query.shape
            width = indices.shape[-1]
            offset = key.shape[1] * ratio + residual - length
            partial = min(length, max(0, ratio * width - 1 - offset))
            target = query.new_zeros(indices.shape, dtype=torch.float32)
            max_keys = min(key.shape[1], (offset + partial) // ratio) if partial else 0
            if max_keys and batch == 1 and max_keys <= 4096:
                prefix = key[:, :max_keys].float()
                counts = ((torch.arange(partial, device=query.device) + offset + 1) // ratio).clamp(max=max_keys).to(torch.int32)
                sinks_f = sinks.float().contiguous()
                k_pad = triton.next_power_of_2(max_keys)
                for start in range(0, partial, chunk_size):
                    end = min(start + chunk_size, partial)
                    q_len = end - start
                    scores = torch.einsum("bhqd,bkd->bhqk", query[:, :, start:end].float(), prefix).contiguous()
                    out_chunk = target[0, start:end, :max_keys]
                    _sink_softmax_sum_kernel[(48,)](
                        scores, sinks_f, counts[start:end].contiguous(), out_chunk,
                        heads, q_len, max_keys, k_pad, float(scale),
                    )
                    del scores
                del prefix
                prefix_target = target[:, :partial]
                prefix_target.div_(prefix_target.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny))
            elif max_keys:
                return orig_teacher(query, key, indices, sinks, scale, ratio, residual, chunk_size)

            bank = kl_mod._teacher_bank(key, batch * (length - partial) * width)
            for start in range(partial, length, chunk_size):
                end = min(start + chunk_size, length)
                target[:, start:end] = fast_compressed_attention_teacher(
                    query[:, :, start:end], bank, indices[:, start:end], sinks, scale,
                )
            return target

    kl_mod.causal_attention_teacher = fast_causal_attention_teacher
