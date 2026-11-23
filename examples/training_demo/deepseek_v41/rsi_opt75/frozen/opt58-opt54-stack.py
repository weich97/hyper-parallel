"""Stage 58 (32-node / 40L, ZERO precision compromise):
  opt54 (opt53 + wq_b wgrad + mhc64)
  + hp-backbone-exp (opt55 mhc_post_backward R1+R2 C++, -63ms)
  + Codex LM head explicit 1024 chunking & first-chunk grad reuse (-66ms, -0.255 GiB)
  + Codex q_b_proj dX layout (~-20ms)
  + Codex EIO expert w1 GEMM operand swap (~-43ms)
  + fast_foreach_p_norms (-7ms)
  + tiled (4, 8) out-of-place _partial_rope_kernel (~-21ms).
"""
from pathlib import Path
import runpy
import sys

import torch
import torch.nn as nn
import triton
import triton.language as tl

import hyper_parallel.components.functional._triton.v41_partial_rope as rope_mod
import importlib
import hyper_parallel.components.losses.chunked_cross_entropy  # noqa: F401
chunk_ce_mod = sys.modules['hyper_parallel.components.losses.chunked_cross_entropy']
import hyper_parallel.core.multicore.modules.mega_moe.function as mega_moe_fn_mod
import hyper_parallel.core.utils.clip_grad as clip_grad_mod

OPT54_PATH = Path("/xw-tree/full-v41/opt54-opt53-wqb-mhc64.py")
CODEX_LMHEAD_DIR = "/xw-tree/full-v41/codex-lmhead-chunk-size-20261118"
CODEX_RECHECK_DIR = "/xw-tree/full-v41/codex-independent-grok-recheck-20261118"


class _FastWqBLinearV2Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return torch.matmul(x, weight.t())

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        go_2d = grad_output.reshape(-1, grad_output.shape[-1])
        x_2d = x.reshape(-1, x.shape[-1])
        grad_x = grad_w = None
        if ctx.needs_input_grad[0]:
            if (
                x.device.type == "npu" and x.dtype == torch.bfloat16
                and weight.dtype == x.dtype and grad_output.dtype == x.dtype
                and weight.shape == (32768, 1280) and x.numel() == 16384 * 1280
                and grad_output.numel() == 16384 * 32768
                and grad_output.is_contiguous() and weight.is_contiguous()
            ):
                grad_x = (weight.T.contiguous() @ go_2d.T).T.contiguous().view_as(x)
            else:
                grad_x = torch.matmul(go_2d, weight).reshape_as(x)
        if ctx.needs_input_grad[1]:
            grad_w = torch.matmul(x_2d.t().contiguous(), go_2d).t()
        return grad_x, grad_w


@triton.jit
def _fast_out_partial_rope_kernel(
    source, destination, cosine, sine,
    tokens: tl.constexpr, backward: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    num_h_blocks = 64 // BLOCK_H
    num_t_blocks = tl.cdiv(tokens, BLOCK_T)
    total = num_t_blocks * num_h_blocks
    head = tl.arange(0, BLOCK_H)
    lane = tl.arange(0, 64)
    odd_lane = lane[None, None, :] % 2 == 1
    prefix = tl.arange(0, 512)
    partner_index = tl.broadcast_to((lane ^ 1)[None, None, :], (BLOCK_T, BLOCK_H, 64))
    for blk in range(pid, total, tl.num_programs(0)):
        t_idx = blk // num_h_blocks
        h_idx = blk % num_h_blocks
        token = t_idx * BLOCK_T + tl.arange(0, BLOCK_T)
        valid = token < tokens
        table_offset = token[:, None] * 64 + lane[None, :]
        cos_value = tl.load(cosine + table_offset, valid[:, None], other=0).to(tl.float32)
        sin_value = tl.load(sine + table_offset, valid[:, None], other=0).to(tl.float32)
        hh = h_idx * BLOCK_H + head
        row_base = (token[:, None, None] * 64 + hh[None, :, None]) * 512
        rotary_offset = row_base + 448 + lane[None, None, :]
        value = tl.load(source + rotary_offset, valid[:, None, None], other=0).to(tl.float32)
        partner = tl.gather(value, partner_index, axis=2)
        if backward:
            direct = (value * cos_value[:, None, :]).to(tl.bfloat16).to(tl.float32)
            crossed = (partner * sin_value[:, None, :]).to(tl.bfloat16).to(tl.float32)
            rotary = direct + tl.where(odd_lane, -crossed, crossed)
        else:
            rotary = value * cos_value[:, None, :] + tl.where(odd_lane, partner, -partner) * sin_value[:, None, :]
        tl.store(destination + rotary_offset, rotary.to(tl.bfloat16), valid[:, None, None])
        prefix_offset = row_base + prefix[None, None, :]
        prefix_valid = valid[:, None, None] & (prefix[None, None, :] < 448)
        prefix_value = tl.load(source + prefix_offset, prefix_valid, other=0)
        tl.store(destination + prefix_offset, prefix_value, prefix_valid)


def install():
    runpy.run_path(str(OPT54_PATH))["install"]()

    # 1. Upgrade q_b_proj linear hook with Codex dX layout
    orig_opt54_linear_forward = nn.Linear.forward

    def fast_wqb_linear_v2(self, input):
        if (
            self.bias is None
            and input.dtype == torch.bfloat16
            and input.shape[-1] == 1280
            and input.numel() == 16384 * 1280
            and self.weight.shape == (32768, 1280)
            and type(self.weight) is torch.Tensor
            and torch.is_grad_enabled()
        ):
            return _FastWqBLinearV2Fn.apply(input, self.weight)
        return orig_opt54_linear_forward(self, input)

    nn.Linear.forward = fast_wqb_linear_v2

    # 2. Codex LM head explicit 1024 chunking + first-chunk grad reuse
    if CODEX_LMHEAD_DIR not in sys.path:
        sys.path.insert(0, CODEX_LMHEAD_DIR)
    import candidate as lmhead_candidate  # pylint: disable=import-outside-toplevel
    lmhead_candidate.install(chunk_ce_mod, skip_empty=False)

    # 3. Codex EIO expert w1 GEMM operand swap
    if CODEX_RECHECK_DIR not in sys.path:
        sys.path.insert(0, CODEX_RECHECK_DIR)
    import wgrad_candidate  # pylint: disable=import-outside-toplevel
    wgrad_candidate.install(mega_moe_fn_mod)

    # 4. Vectorized fast_foreach_p_norms
    def fast_foreach_p_norms(grads, norm_type: float, device: torch.device) -> torch.Tensor:
        total = torch.zeros((), device=device, dtype=torch.float32)
        grouped = clip_grad_mod._group_tensors_by_device_and_dtype([[g.detach() for g in grads]])
        for (dev, _), ([dev_grads], _) in grouped.items():
            if clip_grad_mod._has_foreach_support(dev_grads, dev):
                try:
                    per_norms = torch._foreach_norm(dev_grads, norm_type)
                except RuntimeError:
                    per_norms = None
                if per_norms is not None:
                    if norm_type == 2.0:
                        bf16_n = [n for n in per_norms if n.dtype == torch.bfloat16]
                        fp32_n = [n for n in per_norms if n.dtype == torch.float32]
                        other_n = [n for n in per_norms if n.dtype not in (torch.bfloat16, torch.float32)]
                        if bf16_n:
                            s = torch.stack(bf16_n).to(device=device, dtype=torch.float32)
                            total.add_((s * s).sum())
                        if fp32_n:
                            s = torch.stack(fp32_n).to(device=device)
                            total.add_((s * s).sum())
                        if other_n:
                            s = torch.stack([n.to(device=device, dtype=torch.float32) for n in other_n])
                            total.add_((s * s).sum())
                    else:
                        s = torch.stack([n.to(device=device, dtype=torch.float32) ** norm_type for n in per_norms])
                        total.add_(s.sum())
                else:
                    clip_grad_mod._sum_p_norms(dev_grads, norm_type, device, total)
            else:
                clip_grad_mod._sum_p_norms(dev_grads, norm_type, device, total)
        return total

    clip_grad_mod._foreach_p_norms = fast_foreach_p_norms

    # 5. Tiled (4, 8) out-of-place _partial_rope_kernel for [1, S, 64, 512]
    orig_run_partial_rope = rope_mod._run_partial_rope

    def fast_run_partial_rope(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, backward: bool) -> torch.Tensor:
        if value.is_contiguous() and value.dim() == 4 and value.shape[0] == 1 and value.shape[2] == 64 and value.shape[3] == 512:
            output = torch.empty_like(value)
            _fast_out_partial_rope_kernel[(48,)](
                value, output, cos, sin, value.shape[1], backward,
                BLOCK_T=4, BLOCK_H=8, enable_fp_fusion=False,
            )
            return output
        return orig_run_partial_rope(value, cos, sin, backward=backward)

    rope_mod._run_partial_rope = fast_run_partial_rope

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print("OPT58_STACK_INSTALLED=1", flush=True)
