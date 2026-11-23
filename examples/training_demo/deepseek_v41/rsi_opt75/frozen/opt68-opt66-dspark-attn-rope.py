"""Stage 68 (32-node / 40L, ZERO precision compromise):
  opt66 stack (19.9132s / 25.5518% MFU on 6-step; 19.9255s / 25.5360% MFU on 30-step)
  + DSparkDraftAttention optimizations:
    1. Eliminate redundant inner _checkpoint(_attend_range) inside DSparkDraftAttention.forward
       when DSparkStage.forward already checkpoints _pre_ffn (cuts DSpark FlashAttentionScore
       executions from 18 down to 12 per step while keeping DSparkStage prefix checkpoint enabled).
    2. Replace naive dspark._apply_rope + torch.cat (which caused 44 strided ViewCopy = 71.91 ms,
       94 Slice = 20.02 ms, 188 StridedSlice = 9.17 ms, 36 ConcatD = 8.29 ms) with
       _FastTailRopeTritonFn (custom autograd Function + Triton kernel, out_diff=0.0, grad_diff=0.0,
       9.521 ms -> 1.714 ms per Q/context call).
    3. Replace 3D einsum/bmm in DSparkDraftAttention wo_a grouped output projection backward
       with 8x 2D torch.mm (_FastDSparkGroupedLinearFn).
    Combined single-NPU DSparkDraftAttention stage time: 188.567 -> 133.141 ms/stage (-55.43 ms/stage x 3 = -166.3 ms/step).
"""
from pathlib import Path
import runpy

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from hyper_parallel.models.deepseek_v41 import dspark
from hyper_parallel.models.deepseek_v41.dspark import DSparkDraftAttention

OPT66_PATH = Path("/xw-tree/full-v41/opt66-opt65-dspark-chunk512.py")


@triton.jit
def _dspark_rope_2d_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    M,
    G: tl.constexpr,
    INVERSE: tl.constexpr,
):
    pid = tl.program_id(0)
    BLOCK_M: tl.constexpr = 16
    m_base = pid * BLOCK_M
    m_offs = m_base + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M
    d_offs = tl.arange(0, 32)
    cos = tl.load(cos_ptr + m_offs[:, None] * 32 + d_offs[None, :], mask=m_mask[:, None], other=0.0)
    sin = tl.load(sin_ptr + m_offs[:, None] * 32 + d_offs[None, :], mask=m_mask[:, None], other=0.0)
    for g in range(G):
        row_base = (m_offs * G + g)[:, None] * 64
        x0 = tl.load(x_ptr + row_base + d_offs[None, :] * 2, mask=m_mask[:, None], other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + row_base + d_offs[None, :] * 2 + 1, mask=m_mask[:, None], other=0.0).to(tl.float32)
        if INVERSE:
            r0 = x0 * cos + x1 * sin
            r1 = x1 * cos - x0 * sin
        else:
            r0 = x0 * cos - x1 * sin
            r1 = x0 * sin + x1 * cos
        pair = tl.interleave(r0, r1).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row_base + tl.arange(0, 64)[None, :], pair, mask=m_mask[:, None])


def _fast_rope_triton(x_tail: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool) -> torch.Tensor:
    orig_shape = x_tail.shape
    M = cos.numel() // 32
    G = (x_tail.numel() // 64) // M
    x_2d = x_tail.contiguous().view(M, G, 64)
    c_2d = cos.contiguous().view(M, 32)
    s_2d = sin.contiguous().view(M, 32)
    out = torch.empty_like(x_2d)
    grid = (triton.cdiv(M, 16),)
    _dspark_rope_2d_kernel[grid](x_2d, c_2d, s_2d, out, M, G, INVERSE=inverse)
    return out.view(orig_shape)


class _FastTailRopeTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        ctx.save_for_backward(cos, sin)
        ctx.inverse = inverse
        nope = x.shape[-1] - 64
        rot = _fast_rope_triton(x[..., nope:], cos, sin, inverse)
        return torch.cat((x[..., :nope], rot), dim=-1)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        cos, sin = ctx.saved_tensors
        nope = grad_out.shape[-1] - 64
        grad_rot = _fast_rope_triton(grad_out[..., nope:], cos, sin, not ctx.inverse)
        return torch.cat((grad_out[..., :nope], grad_rot), dim=-1), None, None, None


class _FastDSparkGroupedLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grouped: torch.Tensor, wo_a: torch.Tensor) -> torch.Tensor:
        # grouped: [B, S, J, G, D_in], wo_a: [G, R, D_in]
        B, S, J, G, D_in = grouped.shape
        T = B * S * J
        R = wo_a.shape[1]
        x_g = grouped.reshape(T, G, D_in).permute(1, 0, 2).contiguous()
        out_g = torch.bmm(x_g, wo_a.transpose(1, 2))
        ctx.save_for_backward(x_g, wo_a)
        ctx.shapes = (B, S, J, G, D_in, R)
        return out_g.permute(1, 0, 2).contiguous().view(B, S, J, G * R)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_g, wo_a = ctx.saved_tensors
        B, S, J, G, D_in, R = ctx.shapes
        T = B * S * J
        gg = grad_out.reshape(T, G, R).permute(1, 0, 2).contiguous()
        dx_g = torch.bmm(gg, wo_a)
        dx = dx_g.permute(1, 0, 2).contiguous().view(B, S, J, G, D_in)
        dw = torch.empty((G, R, D_in), dtype=grad_out.dtype, device=grad_out.device)
        for g in range(G):
            torch.mm(gg[g].T, x_g[g], out=dw[g])
        return dx, dw


def _fast_dspark_attn_forward(
    self,
    draft_states: torch.Tensor,
    main_states: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    *,
    sequence_boundaries: tuple[int, ...] | None = None,
    anchor_positions: tuple[int, ...] | None = None,
) -> torch.Tensor:
    seq, block = draft_states.shape[1:3]
    main_seq = main_states.shape[1]
    dspark._validate_draft_anchors(main_seq, seq, anchor_positions)
    if rope_cos.shape[1] < main_seq + block or rope_sin.shape != rope_cos.shape:
        raise ValueError("DSpark rotary tables must cover the main context and every future draft position")
    key_value = self.kv_norm(self.wkv(main_states))
    if key_value.is_npu and self.rope_dim == 64:
        key_value = _FastTailRopeTritonFn.apply(
            key_value, rope_cos[:, :main_seq], rope_sin[:, :main_seq], False
        )
    else:
        nope = self.head_dim - self.rope_dim
        key_value = torch.cat(
            (
                key_value[..., :nope],
                dspark._apply_rope(key_value[..., nope:], rope_cos[:, :main_seq], rope_sin[:, :main_seq]),
            ),
            dim=-1,
        )
    document_positions, _ = dspark._document_geometry(main_seq, sequence_boundaries)
    if anchor_positions is not None:
        document_positions = tuple(document_positions[index] for index in anchor_positions)
    chunk = seq if self.attention_chunk_size <= 0 else min(self.attention_chunk_size, seq)
    if chunk == seq:
        return self._attend_range(
            draft_states, key_value, rope_cos, rope_sin, 0, document_positions, anchor_positions
        )
    outputs = []
    for start in range(0, seq, chunk):
        stop = min(start + chunk, seq)
        chunk_anchors = None if anchor_positions is None else anchor_positions[start:stop]
        outputs.append(
            self._attend_range(
                draft_states[:, start:stop],
                key_value,
                rope_cos,
                rope_sin,
                start,
                document_positions[start:stop],
                chunk_anchors,
            )
        )
    return torch.cat(outputs, dim=1)


def _fast_dspark_attend_range(
    self,
    draft_states: torch.Tensor,
    key_value: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    start: int,
    document_positions: tuple[int, ...],
    anchor_positions: tuple[int, ...] | None = None,
) -> torch.Tensor:
    batch, seq, block, _ = draft_states.shape
    rope = self.rope_dim
    local_positions = torch.tensor(document_positions, device=draft_states.device, dtype=torch.long)
    positions = local_positions[:, None] + 1 + torch.arange(block, device=draft_states.device)[None, :]
    draft_cos = rope_cos.index_select(1, positions.flatten()).view(1, seq, block, rope // 2)
    draft_sin = rope_sin.index_select(1, positions.flatten()).view(1, seq, block, rope // 2)
    query = self.wq_b(self.q_norm(self.wq_a(draft_states)))
    query = query.view(batch, seq, block, self.num_heads, self.head_dim)
    use_fast_rope = query.is_npu and rope == 64
    if use_fast_rope:
        query = _FastTailRopeTritonFn.apply(
            query, draft_cos.unsqueeze(-2), draft_sin.unsqueeze(-2), False
        )
    else:
        query_rot = dspark._apply_rope(query[..., -rope:], draft_cos.unsqueeze(-2), draft_sin.unsqueeze(-2))
        query = torch.cat([query[..., :-rope], query_rot], dim=-1)
    draft_kv = self.kv_norm(self.wkv(draft_states))
    if use_fast_rope:
        draft_kv = _FastTailRopeTritonFn.apply(draft_kv, draft_cos, draft_sin, False)
    else:
        draft_kv = torch.cat(
            (draft_kv[..., :-rope], dspark._apply_rope(draft_kv[..., -rope:], draft_cos, draft_sin)),
            dim=-1,
        )
    if self.attention_backend == "sparse_v2":
        context = dspark.dspark_sparse_attention(
            query,
            key_value,
            draft_kv,
            self.attn_sink.to(query.dtype).float(),
            self.window,
            start,
            document_positions,
            anchor_positions,
        )
        if not self._sparse_engaged:
            dspark.logger.info("DSpark native sparse_v2 training attention engaged with valid-prefix selected KV")
            self._sparse_engaged = True
    else:
        combined_kv, valid = self._gather_context(key_value, draft_kv, start, local_positions, anchor_positions)
        if self.attention_backend == "flash_v3":
            context = self._flash_window_context(query, combined_kv, document_positions)
        else:
            context = self._reference_window_context(query, combined_kv, valid)
    if use_fast_rope:
        context = _FastTailRopeTritonFn.apply(
            context, draft_cos.unsqueeze(-2), draft_sin.unsqueeze(-2), True
        )
    else:
        context = torch.cat(
            (
                context[..., :-rope],
                dspark._apply_rope(context[..., -rope:], draft_cos.unsqueeze(-2), draft_sin.unsqueeze(-2), inverse=True),
            ),
            dim=-1,
        )
    grouped = context.reshape(batch, seq, block, self.o_groups, -1)
    wo_a = self.wo_a.weight.view(self.o_groups, self.o_lora_rank, -1).to(context.dtype)
    if context.is_npu:
        reduced_flat = _FastDSparkGroupedLinearFn.apply(grouped, wo_a)
        return self.wo_b(reduced_flat)
    reduced = torch.einsum("bsjgd,grd->bsjgr", grouped, wo_a)
    return self.wo_b(reduced.flatten(3))


def install():
    runpy.run_path(str(OPT66_PATH))["install"]()

    DSparkDraftAttention.forward = _fast_dspark_attn_forward
    DSparkDraftAttention._attend_range = _fast_dspark_attend_range

    if not dist.is_initialized() or dist.get_rank() == 0:
        print("OPT68_DSPARK_ATTN_ROPE_INSTALLED=1", flush=True)
