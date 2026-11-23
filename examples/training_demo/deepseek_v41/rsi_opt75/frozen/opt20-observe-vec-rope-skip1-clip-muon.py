"""Stage 20: Chain Stage 15 + 64-head vectorized inplace RoPE + skip layer 39 checkpoint + layers 36..39 reshard_after_forward=False + foreach clip_grad + 4x Muon batch + step4 breakdown."""
from functools import wraps
import hashlib
from pathlib import Path
import runpy
import time

import torch
import triton
import triton.language as tl

from hyper_parallel.core.activation_memory import api as act_mem_api
from hyper_parallel.core.dtensor.dtensor import DTensor
import hyper_parallel.components.functional._triton.v41_partial_rope as rope_mod
import hyper_parallel.components.modules.shared_compressed_dsa_attention as csa_mod
import hyper_parallel.core.optimizer.muon as muon_mod
import hyper_parallel.core.utils.clip_grad as clip_grad_mod
import hyper_parallel.trainer.base as trainer_base_mod

PARENT = Path("/xw-tree/full-v41/opt15-observe-fused-v2.py")
PARENT_SHA = "15be7eb4e8f27728bfbb536c5f5d15a75c9e2f661f4be70caf3f73517cb8be07"
SKIP_FROM_LAYER = 39
NO_RESHARD_FROM_LAYER = 36


@triton.jit
def _partial_rope_inplace_vec_kernel(
    tensor_ptr, cosine_ptr, sine_ptr, tokens: tl.constexpr, backward: tl.constexpr,
):
    head = tl.arange(0, 64)
    lane = tl.arange(0, 64)
    odd_lane = lane[None, None, :] % 2 == 1
    partner_index = tl.broadcast_to((lane ^ 1)[None, None, :], (1, 64, 64))
    for tile in range(tl.program_id(0), tokens, tl.num_programs(0)):
        token = tile + tl.arange(0, 1)
        valid = token < tokens
        table_offset = token[:, None] * 64 + lane[None, :]
        cos_value = tl.load(cosine_ptr + table_offset, valid[:, None], other=0).to(tl.float32)
        sin_value = tl.load(sine_ptr + table_offset, valid[:, None], other=0).to(tl.float32)
        rotary_offset = (token[:, None, None] * 64 + head[None, :, None]) * 512 + 448 + lane[None, None, :]
        value = tl.load(tensor_ptr + rotary_offset, valid[:, None, None], other=0).to(tl.float32)
        partner = tl.gather(value, partner_index, axis=2)
        if backward:
            direct = (value * cos_value[:, None, :]).to(tl.bfloat16).to(tl.float32)
            crossed = (partner * sin_value[:, None, :]).to(tl.bfloat16).to(tl.float32)
            rotary = direct + tl.where(odd_lane, -crossed, crossed)
        else:
            rotary = value * cos_value[:, None, :] + tl.where(odd_lane, partner, -partner) * sin_value[:, None, :]
        tl.store(tensor_ptr + rotary_offset, rotary.to(tl.bfloat16), valid[:, None, None])


def _run_inplace_vec_rope(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, backward: bool) -> torch.Tensor:
    _partial_rope_inplace_vec_kernel[(48,)](
        value, cos, sin, value.shape[1], backward, enable_fp_fusion=False,
    )
    return value


class _OwnedVecInplacePartialRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        cos_rep = rope_mod._repeat_cs(cos) if hasattr(rope_mod, "_repeat_cs") else cos.repeat_interleave(2, dim=-1)
        sin_rep = rope_mod._repeat_cs(sin) if hasattr(rope_mod, "_repeat_cs") else sin.repeat_interleave(2, dim=-1)
        ctx.save_for_backward(cos_rep, sin_rep)
        buf = value if value.is_contiguous() else value.contiguous()
        return _run_inplace_vec_rope(buf, cos_rep, sin_rep, backward=False)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        cos_rep, sin_rep = ctx.saved_tensors
        g = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
        return _run_inplace_vec_rope(g, cos_rep, sin_rep, backward=True), None, None


def install():
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError("Stage 15 observer changed")
    runpy.run_path(str(PARENT))["install"]()

    # 1. Replace Stage 15 1-head inplace RoPE with 64-head vectorized inplace RoPE
    orig_v41_rope_cls = rope_mod._V41PartialRope

    class _FastVecPartialRope(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, cos, sin):
            cos_rep = cos.repeat_interleave(2, dim=-1)
            sin_rep = sin.repeat_interleave(2, dim=-1)
            ctx.save_for_backward(cos_rep, sin_rep)
            if not ctx.needs_input_grad[0] and value.is_contiguous() and value.shape[0] == 1 and value.shape[2] == 64 and value.shape[3] == 512:
                return _run_inplace_vec_rope(value, cos_rep, sin_rep, backward=False)
            return rope_mod._run_partial_rope(value, cos_rep, sin_rep, backward=False)

        @staticmethod
        def backward(ctx, grad_output):
            cos_rep, sin_rep = ctx.saved_tensors
            if not grad_output.is_contiguous():
                g = grad_output.contiguous()
                if g.shape[0] == 1 and g.shape[2] == 64 and g.shape[3] == 512:
                    return _run_inplace_vec_rope(g, cos_rep, sin_rep, backward=True), None, None
                return rope_mod._run_partial_rope(g, cos_rep, sin_rep, backward=True), None, None
            return rope_mod._run_partial_rope(grad_output, cos_rep, sin_rep, backward=True), None, None

    rope_mod._V41PartialRope = _FastVecPartialRope

    # 2. _expand_query owned 64-head vectorized inplace RoPE
    @wraps(csa_mod.SharedCompressedDSAAttention._expand_query)
    def fast_expand_query(self, query_residual: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        query = self.q_b_proj(query_residual).view(*query_residual.shape[:2], -1, self.head_dim)
        if self.use_fused_partial_rope and csa_mod._supports_fused_partial_rope(query, cos, sin, 2):
            return _OwnedVecInplacePartialRope.apply(query, cos, sin).transpose(1, 2)
        return csa_mod._apply_v41_rope(
            query, cos, sin, unsqueeze_dim=2, use_fused_partial_rope=self.use_fused_partial_rope,
        ).transpose(1, 2)

    csa_mod.SharedCompressedDSAAttention._expand_query = fast_expand_query

    # 3. Skip activation_checkpoint on layer 39
    orig_checkpoint = act_mem_api.checkpoint

    @wraps(orig_checkpoint)
    def selective_checkpoint(function, *args, **kwargs):
        layer_idx = getattr(function, "layer_idx", None)
        if isinstance(layer_idx, int) and layer_idx >= SKIP_FROM_LAYER:
            call_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in (
                    "swap_inputs", "policy_fn", "context_fn", "group_swap",
                    "cpu_pool", "early_stop", "use_reentrant", "preserve_rng_state",
                )
            }
            return function(*args, **call_kwargs)
        return orig_checkpoint(function, *args, **kwargs)

    act_mem_api.checkpoint = selective_checkpoint

    # 4. Foreach norm + foreach mul_ in clip_grad_norm_
    def fast_foreach_p_norms(grads, norm_type: float, device: torch.device) -> torch.Tensor:
        total = torch.tensor(0.0, device=device, dtype=torch.float32)
        grouped = clip_grad_mod._group_tensors_by_device_and_dtype([[g.detach() for g in grads]])
        for (dev, _), ([dev_grads], _) in grouped.items():
            if clip_grad_mod._has_foreach_support(dev_grads, dev):
                try:
                    per_norms = torch._foreach_norm(dev_grads, norm_type)
                except RuntimeError:
                    per_norms = None
                if per_norms is not None:
                    total.add_(torch.stack([n.to(device=device, dtype=torch.float32) ** norm_type for n in per_norms]).sum())
                else:
                    clip_grad_mod._sum_p_norms(dev_grads, norm_type, device, total)
            else:
                clip_grad_mod._sum_p_norms(dev_grads, norm_type, device, total)
        return total

    clip_grad_mod._foreach_p_norms = fast_foreach_p_norms

    orig_clip_grads_with_norm = clip_grad_mod._clip_grads_with_norm_

    @wraps(orig_clip_grads_with_norm)
    def fast_clip_grads_with_norm(all_grads, max_norm: float, total_norm: torch.Tensor, foreach=None) -> None:
        local_grads = [g.to_local() if isinstance(g, DTensor) else g for g in all_grads]
        return orig_clip_grads_with_norm(local_grads, max_norm, total_norm, foreach=True if foreach is None else foreach)

    clip_grad_mod._clip_grads_with_norm_ = fast_clip_grads_with_norm

    orig_clip_grad_norm = clip_grad_mod.clip_grad_norm_

    @wraps(orig_clip_grad_norm)
    def fast_clip_grad_norm(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None, *, parameter_layouts=None):
        return orig_clip_grad_norm(
            parameters, max_norm, norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=True if foreach is None else foreach,
            parameter_layouts=parameter_layouts,
        )

    clip_grad_mod.clip_grad_norm_ = fast_clip_grad_norm
    trainer_base_mod.clip_grad_norm_ = fast_clip_grad_norm

    # 5. Increase Muon memory-safe batch limit 4x (activations are freed during optimizer step)
    orig_split_batches = muon_mod.Muon._split_into_memory_safe_batches

    @wraps(orig_split_batches)
    def larger_split_batches(self, p_list, shard_size: int = 1, min_batch_size: int = 1):
        max_numel_per_batch = (2048 * 1024 * 1024) // shard_size
        min_batch_size = max(1, min_batch_size)
        batches, current_batch, current_count = [], [], 0
        for p in p_list:
            p_count = p.numel()
            if len(current_batch) >= min_batch_size and current_count + p_count > max_numel_per_batch:
                batches.append(current_batch)
                current_batch = [p]
                current_count = p_count
            else:
                current_batch.append(p)
                current_count += p_count
        if current_batch:
            batches.append(current_batch)
        return batches

    muon_mod.Muon._split_into_memory_safe_batches = larger_split_batches

    # 6. Set reshard_after_forward=False on layers 36..39 + clean BaseTrainer step 4 phase breakdown
    orig_fwd_bwd = trainer_base_mod.BaseTrainer.forward_backward_step
    orig_prep_clip = trainer_base_mod.BaseTrainer.prepare_optimizer_step
    orig_opt_step = trainer_base_mod.BaseTrainer.step_optimizers_and_schedulers
    state = {"step": 0, "configured_reshard": False, "t_fwd": 0.0, "t_clip": 0.0}

    @wraps(orig_fwd_bwd)
    def wrapped_fwd_bwd(self, *args, **kwargs):
        if not state["configured_reshard"]:
            state["configured_reshard"] = True
            inner = getattr(self.model, "model", self.model)
            layers = getattr(inner, "layers", None)
            if layers is not None:
                applied = 0
                for idx in range(NO_RESHARD_FROM_LAYER, len(layers)):
                    layer = layers[idx]
                    if hasattr(layer, "set_reshard_after_forward"):
                        layer.set_reshard_after_forward(False, recurse=True)
                        applied += 1
                if torch.distributed.get_rank() == 0:
                    print(f"OPT20_NO_RESHARD_APPLIED={applied}", flush=True)
        state["step"] += 1
        if state["step"] != 4 or torch.distributed.get_rank() != 0:
            return orig_fwd_bwd(self, *args, **kwargs)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        res = orig_fwd_bwd(self, *args, **kwargs)
        torch.npu.synchronize()
        state["t_fwd"] = time.perf_counter() - t0
        return res

    @wraps(orig_prep_clip)
    def wrapped_prep_clip(self, *args, **kwargs):
        if state["step"] != 4 or torch.distributed.get_rank() != 0:
            return orig_prep_clip(self, *args, **kwargs)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        res = orig_prep_clip(self, *args, **kwargs)
        torch.npu.synchronize()
        state["t_clip"] = time.perf_counter() - t0
        return res

    @wraps(orig_opt_step)
    def wrapped_opt_step(self, *args, **kwargs):
        if state["step"] != 4 or torch.distributed.get_rank() != 0:
            return orig_opt_step(self, *args, **kwargs)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        res = orig_opt_step(self, *args, **kwargs)
        torch.npu.synchronize()
        t_opt = time.perf_counter() - t0
        print(
            f"STEP4_PHASE_BREAKDOWN fwd_bwd={state['t_fwd']:.4f}s prepare_clip={state['t_clip']:.4f}s optimizer={t_opt:.4f}s sum={state['t_fwd']+state['t_clip']+t_opt:.4f}s",
            flush=True,
        )
        return res

    trainer_base_mod.BaseTrainer.forward_backward_step = wrapped_fwd_bwd
    trainer_base_mod.BaseTrainer.prepare_optimizer_step = wrapped_prep_clip
    trainer_base_mod.BaseTrainer.step_optimizers_and_schedulers = wrapped_opt_step
