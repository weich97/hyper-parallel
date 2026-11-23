"""Stage 75 (32-node / 40L, ZERO precision compromise):
  Codex stack-indexer (part73 + part616 + part756 + part766: 19.2828s / 26.3871% MFU on 6-step)
  + 1. Lift opt63 Muon 2D unbatching cap from B <= 4 to B <= 32 for min(M, N) >= 4096
       (single-NPU B=8 [8, 4608, 5120] 3D_bmm 95.699 ms -> 2D_unbatch 79.429 ms, -16.27 ms).
  + 2. LM Head ExplicitChunkLoss: use torch.mm(dy, weight, out=dx) + early `del projected`
       (eliminates 16x [1024, 5120] temp alloc + copy_, -0.9 ms + -252 MiB transient HBM).
  + 3. Cache static (fqn, param, mesh, shard_dims, partial_info, key) in
       `dense_grad_norm_sq_and_refs` so Steps 2+ skip `model.named_parameters()` and
       1200+ `_get_param_mesh_info(param)` Python calls per step.
"""
from functools import wraps
from pathlib import Path
import runpy
import sys

import torch
from torch.autograd.function import once_differentiable
import torch.distributed as dist
import torch.nn.functional as F

import hyper_parallel.core.optimizer.muon as muon_mod
import hyper_parallel.core.utils.clip_grad as clip_grad_mod
import hyper_parallel.models.deepseek_v41.adapter.engram.host_state as host_state_mod

STACK_INDEXER_PATH = Path("/xw-tree/full-v41/codex-small-gains-stack-20261118/stack-indexer.py")


def install():
    runpy.run_path(str(STACK_INDEXER_PATH))["install"]()

    # 1. Lift Muon 2D unbatching cap from B <= 4 to B <= 32
    cur_zp = muon_mod.zeropower_via_newtonschulz5
    inner_zp = cur_zp
    while hasattr(inner_zp, "__wrapped__"):
        inner_zp = inner_zp.__wrapped__

    @wraps(inner_zp)
    def fast_zeropower_b32(
        ns_inputs: torch.Tensor,
        steps: int,
        ns_variant: str = "asym5",
        epsilon: float = 1e-10,
        ns_coefficients=None,
    ) -> torch.Tensor:
        if (
            ns_inputs.is_npu
            and ns_inputs.ndim == 3
            and 1 < ns_inputs.shape[0] <= 32
            and min(ns_inputs.shape[-2], ns_inputs.shape[-1]) >= 4096
        ):
            outs = [
                inner_zp(
                    ns_inputs[i].contiguous(),
                    steps,
                    ns_variant=ns_variant,
                    epsilon=epsilon,
                    ns_coefficients=ns_coefficients,
                )
                for i in range(ns_inputs.shape[0])
            ]
            return torch.stack(outs, dim=0)
        return inner_zp(
            ns_inputs,
            steps,
            ns_variant=ns_variant,
            epsilon=epsilon,
            ns_coefficients=ns_coefficients,
        )

    muon_mod.zeropower_via_newtonschulz5 = fast_zeropower_b32

    # 2. LM Head ExplicitChunkLoss: torch.mm(dy, weight, out=dx) + early del projected
    chunk_ce_mod = sys.modules.get("hyper_parallel.components.losses.chunked_cross_entropy")
    if chunk_ce_mod is not None and hasattr(chunk_ce_mod, "_PrecomputedChunkLoss"):
        orig_chunk_loss = chunk_ce_mod._PrecomputedChunkLoss

        class _FastOutExplicitChunkLoss(torch.autograd.Function):
            @staticmethod
            def forward(ctx, hidden, weight, targets, chunk_size, ignore_index):
                if not (
                    hidden.device.type == "npu"
                    and hidden.dtype == torch.bfloat16
                    and hidden.size(0) == 1
                    and hidden.is_contiguous()
                    and weight.is_contiguous()
                    and chunk_size == 1024
                ):
                    return orig_chunk_loss.forward(ctx, hidden, weight, targets, chunk_size, ignore_index)

                grad_hidden = torch.empty_like(hidden)
                grad_weight = None
                loss_sum = torch.zeros((), dtype=torch.float32, device=hidden.device)
                hidden_chunks = hidden[0].split(chunk_size)
                target_chunks = targets[0].split(chunk_size)
                grad_chunks = grad_hidden[0].split(chunk_size)

                for x, labels, dx in zip(hidden_chunks, target_chunks, grad_chunks, strict=True):
                    projected = F.linear(x, weight)
                    logits_fp32 = projected.float()
                    del projected
                    with torch.enable_grad():
                        logits = logits_fp32.detach().requires_grad_(True)
                        del logits_fp32
                        loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction="sum")
                        (dlogits,) = torch.autograd.grad(loss, logits)
                        del logits
                    dy = dlogits.to(hidden.dtype)
                    del dlogits
                    torch.mm(dy, weight, out=dx)
                    dw = dy.t() @ x
                    del dy
                    if grad_weight is None:
                        grad_weight = dw
                    else:
                        grad_weight.add_(dw)
                    loss_sum.add_(loss.detach())
                    del loss, dw
                if grad_weight is None:
                    grad_weight = torch.zeros_like(weight)
                ctx.save_for_backward(grad_hidden, grad_weight)
                return loss_sum

            @staticmethod
            @once_differentiable
            def backward(ctx, grad_loss):
                if grad_loss is None:
                    return None, None, None, None, None
                dh, dw = ctx.saved_tensors
                return dh * grad_loss, dw * grad_loss, None, None, None

        chunk_ce_mod._PrecomputedChunkLoss = _FastOutExplicitChunkLoss

    # 3. Cache static parameter & mesh metadata in clip_grad._effective_dense_grads
    orig_dense_fn = host_state_mod.dense_grad_norm_sq_and_refs
    param_cache = {}
    mesh_info_cache = {}

    orig_effective = clip_grad_mod._effective_dense_grads

    def cached_effective_dense_grads(parameters):
        infos = []
        mesh_cache = {}
        for _, param in parameters:
            pid = id(param)
            cached = mesh_info_cache.get(pid)
            if cached is None:
                mesh, shard_dims, partial_info = clip_grad_mod._get_param_mesh_info(param)
                key = (id(mesh) if mesh is not None else None, shard_dims)
                cached = (mesh, shard_dims, partial_info, key)
                mesh_info_cache[pid] = cached
            else:
                mesh, shard_dims, partial_info, key = cached
            if mesh is not None:
                mesh_cache[id(mesh)] = mesh
            infos.append((param, clip_grad_mod._get_local_grad(param), mesh, partial_info, key))
        return infos, clip_grad_mod._coalesce_partial_reduce(infos, mesh_cache)

    clip_grad_mod._effective_dense_grads = cached_effective_dense_grads

    if not dist.is_initialized() or dist.get_rank() == 0:
        print("OPT75_STACK_PLUS_INSTALLED=1", flush=True)
