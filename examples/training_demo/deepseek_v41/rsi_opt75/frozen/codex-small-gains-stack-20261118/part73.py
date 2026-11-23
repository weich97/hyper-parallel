"""Stage 73 (32-node / 40L, ZERO precision compromise):
  opt71 stack (19.5105s / 26.0791% MFU, peak 19.4912s / 26.1050% MFU, max_alloc=50.0677 GB)
  + Vectorize HostEngramState.prepare_optimizer_step dense gradient norm & scaling
    (prepare_clip = 315.2 ms/step in opt71):
    1. Replace per-parameter Python loop `view.float().square().sum().to(device)` (~1000+ params,
       >4000 kernel launches) in `dense_grad_norm_sq_and_refs` with `torch._foreach_norm(..., 2.0)`.
    2. Replace per-tensor Python loop `grad.mul_(coefficient)` in `scale_grad_refs_` with
       grouped `torch._foreach_mul_` (single-NPU 51.06 ms -> 18.85 ms).
    3. Replace 256-rank `dist.all_gather_object` + Python dict pickling of >1000 parameter names
       in `_check_dense_grad_presence` with a 1-int `ReduceOp.MIN` fast-path (falling back to
       `orig_check_dense_grad_presence` only if any gradient is None).
"""
from functools import wraps
from pathlib import Path
import runpy

import torch
import torch.distributed as dist

import hyper_parallel.core.utils.clip_grad as clip_grad_mod
import hyper_parallel.models.deepseek_v41.adapter.engram.host_state as host_state_mod

OPT71_PATH = Path("/xw-tree/full-v41/opt71-opt70-dspark-ce-step4async.py")


def install():
    runpy.run_path(str(OPT71_PATH))["install"]()

    orig_check_presence = clip_grad_mod._check_dense_grad_presence
    reported = False

    def fast_check_dense_grad_presence(present, replica_ranks):
        if not dist.is_initialized():
            return
        dev = torch.device("npu", torch.npu.current_device()) if torch.npu.is_available() else torch.device("cpu")
        all_ok = torch.tensor(1 if all(present.values()) else 0, dtype=torch.int32, device=dev)
        dist.all_reduce(all_ok, op=dist.ReduceOp.MIN)
        if int(all_ok.item()) == 1:
            return
        orig_check_presence(present, replica_ranks)

    def fast_dense_grad_norm_sq_and_refs(model, excluded_params, norm_shard_plan, mesh_context):
        nonlocal reported
        excluded_ids = {id(param) for param in excluded_params}
        parameters = [
            (fqn, param) for fqn, param in model.named_parameters()
            if param.requires_grad and id(param) not in excluded_ids
        ]
        device = next((clip_grad_mod._param_device(param) for _, param in parameters), None)
        if device is None:
            mesh = getattr(mesh_context, "device_mesh", None)
            device = torch.device(getattr(mesh, "device_type", "cpu"))
        infos, reduced = clip_grad_mod._effective_dense_grads(parameters)
        refs = []
        norm_views = []
        present = {}
        for index, (fqn, _) in enumerate(parameters):
            grad = infos[index][1]
            present[fqn] = grad is not None
            if grad is None:
                continue
            refs.append(grad)
            if norm_shard_plan.representatives[fqn]:
                norm_views.append(reduced.get(index, grad))

        local_sq = torch.zeros((), device=device, dtype=torch.float32)
        if norm_views:
            grouped = clip_grad_mod._group_tensors_by_device_and_dtype([[v.detach() for v in norm_views]])
            for (dev, _), ([dev_views], _) in grouped.items():
                if clip_grad_mod._has_foreach_support(dev_views, dev):
                    try:
                        per_norms = (
                            torch._foreach_norm(dev_views, 2.0, dtype=torch.float32)
                            if dev_views[0].dtype == torch.bfloat16
                            else torch._foreach_norm(dev_views, 2.0)
                        )
                    except RuntimeError:
                        per_norms = None
                    if per_norms is not None:
                        bf16_n = [n for n in per_norms if n.dtype == torch.bfloat16]
                        fp32_n = [n for n in per_norms if n.dtype == torch.float32]
                        other_n = [n for n in per_norms if n.dtype not in (torch.bfloat16, torch.float32)]
                        if bf16_n:
                            s = torch.stack(bf16_n).to(device=device, dtype=torch.float32)
                            local_sq.add_((s * s).sum())
                        if fp32_n:
                            s = torch.stack(fp32_n).to(device=device)
                            local_sq.add_((s * s).sum())
                        if other_n:
                            s = torch.stack([n.to(device=device, dtype=torch.float32) for n in other_n])
                            local_sq.add_((s * s).sum())
                        continue
                for v in dev_views:
                    local_sq.add_(v.float().square().sum().to(device))

        if not reported:
            report = {}
            for view in norm_views:
                key = str(view.dtype)
                count, elements = report.get(key, (0, 0))
                report[key] = (count + 1, elements + view.numel())
            print(f"OPT73_FP32_CLIP_LOCAL_VIEWS rank={dist.get_rank() if dist.is_initialized() else 0} {report}", flush=True)
            reported = True
        fast_check_dense_grad_presence(present, norm_shard_plan.replica_ranks)
        for name in getattr(getattr(mesh_context, "device_mesh", None), "mesh_dim_names", ()) or ():
            axis = mesh_context.device_mesh[name]
            if axis.size() > 1:
                dist.all_reduce(local_sq, op=dist.ReduceOp.SUM, group=axis.get_group())
        return local_sq, refs

    def fast_scale_grad_refs_(refs, coefficient):
        ref_list = list(refs)
        if not ref_list:
            return
        grouped = clip_grad_mod._group_tensors_by_device_and_dtype([ref_list])
        for (dev, dt), ([dev_grads], _) in grouped.items():
            coef = coefficient.to(device=dev, dtype=dt)
            if clip_grad_mod._has_foreach_support(dev_grads, dev):
                try:
                    torch._foreach_mul_(dev_grads, coef)
                    continue
                except RuntimeError:
                    pass
            for g in dev_grads:
                g.mul_(coef)

    clip_grad_mod._check_dense_grad_presence = fast_check_dense_grad_presence
    clip_grad_mod.dense_grad_norm_sq_and_refs = fast_dense_grad_norm_sq_and_refs
    clip_grad_mod.scale_grad_refs_ = fast_scale_grad_refs_
    host_state_mod.dense_grad_norm_sq_and_refs = fast_dense_grad_norm_sq_and_refs
    combined = host_state_mod.DeepseekV41HostState._combined_gradient_norm
    scope = combined.__globals__
    if "dense_grad_norm_sq_and_refs" not in scope or "dense_grad_norm_sq_and_refs" not in combined.__code__.co_names:
        raise RuntimeError("Combined norm observer no longer exposes the dense norm binding")
    scope["dense_grad_norm_sq_and_refs"] = fast_dense_grad_norm_sq_and_refs
    host_state_mod.scale_grad_refs_ = fast_scale_grad_refs_

    if not dist.is_initialized() or dist.get_rank() == 0:
        print("OPT73_FP32_CLIP_NORMS_INSTALLED=1", flush=True)
