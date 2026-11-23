"""Stage 61 (32-node / 40L, ZERO precision compromise):
  opt58 stack
  + eliminate 256x .item() + 512x .cpu() D2H syncs in host_sparse_grad.gather_sparse_along_axis (~140ms/step uncovered gap)
  + eliminate redundant reshape(-1).narrow() on 926 params in hsdp_param.apply_reduced_grad (~28ms/step CPU gap).
"""
from functools import wraps
from pathlib import Path
import runpy

import torch
import torch.distributed as dist

import hyper_parallel.core.fully_shard.hsdp_param as hsdp_param_mod
import hyper_parallel.models.deepseek_v41.adapter.distributed.host_sparse_grad as host_sparse_grad_mod

OPT58_PATH = Path("/xw-tree/full-v41/opt58-opt54-stack.py")


def fast_gather_sparse_along_axis(coo_cpu: torch.Tensor, axis_mesh, width: int) -> torch.Tensor:
    """Gather variable COO rows using all_gather_into_tensor and single D2H transfers."""
    if axis_mesh is None or axis_mesh.size() == 1:
        return coo_cpu
    group = axis_mesh.get_group()
    world = axis_mesh.size()
    device = host_sparse_grad_mod._communication_device(axis_mesh)
    coo_cpu = coo_cpu.coalesce()
    nnz = coo_cpu._nnz()
    local_count = torch.tensor([nnz], dtype=torch.long, device=device)
    count_buf = torch.empty(world, dtype=torch.long, device=device)
    dist.all_gather_into_tensor(count_buf, local_count, group=group)
    counts = count_buf.cpu().tolist()
    max_count = max(counts)
    if max_count == 0:
        return coo_cpu
    ids = torch.zeros(max_count, dtype=torch.long, device=device)
    values = torch.zeros((max_count, width), dtype=torch.float32, device=device)
    if nnz:
        ids[:nnz].copy_(coo_cpu.indices()[0].to(device, non_blocking=True))
        values[:nnz].copy_(coo_cpu.values().to(device, non_blocking=True))
    id_buf = torch.empty((world, max_count), dtype=torch.long, device=device)
    val_buf = torch.empty((world, max_count, width), dtype=torch.float32, device=device)
    dist.all_gather_into_tensor(id_buf, ids, group=group)
    dist.all_gather_into_tensor(val_buf, values, group=group)
    if all(c == max_count for c in counts):
        gathered_ids = id_buf.reshape(-1).cpu()
        gathered_values = val_buf.reshape(-1, width).cpu()
    else:
        id_slices = [id_buf[i, :c] for i, c in enumerate(counts) if c > 0]
        val_slices = [val_buf[i, :c] for i, c in enumerate(counts) if c > 0]
        gathered_ids = torch.cat(id_slices, dim=0).cpu()
        gathered_values = torch.cat(val_slices, dim=0).cpu()
    return torch.sparse_coo_tensor(
        gathered_ids.unsqueeze(0), gathered_values, coo_cpu.shape,
    ).coalesce()


def install():

    # 1. Fast gather_sparse_along_axis (eliminates 256x .item() + 512x .cpu())
    host_sparse_grad_mod.gather_sparse_along_axis = fast_gather_sparse_along_axis

    # 2. Fast apply_reduced_grad (eliminates 926x redundant reshape(-1).narrow() on 1D contiguous shards)
    orig_apply_reduced_grad = hsdp_param_mod.HSDPParamV2.apply_reduced_grad

    @wraps(orig_apply_reduced_grad)
    def fast_apply_reduced_grad(self, reduced_grad: torch.Tensor) -> bool:
        if (
            reduced_grad.dim() == 1
            and reduced_grad.is_contiguous()
            and reduced_grad.numel() == self.sharded_size.numel()
        ):
            sharded_grad = (
                self.sharded_param.main_grad
                if self.apply_grad_on_fp32_main_grad and hasattr(self.sharded_param, "main_grad")
                else (None if self.apply_grad_on_fp32_main_grad else self.sharded_param.grad)
            )
            if (
                not self.offload_to_cpu
                and getattr(self.sharded_param, "_hp_staged_main_grad", None) is None
                and sharded_grad is None
            ):
                fp32_main = self.apply_grad_on_fp32_main_grad
                if fp32_main or reduced_grad.dtype == self.orig_dtype:
                    view = self.to_sharded_dtensor(reduced_grad.view(self.sharded_size))
                    if fp32_main:
                        self.sharded_param.main_grad = view
                        self.sharded_param.grad = None
                    else:
                        self.sharded_param.grad = view
                    if self.unsharded_accumulated_grad_data is not None:
                        self.unsharded_accumulated_grad = None
                    elif self.unsharded_param.grad is not None:
                        self.unsharded_param.grad = None
                    return False
        return orig_apply_reduced_grad(self, reduced_grad)

    hsdp_param_mod.HSDPParamV2.apply_reduced_grad = fast_apply_reduced_grad

    if not dist.is_initialized() or dist.get_rank() == 0:
        print("OPT61_GAP_FIXES_INSTALLED=1", flush=True)
