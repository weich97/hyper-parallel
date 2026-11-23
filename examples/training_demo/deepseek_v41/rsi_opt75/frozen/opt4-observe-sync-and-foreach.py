"""Stage 4: Chain Stage 2 (EDP balance + Vision comm_fusion) + foreach clip_grad scaling + skip per-step Engram bounds all_reduce."""
from collections import defaultdict
from functools import wraps
import hashlib
import os
from pathlib import Path
import runpy

import torch
import torch.distributed as dist

from hyper_parallel.core.utils import clip_grad as clip_grad_mod
from hyper_parallel.models.deepseek_v41.adapter.engram import host_lookup as host_lookup_mod
from hyper_parallel.models.deepseek_v41.adapter.engram.host_lookup import HostRowsFn, ep_all_to_all, stage_check

PARENT = Path('/xw-tree/full-v41/opt2-observe-vision-comm-fusion.py')
PARENT_SHA = '6036ed4e8c149833de564f5c40e3e41ee14a51934efdcbe769fc8baa895a3eb3'


def install():
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError('Stage 2 observer changed')
    runpy.run_path(str(PARENT))['install']()

    # 1. Replace per-tensor grad.mul_(0D_tensor) loop (1,262 .item() syncs) with single .item() + _foreach_mul_
    @wraps(clip_grad_mod.scale_grad_refs_)
    def fast_scale_grad_refs_(refs, coefficient):
        ref_list = list(refs)
        if not ref_list:
            return
        coef = float(coefficient.item()) if isinstance(coefficient, torch.Tensor) else float(coefficient)
        if coef == 1.0:
            return
        by_group = defaultdict(list)
        for g in ref_list:
            by_group[(g.device, g.dtype)].append(g)
        for (_, _), group_grads in by_group.items():
            torch._foreach_mul_(group_grads, coef)

    clip_grad_mod.scale_grad_refs_ = fast_scale_grad_refs_
    # Also update host_state's imported reference if it imported scale_grad_refs_ directly
    from hyper_parallel.models.deepseek_v41.adapter.engram import host_state as host_state_mod
    if hasattr(host_state_mod, 'scale_grad_refs_'):
        host_state_mod.scale_grad_refs_ = fast_scale_grad_refs_

    # 2. Validate Engram bounds on first few calls only, avoiding 8x 256-rank all_reduce + 16 D2H syncs/step
    lookup_calls = [0]

    @wraps(host_lookup_mod.host_lookup)
    def fast_host_lookup(table, global_ids):
        if table.rows_per_owner is None or table.global_row_start is None:
            raise RuntimeError('Host Engram table is not bound')
        if table.weight.device.type != 'cpu' or table.weight.dtype != torch.float32:
            raise RuntimeError('Host Engram weight must remain CPU FP32')
        lookup_calls[0] += 1
        do_check = lookup_calls[0] <= 4
        flat_ids = global_ids.reshape(-1).long()
        unique_ids, inverse = torch.unique(flat_ids, sorted=True, return_inverse=True)
        table.lookup_requests += flat_ids.numel()
        table.lookup_unique_ids += unique_ids.numel()
        if do_check:
            invalid_ids = bool(unique_ids.numel() and
                               (unique_ids.min() < 0 or unique_ids.max() >= table.logical_rows))
            if stage_check(invalid_ids, table.mesh_context):
                raise IndexError('Engram hash requested a row outside its logical table')

        if table.ep_size == 1:
            local_ids = unique_ids.to('cpu')
            owner_values = HostRowsFn.apply(table.weight, local_ids, table)
            transfer_values = owner_values.to(table.lookup_compute_dtype)
            table.host_to_device_bytes += transfer_values.numel() * transfer_values.element_size()
            unique_values = transfer_values.to(global_ids.device)
        else:
            owner = torch.div(unique_ids, table.rows_per_owner, rounding_mode='floor')
            send_counts_tensor = torch.bincount(owner, minlength=table.ep_size).to(torch.long)
            count_parts = [torch.empty_like(send_counts_tensor) for _ in range(table.ep_size)]
            dist.all_gather(count_parts, send_counts_tensor, group=table.ep_group)
            send_counts = send_counts_tensor.cpu().tolist()
            recv_counts = torch.stack(count_parts)[:, table.ep_rank].cpu().tolist()
            owned_global_ids = ep_all_to_all(unique_ids, send_counts, recv_counts, table.ep_group)
            owned_local_ids = (owned_global_ids - table.global_row_start).to('cpu')
            if do_check:
                invalid_owner = bool(owned_local_ids.numel() and (
                    owned_local_ids.min() < 0 or owned_local_ids.max() >= table.rows_per_owner))
                if stage_check(invalid_owner, table.mesh_context):
                    raise IndexError('Engram EP routing delivered a row to the wrong owner')
            owner_values = HostRowsFn.apply(table.weight, owned_local_ids, table)
            transfer_values = owner_values.to(table.lookup_compute_dtype)
            table.forward_a2a_bytes += unique_ids.numel() * unique_ids.element_size()
            table.forward_a2a_bytes += transfer_values.numel() * transfer_values.element_size()
            table.host_to_device_bytes += transfer_values.numel() * transfer_values.element_size()
            staged_values = transfer_values.to(global_ids.device)
            unique_values = ep_all_to_all(staged_values, recv_counts, send_counts, table.ep_group)

        result = unique_values.index_select(0, inverse)
        return result.view(*global_ids.shape, table.width).to(table.lookup_compute_dtype)

    host_lookup_mod.host_lookup = fast_host_lookup
    import hyper_parallel.models.deepseek_v41.adapter.engram as engram_pkg
    if hasattr(engram_pkg, 'host_lookup'):
        engram_pkg.host_lookup = fast_host_lookup
