"""Stage 9: Chain Stage 5 + cache Host Engram forward lookup metadata/values into backward recompute."""
from functools import wraps
import hashlib
from pathlib import Path
import runpy

import torch
import torch.distributed as dist

from hyper_parallel.models.deepseek_v41.adapter.engram import host_lookup as host_lookup_mod
from hyper_parallel.models.deepseek_v41.adapter.engram import host_table as host_table_mod
from hyper_parallel.models.deepseek_v41.adapter.engram.host_lookup import HostRowsFn, ep_all_to_all, stage_check

PARENT = Path('/xw-tree/full-v41/opt5-observe-engram-fast.py')
PARENT_SHA = 'bedb0f29e33aba1f000d4c958eece5e5b781abc5bebbc8eac8178a4170d8b5e6'


class _CachedHostRowsFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, local_ids, cached_transfer_values, table):
        ctx.table = table
        ctx.save_for_backward(local_ids)
        return cached_transfer_values

    @staticmethod
    def backward(ctx, grad_rows):
        (local_ids,) = ctx.saved_tensors
        ctx.table.append_rows(local_ids, grad_rows.float().contiguous())
        return None, None, None, None


def install():
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError('Stage 5 observer changed')
    runpy.run_path(str(PARENT))['install']()

    lookup_calls = [0]

    def cached_host_lookup(table, global_ids):
        if table.rows_per_owner is None or table.global_row_start is None:
            raise RuntimeError('Host Engram table is not bound')
        if table.weight.device.type != 'cpu' or table.weight.dtype != torch.float32:
            raise RuntimeError('Host Engram weight must remain CPU FP32')

        # If in recompute (grad enabled) and we have a cached forward entry for this table, reuse it!
        if torch.is_grad_enabled():
            cache = getattr(table, '_dsv41_recompute_cache', None)
            if cache is not None:
                table._dsv41_recompute_cache = None
                inverse, send_counts, recv_counts, owned_local_ids, cached_transfer_values = cache
                if table.ep_size == 1:
                    owner_values = _CachedHostRowsFn.apply(table.weight, owned_local_ids, cached_transfer_values, table)
                    unique_values = owner_values.to(global_ids.device)
                else:
                    transfer_values = _CachedHostRowsFn.apply(table.weight, owned_local_ids, cached_transfer_values, table)
                    staged_values = transfer_values.to(global_ids.device)
                    unique_values = ep_all_to_all(staged_values, recv_counts, send_counts, table.ep_group)
                result = unique_values.index_select(0, inverse)
                return result.view(*global_ids.shape, table.width).to(table.lookup_compute_dtype)

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
            if not torch.is_grad_enabled():
                table._dsv41_recompute_cache = (inverse, None, None, local_ids, transfer_values)
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
            if not torch.is_grad_enabled():
                table._dsv41_recompute_cache = (inverse, send_counts, recv_counts, owned_local_ids, transfer_values)
            staged_values = transfer_values.to(global_ids.device)
            unique_values = ep_all_to_all(staged_values, recv_counts, send_counts, table.ep_group)

        result = unique_values.index_select(0, inverse)
        return result.view(*global_ids.shape, table.width).to(table.lookup_compute_dtype)

    host_lookup_mod.host_lookup = cached_host_lookup
    host_table_mod.host_lookup = cached_host_lookup
