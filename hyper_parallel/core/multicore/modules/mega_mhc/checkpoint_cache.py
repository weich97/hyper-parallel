# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Invocation-local mHC coefficient retention for activation checkpoint replay."""

from collections import deque
from functools import lru_cache

import torch

from hyper_parallel.core.activation_memory.recompute_state import get_recompute_state
from hyper_parallel.core.multicore.modules.mega_mhc_grad.function import _resolve_core_counts
from hyper_parallel.core.multicore.scheduler.builder import build_runtime_config
from hyper_parallel.core.multicore.scheduler.config import TaskType, _validate_runtime_task_queues
from hyper_parallel.core.multicore.scheduler.runtime import serialize_runtime_config

from .graph import build_event_layout, build_mega_mhc_graph, resolve_token_tile


_RESOURCE_KEY = object()
_COUNTS = {"saved": 0, "replayed": 0, "retained_bytes_total": 0}


def checkpoint_cache_counts():
    """Return inexpensive process-local evidence of active cache use."""
    return dict(_COUNTS)


def validate_cache_mode(mode):
    """Reject unknown retention policies before the first native launch."""
    if mode not in ("none", "projection", "mappings"):
        raise ValueError(f"Unknown mHC checkpoint cache policy: {mode!r}")


@lru_cache(maxsize=16)
def _replay_plan(tokens, hidden, device_index, token_tile, mode):
    """Keep native residual/input tasks and omit the retained producers."""
    cubes, _ = _resolve_core_counts(device_index)
    token_tile = resolve_token_tile(tokens, token_tile, cubes)
    graph, topology = build_mega_mhc_graph(tokens, hidden, cubes, token_tile)
    cfg = build_runtime_config(graph, topology, rank_id=0, num_cube_cores=cubes)
    layout = build_event_layout(graph.get_op("mhc_post").task_num)
    removed = {TaskType.TASK_MHC_NORM_CAST, TaskType.TASK_MHC_PROJECTION}
    if mode == "mappings":
        removed.add(TaskType.TASK_MHC_MAPPING)
        cfg.all_event_num_triggers[layout.final] //= 2
    for index in range(cfg.task_num):
        task = cfg.all_tasks[index]
        if task.task_type == TaskType.TASK_MHC_INPUT_MIX:
            task.dependent_event = layout.post_done
        elif task.task_type == TaskType.TASK_MHC_MAPPING and mode == "projection":
            task.dependent_event = layout.post_done
    queues = (cfg.cube_task_indices, cfg.vector_task_indices, cfg.mix_task_indices)
    for kind, queue in enumerate(queues):
        kept = [queue[i] for i in range(cfg.task_index_num[kind])
                if TaskType(cfg.all_tasks[queue[i]].task_type) not in removed]
        for index, task_id in enumerate(kept):
            queue[index] = task_id
        cfg.task_index_num[kind] = len(kept)
    _validate_runtime_task_queues(cfg, cfg.event_capacity)
    host_image = torch.frombuffer(bytearray(serialize_runtime_config(cfg)), dtype=torch.uint8)
    return host_image.to(device=f"npu:{device_index}"), int(cfg.event_capacity)


def _retain(outputs, cache, mode):
    """Retain compact coefficients, never the large residual or mixed input."""
    saved = dict(hc_before_norm=cache.hc_before_norm, inv_rms=cache.inv_rms)
    if mode == "mappings":
        saved.update(next_pre=outputs[1], next_post=outputs[2], next_res=outputs[3],
                     sum_out=cache.sum_out, norm_out=cache.norm_out)
    return {name: value.detach() for name, value in saved.items()}


def _replay(inputs, saved, mode, options):
    """Run the unchanged native arithmetic for the uncached outputs."""
    from .function import _restore_forward_outputs  # pylint: disable=import-outside-toplevel

    previous, residual, pre, post, res_mix, phi, alpha, bias, norm_weight = inputs
    leading, hidden = tuple(residual.shape[:-2]), residual.shape[-1]
    tokens = residual.numel() // (4 * hidden)
    residual = residual.reshape(1, tokens, 4, hidden)
    plan, capacity = _replay_plan(tokens, hidden, residual.device.index, options["token_tile"], mode)
    plan.record_stream(torch.npu.current_stream(residual.device))
    events = torch.zeros(capacity * 4, dtype=torch.uint8, device=residual.device)
    new_residual = torch.empty_like(residual)
    block = torch.empty((1, tokens, hidden), dtype=residual.dtype, device=residual.device)
    mixed = torch.empty_like(block)
    rstd = torch.empty((1, tokens, 1), dtype=torch.float32, device=residual.device)
    if mode == "mappings":
        next_pre, next_post = (saved[name].reshape(1, tokens, 4) for name in ("next_pre", "next_post"))
        next_res = saved["next_res"].reshape(1, tokens, 16)
        sum_out, norm_out = saved["sum_out"], saved["norm_out"]
    else:
        next_pre = torch.empty((1, tokens, 4), dtype=torch.float32, device=residual.device)
        next_post = torch.empty_like(next_pre)
        next_res = torch.empty((1, tokens, 16), dtype=torch.float32, device=residual.device)
        sum_out = torch.empty((2 * options["num_iters"], 1, tokens, 4), dtype=torch.float32, device=residual.device)
        norm_out = torch.empty((2 * options["num_iters"], 1, tokens, 4, 4), dtype=torch.float32, device=residual.device)
    cache = (saved["hc_before_norm"], saved["inv_rms"], sum_out, norm_out, mixed, rstd)
    torch.ops.hyper_parallel.mega_mhc(
        previous.reshape(1, tokens, hidden).contiguous(), residual.contiguous(), pre.reshape(1, tokens, 4).contiguous(),
        post.reshape(1, tokens, 4).contiguous(), res_mix.reshape(1, tokens, 4, 4).contiguous(),
        phi.contiguous(), alpha.contiguous(), bias.contiguous(), norm_weight.contiguous(), plan, events, events,
        new_residual, next_pre, next_post, next_res, block, *cache,
        options["hc_eps"], options["norm_eps"], options["num_iters"], True,
    )
    return _restore_forward_outputs((new_residual, next_pre, next_post,
                                    next_res.reshape(1, tokens, 4, 4), block, *cache), leading, hidden)


def checkpoint_forward(forward, inputs, mode, **options):
    """Reuse coefficients only within the same HP checkpoint invocation.

    Original native backward and its saved-tensor order remain unchanged. Each
    invocation owns its queue, so microbatches cannot share stale coefficients.
    The HP checkpoint lifecycle releases the queue, including on early stop.
    """
    validate_cache_mode(mode)
    state = get_recompute_state() if mode != "none" else None
    if state is None:
        return forward(*inputs, need_backward=True, **options)
    signature = (mode, tuple((tuple(t.shape), t.dtype, t.device) for t in inputs),
                 tuple(sorted(options.items())))
    queue = state.get_resource(_RESOURCE_KEY, deque)
    if state.is_recomputing:
        if not queue:
            raise RuntimeError("mHC checkpoint replay has no matching coefficient cache")
        original_signature, saved = queue.popleft()
        if original_signature != signature:
            raise RuntimeError("mHC checkpoint replay changed its input metadata or options")
        result = _replay(inputs, saved, mode, options)
        _COUNTS["replayed"] += 1
        return result
    outputs, cache = forward(*inputs, need_backward=True, **options)
    saved = _retain(outputs, cache, mode)
    queue.append((signature, saved))
    _COUNTS["saved"] += 1
    _COUNTS["retained_bytes_total"] += sum(t.numel() * t.element_size() for t in saved.values())
    return outputs, cache
