"""Capture abnormal gradients at the existing scalar-norm synchronization point."""
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import runpy
import statistics
import textwrap

import torch

from hyper_parallel.core.utils.clip_grad import _get_local_grad
from hyper_parallel.models.deepseek_v41.adapter.engram import host_state


def install():
    root = Path('/xw-tree/full-v41')
    parent = root / 'ac736691-observe-nondeterministic-repeat.py'
    if hashlib.sha256(parent.read_bytes()).hexdigest() != 'ac736691f2641e236734ee7dca37728faf3dc6140d8cfc1fbb7a15f504bf1463':
        raise ValueError('The long-horizon observer changed')
    runpy.run_path(str(parent))['install']()
    original = host_state.DeepseekV41HostState._combined_gradient_norm
    source = textwrap.dedent(inspect.getsource(original))
    boundary = '    if not bool(torch.isfinite(global_sq)):\n'
    if source.count(boundary) != 1:
        raise ValueError('Unknown combined-norm synchronization boundary')
    replacement = ('    scalar_square = float(global_sq)\n'
                   '    _observe_norm_square(self, scalar_square)\n'
                   '    if not math.isfinite(scalar_square):\n')
    adapted = source.replace(boundary, replacement)
    rank = int(os.environ['RANK'])
    directory = Path(os.environ['DSV41_FULL_RUN_DIR']) / 'norm-guard'
    directory.mkdir(exist_ok=True)
    values = []
    thresholds = []

    def save(state, **extra):
        path = directory / f'rank-{rank:03d}.json'
        payload = dict(rank=rank, state=state, observed_updates=len(values), norm_squares=values,
                       limit_squared=1e12, rolling_norm_multiplier=10.0, rolling_window=10, minimum_history=2,
                       thresholds_squared=thresholds, actual_global_determinism=torch.are_deterministic_algorithms_enabled(),
                       original_method_sha256=hashlib.sha256(source.encode()).hexdigest(),
                       executed_method_sha256=hashlib.sha256(adapted.encode()).hexdigest(),
                       normal_path='One scalar read at the existing finite-check synchronization point; no tensor probes or additional collective.',
                       **extra)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, allow_nan=False) + '\n')
        temporary.replace(path)

    def observe(state, scalar_square):
        history = [math.sqrt(v) for v in values[-10:] if isinstance(v, (int, float)) and v > 0]
        limit_squared = min(1e12, (10.0 * statistics.median(history)) ** 2) if len(history) >= 2 else 1e12
        thresholds.append(limit_squared)
        values.append(scalar_square if math.isfinite(scalar_square) else str(scalar_square))
        if math.isfinite(scalar_square) and scalar_square <= limit_squared:
            if len(values) == 30:
                save('NO_CATASTROPHIC_NORM_IN_30_UPDATES')
            return
        # Only failed runs execute these tensor scans. They are never timed as
        # valid performance measurements and are captured before clipping.
        names, maxima, metadata, weights = [], [], [], []
        for name, parameter in state.model.named_parameters():
            grad = _get_local_grad(parameter)
            if grad is None or grad.is_sparse or grad.device.type != 'npu' or not grad.numel():
                continue
            weight = getattr(parameter, '_local_tensor', parameter)
            names.append(name)
            maxima.append(grad.detach().abs().max().float())
            weights.append(weight.detach().abs().max().float())
            metadata.append(dict(numel=grad.numel(), dtype=str(grad.dtype), shape=list(grad.shape),
                                 norm_representative=state.norm_shard_plan.representatives.get(name)))
        gradient_values = torch.stack(maxima).cpu().tolist()
        weight_values = torch.stack(weights).cpu().tolist()
        rows = [dict(name=name, max_abs=g if math.isfinite(g) else str(g),
                     weight_max_abs=w if math.isfinite(w) else str(w), **meta)
                for name, g, w, meta in zip(names, gradient_values, weight_values, metadata)]
        sparse = [dict(name=name, nnz=table.pending._nnz(),
                       max_abs=float(table.pending.values().abs().max()) if table.pending._nnz() else 0.0,
                       square_sum_fp64=float(table.pending.values().double().square().sum()))
                  for name, table in state.tables.items()]
        save('CATASTROPHIC_NORM_CAPTURED_BEFORE_CLIP', dense_gradient_maxima=rows, local_sparse=sparse)
        print('CATASTROPHIC_GRADIENT_CAPTURED rank=' + str(rank) + ' update=' + str(len(values)), flush=True)
        raise RuntimeError('Catastrophic gradient norm captured before clipping; reject this run')

    namespace = dict(original.__globals__, math=math, _observe_norm_square=observe)
    exec(compile(adapted, str(Path(__file__)), 'exec'), namespace)
    host_state.DeepseekV41HostState._combined_gradient_norm = namespace['_combined_gradient_norm']
    save('INSTALLED')
