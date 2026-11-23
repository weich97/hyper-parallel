# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Measure the native parallel CSA2 derivative without shadow execution."""

from functools import wraps
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy

import torch
import torch.distributed as dist

from hyper_parallel.trainer.vlm_trainer import VLMTrainer


def install():
    """Retain the workload observer and add only native-path execution counts."""
    parent = Path('/xw-tree/full-v41/245d2cd3-observe-full-single-microbatch.py')
    expected_parent = '245d2cd322413aea174f709d90a248693e90ddb047e64325acf12796aea01e35'
    if hashlib.sha256(parent.read_bytes()).hexdigest() != expected_parent:
        raise ValueError('Parent observer changed')
    runpy.run_path(str(parent))['install']()
    begin, end = VLMTrainer.on_train_begin, VLMTrainer.on_train_end
    state = dict(calls=0)
    binding = {}

    @wraps(begin)
    def begun(trainer):
        result = begin(trainer)
        module = importlib.import_module(
            'cann_ops_transformer.ops.attention.sparse_flash_mla_grad.sparse_flash_mla_grad'
        )
        overlay = Path('/xw-tree/full-v41/smla-dual-0100acf2')
        Path(module.__file__).resolve().relative_to(overlay.resolve())
        digest = 'ab71a8ce052cb258b5219eb21f25a8dd4fbaa641d05a7c85801985fbb2b6109e'
        if hashlib.sha256((overlay / 'overlay.json').read_bytes()).hexdigest() != digest:
            raise ValueError('Native overlay manifest changed')
        manifest = json.loads((overlay / 'overlay.json').read_text())
        for name, expected in manifest['files'].items():
            path = overlay / 'cann_ops_transformer' / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Native extension source changed')
        native = module.smlag_op_builder.load()
        fast = native.sparse_flash_mla_grad_fast
        binding.update(overlay=overlay.name, manifest_sha256=digest,
                       binary_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest())

        @wraps(fast)
        def counted(*args, **kwargs):
            if not torch.are_deterministic_algorithms_enabled():
                raise ValueError('Global deterministic policy changed')
            state['calls'] += 1
            return fast(*args, **kwargs)

        native.sparse_flash_mla_grad = counted
        print('CSA_FAST_PERFORMANCE_READY=' + json.dumps(dict(rank=dist.get_rank(), **binding)), flush=True)
        return result

    @wraps(end)
    def ended(trainer):
        result = end(trainer)
        updates = trainer.base.state.global_step
        if updates != 8 or state['calls'] != 40 * updates:
            raise ValueError('Missing actual native CSA2 backward execution')
        target = Path(os.environ['DSV41_FULL_RUN_DIR']) / 'csa-fast-performance'
        target.mkdir(exist_ok=True)
        payload = dict(rank=dist.get_rank(), updates=updates, backward_calls=state['calls'], binding=binding,
                       shadow_execution=False, formal_performance_measurement=True, final_acceptance=False,
                       global_determinism=torch.are_deterministic_algorithms_enabled())
        (target / f'rank-{dist.get_rank():03d}.json').write_text(json.dumps(payload) + '\n')
        return result

    VLMTrainer.on_train_begin = begun
    VLMTrainer.on_train_end = ended
