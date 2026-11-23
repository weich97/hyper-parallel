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

"""Measure the existing parallel mHC schedule on the CSA2/router candidate."""

from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import runpy

from hyper_parallel.trainer.vlm_trainer import VLMTrainer


def install():
    """Preserve controls and explicitly attest the changed native reduction mode."""
    root = Path('/xw-tree/full-v41')
    parent = root / '652863d0-observe-csa-router-reduction.py'
    if hashlib.sha256(parent.read_bytes()).hexdigest() != '652863d0f0c76c98c1763c4b74357a0c7dd8166acecb470a038fa381d6b90fb7':
        raise ValueError('Parent execution recipe changed')
    original = runpy.run_path
    modified = []

    def load(path, *args, **kwargs):
        observed = Path(path)
        if observed.name != '245d2cd3-observe-full-single-microbatch.py':
            return original(path, *args, **kwargs)
        data = observed.read_bytes()
        if hashlib.sha256(data).hexdigest() != '245d2cd322413aea174f709d90a248693e90ddb047e64325acf12796aea01e35':
            raise ValueError('mHC observer changed')
        text = data.decode()
        if text.count('ORDERED_PROJECTION = True') != 1:
            raise ValueError('Unknown projection policy')
        text = text.replace('ORDERED_PROJECTION = True', 'ORDERED_PROJECTION = False')
        text = text.replace('ordered_projection_parallel_vector_reductions', 'parallel_atomic_optin')
        namespace = {'__name__': 'explicit_mhc_parallel_candidate', '__file__': str(observed)}
        exec(compile(text, str(observed), 'exec'), namespace)
        modified.append(dict(source_sha256=hashlib.sha256(data).hexdigest(),
                             executed_sha256=hashlib.sha256(text.encode()).hexdigest()))
        return namespace

    runpy.run_path = load
    try:
        original(str(parent))['install']()
    finally:
        runpy.run_path = original
    if len(modified) != 1:
        raise ValueError('Expected exactly one explicitly changed mHC policy')
    end = VLMTrainer.on_train_end

    @wraps(end)
    def finished(trainer):
        result = end(trainer)
        directory = Path(os.environ['DSV41_FULL_RUN_DIR']) / 'execution-mode'
        directory.mkdir(exist_ok=True)
        rank = int(os.environ['RANK'])
        receipt = dict(rank=rank, updates=trainer.base.state.global_step,
                       parameter_reduction='parallel_atomic_optin', execution_override=modified[0],
                       original_mhc_gate='FAIL_PRESERVED', original_csa_gate='FAIL_PRESERVED',
                       final_acceptance=False, screening_only=True)
        (directory / f'rank-{rank:03d}.json').write_text(json.dumps(receipt) + '\n')
        return result

    VLMTrainer.on_train_end = finished
