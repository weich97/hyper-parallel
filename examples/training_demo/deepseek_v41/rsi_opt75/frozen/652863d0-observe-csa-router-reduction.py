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

"""Add the module-qualified auxiliary reduction to the native CSA2 speed candidate."""

from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import runpy

from hyper_parallel.trainer.vlm_trainer import VLMTrainer


def install():
    """Preserve the parent native path and record executed sequence reductions."""
    root = Path('/xw-tree/full-v41')
    parent = root / 'bb572c9a-observe-csa-fast-performance.py'
    expected = os.environ['DSV41_CSA_PERFORMANCE_PARENT_SHA']
    if hashlib.sha256(parent.read_bytes()).hexdigest() != expected:
        raise ValueError('CSA2 parent observer changed')
    runpy.run_path(str(parent))['install']()
    patch = root / '2f9eee19-router-segment-reduction-candidate.py'
    if hashlib.sha256(patch.read_bytes()).hexdigest() != '2f9eee19fd86739fcc7e807eeaa8fbdc19a68d069a2d40d926f3a20fee040d6d':
        raise ValueError('Router reduction patch changed')
    counts = runpy.run_path(str(patch))['install']()
    end = VLMTrainer.on_train_end

    @wraps(end)
    def ended(trainer):
        result = end(trainer)
        if trainer.base.state.global_step != 8 or counts['dense_calls'] < 40 * 8:
            raise ValueError('Missing real auxiliary-loss execution')
        target = Path(os.environ['DSV41_FULL_RUN_DIR']) / 'router-segment-reduction'
        target.mkdir(exist_ok=True)
        rank = int(os.environ['RANK'])
        payload = dict(rank=rank, updates=trainer.base.state.global_step, **counts,
                       objective='Unchanged expert counts, sample masks, FP32 probability normalization and partition SUM')
        (target / f'rank-{rank:03d}.json').write_text(json.dumps(payload) + '\n')
        return result

    VLMTrainer.on_train_end = ended
