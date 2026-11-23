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
# ============================================================================
"""Bind the measured mHC tile while preserving the recovered full-model runtime."""

from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import runpy

import torch

from hyper_parallel.core.multicore.modules.mega_mhc import function as forward_module
from hyper_parallel.core.multicore.modules.mega_mhc_grad import function as backward_module
from hyper_parallel.trainer.vlm_trainer import VLMTrainer


def install():
    """Preserve the baseline policies and attest explicit mHC backward dispatch."""
    parent_sha = "2e4edeeccb71090d550f961d2299b8fca076f92fdfc2c133078ac06a32f44560"
    parent = Path("/xw-tree/full-v41/2e4edeec-observe-full-stage-replay-storage.py")
    if hashlib.sha256(parent.read_bytes()).hexdigest() != parent_sha:
        raise ValueError("The baseline runtime observer changed")
    original = backward_module.hyper_mega_mhc_grad
    if forward_module.hyper_mega_mhc_grad is not original:
        raise RuntimeError("Unexpected mHC backward binding")
    runpy.run_path(str(parent))["install"]()
    end = VLMTrainer.on_train_end
    native_calls = 0

    @wraps(original)
    def parallel(*args, **kwargs):
        nonlocal native_calls
        residual = args[6]
        if (residual.ndim not in (3, 4)
                or residual.numel() // (4 * residual.shape[-1]) != 16384
                or residual.shape[-1] != 5120 or not torch.are_deterministic_algorithms_enabled()):
            raise RuntimeError("Unexpected training shape or determinism state for this benchmark")
        result = original(*args, **kwargs, parallel_previous_x=True)
        native_calls += 1
        return result

    @wraps(end)
    def checked_end(trainer):
        result = end(trainer)
        steps = trainer.base.state.global_step
        if native_calls != 80 * steps:
            raise RuntimeError(f"Expected all backbone mHC backward calls, observed {native_calls} for {steps} steps")
        rank = int(os.environ["RANK"])
        target = Path(os.environ["DSV41_FULL_RUN_DIR"]) / "ordered-output-mhc"
        target.mkdir(exist_ok=True)
        row = {"rank": rank, "completed_steps": steps, "native_parallel_calls": native_calls,
               "tokens": 16384, "parameter_reduction": "original_order", "global_determinism": torch.are_deterministic_algorithms_enabled()}
        path = target / f"rank-{rank:03d}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(row) + "\n", encoding="utf-8")
        temporary.replace(path)
        return result

    backward_module.hyper_mega_mhc_grad = parallel
    forward_module.hyper_mega_mhc_grad = parallel
    VLMTrainer.on_train_end = checked_end
    print("FULL_V41_MHC_REDUCTION_MODE=ordered_reductions_parallel_activation_output", flush=True)
