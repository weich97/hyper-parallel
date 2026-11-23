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
"""Diagnose whole-model mHC reduction ordering without replacing the native kernel."""

from collections import Counter
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import runpy

from hyper_parallel.core.multicore.modules.mega_mhc import function as forward_module
from hyper_parallel.core.multicore.modules.mega_mhc_grad import function as backward_module
from hyper_parallel.trainer.vlm_trainer import VLMTrainer


def install() -> None:
    """Retain identity observers and force one scheduler tile per native backward."""
    observer = Path(__file__).with_name("df307082-verify-full-v41-dspark-initialization.py")
    expected = "df307082da2cd4ef416cf5ad58590b8f79014962636da766bb704e3d3db27262"
    if hashlib.sha256(observer.read_bytes()).hexdigest() != expected:
        raise ValueError("The frozen initialization and native-path observer changed")
    runpy.run_path(str(observer))["install"]()
    original = backward_module.hyper_mega_mhc_grad
    if forward_module.hyper_mega_mhc_grad is not original:
        raise ValueError("Unexpected mHC backward binding before diagnostic installation")
    calls = Counter()
    original_end = VLMTrainer.on_train_end

    @wraps(original)
    def ordered(*args, **kwargs):
        residual = args[6] if len(args) > 6 else kwargs["residual"]
        tokens = residual.numel() // (4 * residual.shape[-1])
        kwargs["token_tile"] = tokens
        result = original(*args, **kwargs)
        calls[tokens] += 1
        return result

    def write_receipt() -> None:
        if not calls:
            raise RuntimeError("The ordered mHC backward path was not executed")
        rank = int(os.environ["RANK"])
        destination = Path(os.environ["DSV41_FULL_RUN_DIR"]) / "mhc-reduction-order"
        destination.mkdir(exist_ok=True)
        report = {"rank": rank, "mode": "single_native_scheduler_tile", "calls_by_tokens": dict(calls),
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "purpose": "Causal diagnosis only; not the accepted performance execution mode"}
        path = destination / f"rank-{rank:03}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    @wraps(original_end)
    def end(trainer):
        result = original_end(trainer)
        # The distributed worker uses os._exit after teardown, bypassing atexit callbacks.
        write_receipt()
        return result

    backward_module.hyper_mega_mhc_grad = ordered
    forward_module.hyper_mega_mhc_grad = ordered
    VLMTrainer.on_train_end = end
    print("FULL_V41_MHC_REDUCTION_MODE=single_native_scheduler_tile diagnostic_only=1", flush=True)
