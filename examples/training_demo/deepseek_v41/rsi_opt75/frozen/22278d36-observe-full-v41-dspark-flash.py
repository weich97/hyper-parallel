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
"""Record initial-state identity and actual DSpark native-path gradient traversal."""

from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import runpy

from hyper_parallel.models.deepseek_v41.dspark import DSparkDraftAttention
from hyper_parallel.trainer.vlm_trainer import VLMTrainer


def install() -> None:
    """Attach gate-only observers that leave tensors and gradients unchanged."""
    fingerprint = Path(__file__).with_name("1dd50fd7-fingerprint-full-v41-initial-state.py")
    expected = "1dd50fd7e31ebb786e93343964e1e5409fe633ac071cae738d44d5d096e143c5"
    if hashlib.sha256(fingerprint.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Initial-state observer identity changed")
    runpy.run_path(str(fingerprint))["install"]()
    original_begin = VLMTrainer.on_train_begin
    original_step = VLMTrainer.train_step
    original_context = DSparkDraftAttention._flash_window_context
    names = {}
    counters = {}
    steps = []

    @wraps(original_begin)
    def begin(trainer):
        original_begin(trainer)
        for name, module in trainer.base.model.named_modules():
            if isinstance(module, DSparkDraftAttention):
                if module.attention_backend != "flash_v3":
                    raise RuntimeError(f"DSpark replacement not selected for {name}")
                names[id(module)] = name
                counters[name] = {"forward_calls": 0, "gradient_traversals": 0}
        if len(names) != 3:
            raise RuntimeError(f"Expected three full-model DSpark attention stages, found {len(names)}")

    @wraps(original_context)
    def context(module, *args, **kwargs):
        output = original_context(module, *args, **kwargs)
        if id(module) not in names:
            # Model construction may execute a preparation call before the observed training window.
            return output
        counter = counters[names[id(module)]]
        counter["forward_calls"] += 1
        if output.requires_grad:
            def observe_gradient(_gradient):
                counter["gradient_traversals"] += 1
                return None

            output.register_hook(observe_gradient)
        return output

    @wraps(original_step)
    def step(trainer, iterator):
        before = {name: dict(counts) for name, counts in counters.items()}
        result = original_step(trainer, iterator)
        delta = {
            name: {key: count - before[name][key] for key, count in counts.items()}
            for name, counts in counters.items()
        }
        if len(delta) != 3 or any(value <= 0 for counts in delta.values() for value in counts.values()):
            raise RuntimeError(f"Missing DSpark native forward or gradient traversal: {delta}")
        steps.append({"step": trainer.base.state.global_step, "stages": delta})
        directory = Path(os.environ["DSV41_FULL_RUN_DIR"]) / "native-path"
        directory.mkdir(exist_ok=True)
        path = directory / f"rank-{int(os.environ['RANK']):03}.json"
        temporary = path.with_suffix(".tmp")
        receipt = {
            "rank": int(os.environ["RANK"]), "backend": "flash_v3", "steps": steps,
            "scope": "Actual native-call output and gradient traversal; device-kernel timing is separate.",
        }
        temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return result

    VLMTrainer.on_train_begin = begin
    VLMTrainer.train_step = step
    DSparkDraftAttention._flash_window_context = context
