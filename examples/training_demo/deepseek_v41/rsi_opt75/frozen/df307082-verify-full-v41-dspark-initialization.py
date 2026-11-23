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
"""Check canonical DSpark local shards in the real full-model initialization."""

import hashlib
import json
import os
from pathlib import Path
import runpy

import torch

from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.trainer.vlm_trainer import VLMTrainer
from hyper_parallel.models.deepseek_v41.dspark import DSparkTokenEmbedding


def tensor_hash(value: torch.Tensor) -> str:
    """Hash tensor contents without changing the tensor or random state."""
    raw = value.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(memoryview(raw.numpy())).hexdigest()


def install() -> None:
    """Compose identity and native-path observers with a zero-tolerance shard check."""
    native = Path(__file__).with_name("22278d36-observe-full-v41-dspark-flash.py")
    expected = "22278d368b413228b018e7b81cff22dfe6fe1789e443344c3094f169baa42082"
    if hashlib.sha256(native.read_bytes()).hexdigest() != expected:
        raise ValueError("The frozen DSpark native-path observer changed")
    runpy.run_path(str(native))["install"]()
    embedding_forward = DSparkTokenEmbedding.forward
    checked = False

    def check_embedding(module, input):
        nonlocal checked
        output = embedding_forward(module, input)
        if not checked:
            weight = module.weight.full_tensor() if isinstance(module.weight, DTensor) else module.weight
            row = weight[module.noise_token_id].detach()
            if module.cached_noise_embedding.requires_grad or not torch.equal(row, module.cached_noise_embedding):
                raise ValueError("FSDP exposed an inconsistent shared noise-token row")
            rank = int(os.environ["RANK"])
            directory = Path(os.environ["DSV41_FULL_RUN_DIR"]) / "dspark-shared-noise"
            directory.mkdir(exist_ok=True)
            report = {"rank": rank, "status": "PASS", "shape": list(row.shape),
                      "row_sha256": tensor_hash(row), "scope": "First unsharded backbone embedding forward"}
            path = directory / f"rank-{rank:03}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
            checked = True
        return output

    DSparkTokenEmbedding.forward = check_embedding
    original = VLMTrainer.on_train_begin

    def begin(trainer: VLMTrainer) -> None:
        original(trainer)
        model = trainer.base.model
        drafter = model.get_submodule("dspark")
        std = model.config.initializer_range
        names = {}
        if "noise_embedding" in dict(drafter.named_parameters()):
            raise ValueError("DSpark must reuse the backbone noise-token embedding")
        for index in range(len(drafter.stages)):
            names[f"stages.{index}.attn_hc.fn"] = 0x44530100 + index
            names[f"stages.{index}.ffn_hc.fn"] = 0x44530200 + index
        rows = []
        for name, seed in names.items():
            parameter = drafter.get_parameter(name)
            local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
            canonical = torch.randn(parameter.shape, device="cpu", generator=torch.Generator().manual_seed(seed)) * std
            expected_values = canonical.to(local.device, local.dtype)
            if isinstance(parameter, DTensor):
                expected_values = distribute_tensor(
                    expected_values, parameter.device_mesh, parameter.placements,
                ).to_local()
            matched = bool(torch.equal(local, expected_values))
            rows.append({"name": name, "global_shape": list(parameter.shape), "local_shape": list(local.shape),
                         "dtype": str(local.dtype), "canonical_global_sha256": tensor_hash(canonical),
                         "local_sha256": tensor_hash(local), "expected_local_sha256": tensor_hash(expected_values),
                         "exact": matched})
        rank = int(os.environ["RANK"])
        receipt = {"rank": rank, "step": 0, "source": "fixed-seed canonical global parameter",
                   "parameters": rows, "status": "PASS" if all(row["exact"] for row in rows) else "FAIL"}
        directory = Path(os.environ["DSV41_FULL_RUN_DIR"]) / "dspark-initialization"
        directory.mkdir(exist_ok=True)
        output = directory / f"rank-{rank:03}.json"
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
        if receipt["status"] != "PASS":
            raise ValueError(f"DSpark canonical initialization mismatch on rank {rank}")
        print(f"FULL_V41_DSPARK_INIT_PASS rank={rank}", flush=True)

    VLMTrainer.on_train_begin = begin
