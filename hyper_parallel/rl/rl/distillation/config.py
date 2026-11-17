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
"""Validated optional distillation configuration and resume identity."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union
from urllib.parse import urlparse


@dataclass(frozen=True)
class TeacherConfig:
    """One frozen, externally managed vLLM teacher."""

    model_path: str
    endpoint: str
    served_model_name: str
    tokenizer_path: str
    max_model_len: int


@dataclass(frozen=True)
class DistillationConfig:
    """Sampled k1 PG objective independent of the task algorithm."""

    enabled: bool = False
    use_task_rewards: bool = True
    coefficient: float = 1.0
    clip_ratio: float = 0.2
    dual_clip: float = 3.0
    loss_max_clamp: Optional[float] = None
    teacher_key: str = "teacher_key"
    timeout_seconds: float = 120.0
    max_mean_logprob_diff: float = 0.1
    max_logprob_diff: float = 1.0
    max_ratio_outside_fraction: float = 0.05
    teachers: tuple[tuple[str, TeacherConfig], ...] = ()


def _positive_float(value: Any, name: str, *, allow_zero: bool = False) -> float:
    """Validate a finite numerical configuration value."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not a boolean")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (not allow_zero and result == 0):
        raise ValueError(f"{name} must be finite and positive")
    return result


def _parse_loss(loss: Any) -> tuple[Mapping[str, Any], bool, float, float, Optional[float]]:
    """Validate the sampled-k1 PG loss configuration."""
    if not isinstance(loss, Mapping):
        raise ValueError("distillation.loss must be a mapping")
    allowed_loss = {
        "mode",
        "use_policy_gradient",
        "use_task_rewards",
        "coefficient",
        "clip_ratio",
        "dual_clip",
        "loss_max_clamp",
    }
    if set(loss) - allowed_loss:
        raise ValueError(f"Unsupported distillation.loss fields: {sorted(set(loss) - allowed_loss)}")
    if loss.get("mode", "k1") != "k1" or loss.get("use_policy_gradient", True) is not True:
        raise ValueError("Distillation currently supports k1 with use_policy_gradient=true only")
    use_task = loss.get("use_task_rewards", True)
    if not isinstance(use_task, bool):
        raise ValueError("distillation.loss.use_task_rewards must be a boolean")
    clip = _positive_float(loss.get("clip_ratio", 0.2), "clip_ratio", allow_zero=True)
    dual = _positive_float(loss.get("dual_clip", 3.0), "dual_clip")
    if clip >= 1 or dual <= 1:
        raise ValueError("Distillation requires clip_ratio < 1 and dual_clip > 1")
    clamp = loss.get("loss_max_clamp")
    if clamp is not None:
        clamp = _positive_float(clamp, "loss_max_clamp")
    return loss, use_task, clip, dual, clamp


def _parse_teachers(raw_teachers: Any) -> tuple[tuple[str, TeacherConfig], ...]:
    """Validate frozen teacher endpoints and model configuration."""
    if not isinstance(raw_teachers, Mapping) or not raw_teachers:
        raise ValueError("Enabled distillation requires at least one teacher")
    teachers = []
    for key, teacher in raw_teachers.items():
        if not isinstance(key, str) or not key or not isinstance(teacher, Mapping):
            raise ValueError("Teacher entries require a nonempty key and mapping")
        fields = {"model_path", "endpoint", "served_model_name", "tokenizer_path", "max_model_len"}
        if set(teacher) - fields:
            raise ValueError(f"Unsupported teacher fields for {key}: {sorted(set(teacher) - fields)}")
        path = str(teacher.get("model_path", ""))
        endpoint = str(teacher.get("endpoint", "")).rstrip("/")
        parsed = urlparse(endpoint)
        if not path or parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
            raise ValueError(f"Teacher {key} requires model_path and an HTTP(S) endpoint without credentials")
        length = teacher.get("max_model_len", 0)
        if isinstance(length, bool) or not isinstance(length, int) or length < 2:
            raise ValueError(f"Teacher {key} requires integer max_model_len >= 2")
        teachers.append(
            (
                key,
                TeacherConfig(
                    path,
                    endpoint,
                    str(teacher.get("served_model_name", path)),
                    str(teacher.get("tokenizer_path", path)),
                    length,
                ),
            )
        )
    return tuple(teachers)


def parse_distillation_config(value: Optional[Mapping[str, Any]]) -> DistillationConfig:
    """Validate configuration without contacting a teacher or loading a model."""
    if value is None:
        return DistillationConfig()
    if not isinstance(value, Mapping):
        raise ValueError("distillation must be a mapping")
    allowed = {"enabled", "teacher_key", "teachers", "loss", "timeout_seconds", "probability_gate"}
    if set(value) - allowed:
        raise ValueError(f"Unsupported distillation fields: {sorted(set(value) - allowed)}")
    enabled = value.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("distillation.enabled must be a boolean")
    if not enabled:
        return DistillationConfig()
    loss, use_task, clip, dual, clamp = _parse_loss(value.get("loss", {}))
    routing = value.get("teacher_key", "teacher_key")
    if not isinstance(routing, str) or not routing:
        raise ValueError("distillation.teacher_key must be a nonempty string")
    raw_teachers = value.get("teachers")
    gate = value.get("probability_gate", {})
    gate_fields = {"max_mean_logprob_diff", "max_logprob_diff", "max_ratio_outside_fraction"}
    if not isinstance(gate, Mapping) or set(gate) - gate_fields:
        raise ValueError("Unsupported distillation.probability_gate configuration")
    gate_limits = {
        name: _positive_float(gate.get(name, default), name, allow_zero=True)
        for name, default in (("max_mean_logprob_diff", 0.1), ("max_logprob_diff", 1.0),
                              ("max_ratio_outside_fraction", 0.05))
    }
    if gate_limits["max_ratio_outside_fraction"] > 1:
        raise ValueError("max_ratio_outside_fraction must be in [0, 1]")
    teachers = _parse_teachers(raw_teachers)
    return DistillationConfig(
        enabled=True,
        use_task_rewards=use_task,
        coefficient=_positive_float(loss.get("coefficient", 1.0), "coefficient"),
        clip_ratio=clip,
        dual_clip=dual,
        loss_max_clamp=clamp,
        teacher_key=routing,
        timeout_seconds=_positive_float(value.get("timeout_seconds", 120.0), "timeout_seconds"),
        teachers=teachers,
        **gate_limits,
    )


def tokenizer_identity(path: str) -> tuple[str, bytes, bytes]:
    """Require identical tokenizer artifacts, including special-token configuration."""
    directory = Path(path)
    tokenizer = directory / "tokenizer.json"
    if not tokenizer.is_file():
        raise ValueError(f"Distillation requires tokenizer.json: {tokenizer}")
    optional = []
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        source = directory / name
        optional.append(source.read_bytes() if source.is_file() else b"")
    return hashlib.sha256(tokenizer.read_bytes()).hexdigest(), *optional


def teacher_weight_files(path: str) -> tuple[Path, ...]:
    """List HF weights and indices, rejecting missing or unsafe indexed shards."""
    directory = Path(path)
    files = set(directory.glob("*.safetensors")) | set(directory.glob("pytorch_model*.bin"))
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = directory / name
        if not index.is_file():
            continue
        contents = json.loads(index.read_text(encoding="utf-8"))
        weight_map = contents.get("weight_map") if isinstance(contents, Mapping) else None
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ValueError(f"Teacher weight index requires a nonempty weight_map: {index}")
        files.add(index)
        for shard in weight_map.values():
            if not isinstance(shard, str) or not shard or Path(shard).is_absolute() or ".." in Path(shard).parts:
                raise ValueError(f"Invalid teacher shard path in {index}: {shard!r}")
            source = directory / shard
            if not source.is_file():
                raise ValueError(f"Teacher weight shard is missing: {source}")
            files.add(source)
    return tuple(sorted(files))


def _file_fingerprint(path: Path) -> str:
    """Hash weight contents with bounded memory and reject changes during reading."""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    fields = ("st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise ValueError(f"Teacher artifact changed while computing its identity: {path}")
    return digest.hexdigest()


def distillation_fingerprint(value: Optional[Union[Mapping[str, Any], DistillationConfig]]) -> Optional[str]:
    """Bind resume to the objective, routing and locally available teacher artifacts."""
    config = value if isinstance(value, DistillationConfig) else parse_distillation_config(value)
    if not config.enabled:
        return None
    identity = asdict(config)
    identity["probability_contract"] = "sampled-rollout-anchor-v2"
    identity["teacher_identity_contract"] = "weight-content-sha256-v1"
    identity["teachers"] = []
    for key, teacher in config.teachers:
        source = Path(teacher.model_path) / "config.json"
        identity["teachers"].append(
            {
                "key": key,
                "model_path": teacher.model_path,
                "max_model_len": teacher.max_model_len,
                "model_config": hashlib.sha256(source.read_bytes()).hexdigest(),
                "weights": {str(path.relative_to(teacher.model_path)): _file_fingerprint(path)
                            for path in teacher_weight_files(teacher.model_path)},
                "tokenizer": hashlib.sha256(repr(tokenizer_identity(teacher.tokenizer_path)).encode()).hexdigest(),
            }
        )
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
