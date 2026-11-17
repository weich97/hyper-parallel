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
"""Compare every declared VLM matrix row using structured Trainer evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from run_vlm_matrix import _changed_assets, _source_sha256

from hyper_parallel.tools.model_integration.case_compare import (
    compare_parameter_probes,
    compare_preflight_configs,
    compare_scalar_curves,
    load_jsonl,
)


def _runtime_path(case: dict[str, Any]) -> Path:
    """Return the model-integration runtime evidence directory for one row."""
    return Path(case["evidence_dir"]) / "cases" / "runtime"


def _required_evidence(case: dict[str, Any]) -> list[str]:
    """List missing per-step VLM evidence before comparing numbers."""
    root = _runtime_path(case)
    missing = []
    if not (Path(case["evidence_dir"]) / "trainer_config.resolved.json").is_file():
        missing.append("trainer_config.resolved.json")
    load_report = f"after_load_{Path(case['checkpoint']).name}_rank0.json"
    if not (Path(case["evidence_dir"]) / "checkpoint" / load_report).is_file():
        missing.append(f"checkpoint/{load_report}")
    if not (root / "metrics.jsonl").is_file():
        missing.append("metrics.jsonl")
    for step in case["measured_steps"]:
        for stem in ("modality_gradients", "optimizer_layout"):
            name = f"{stem}_step{step}.json"
            if not (root / name).is_file():
                missing.append(name)
    return missing


def _max_delta(result: dict[str, Any], metric: str) -> float | None:
    """Return the largest recorded absolute scalar error."""
    rows = result.get("steps") or ()
    return max((row[metric] for row in rows), default=None)


def _identity_check(case: dict[str, Any], parent: dict[str, Any],
                    baseline_rows: list[dict[str, Any]],
                    candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Require the same warm-start and exact per-step input and schedule identity."""
    mismatches = []
    baseline_config = json.loads((Path(parent["evidence_dir"]) / "trainer_config.resolved.json").read_text())
    candidate_config = json.loads((Path(case["evidence_dir"]) / "trainer_config.resolved.json").read_text())
    expected_checkpoint = str(Path(parent["checkpoint"]).resolve())
    for name, config in ((parent["case_id"], baseline_config), (case["case_id"], candidate_config)):
        actual = str(Path(config["checkpoint"]["restore_from"]).resolve())
        if actual != expected_checkpoint:
            mismatches.append({"case_id": name, "field": "checkpoint.restore_from",
                               "expected": expected_checkpoint, "actual": actual})
    for item in (parent, case):
        report_name = f"after_load_{Path(item['checkpoint']).name}_rank0.json"
        report_path = Path(item["evidence_dir"]) / "checkpoint" / report_name
        load_report = json.loads(report_path.read_text())
        actual = str(Path(load_report["checkpoint_path"]).resolve())
        if actual != expected_checkpoint:
            mismatches.append({"case_id": item["case_id"], "field": "actual_checkpoint_load",
                               "expected": expected_checkpoint, "actual": actual})
    if str(Path(case["checkpoint"]).resolve()) != expected_checkpoint:
        mismatches.append({"case_id": case["case_id"], "field": "ledger.checkpoint"})
    for baseline, candidate in zip(baseline_rows, candidate_rows):
        for field in ("step", "global_input_sha256", "lr"):
            if baseline.get(field) != candidate.get(field):
                mismatches.append({"step": baseline.get("step"), "field": field,
                                   "baseline": baseline.get(field), "candidate": candidate.get(field)})
    return {"status": "FAIL" if mismatches else "PASS", "mismatches": mismatches}


def _compare_case(case: dict[str, Any], parent: dict[str, Any],
                  acceptance: dict[str, Any]) -> dict[str, Any]:
    """Compare one VLM candidate to its declared same-workload parent."""
    baseline = Path(parent["evidence_dir"])
    candidate = Path(case["evidence_dir"])
    baseline_rows = load_jsonl(_runtime_path(parent) / "metrics.jsonl")
    candidate_rows = load_jsonl(_runtime_path(case) / "metrics.jsonl")
    strict = compare_scalar_curves(baseline_rows, candidate_rows, acceptance["strict"])
    bf16 = compare_scalar_curves(baseline_rows, candidate_rows,
                                 acceptance["bf16_cross_layout"])
    preflight = compare_preflight_configs(baseline, candidate)
    identity = _identity_check(case, parent, baseline_rows, candidate_rows)
    parameters = (compare_parameter_probes(baseline, candidate,
                                           acceptance["parameters"],
                                           compare_optimizer_states=False)
                  if bf16["status"] != "PASS" else None)
    checks = (bf16["status"], preflight["status"], identity["status"])
    status = "FAIL" if "FAIL" in checks else "BLOCKED" if "BLOCKED" in checks else "PASS"
    scalar_failed_steps = [row["step"] for row in bf16.get("steps", ()) if row["status"] != "PASS"]
    parameter_failures = ([value for value in parameters.get("values", ()) if value["status"] != "PASS"]
                          if parameters is not None else [])
    failed_steps = scalar_failed_steps
    first_parameter_failure = (min(parameter_failures, key=lambda value: value["key"])
                               if parameter_failures else None)
    failed_checks = [name for name, result in (
        ("bf16_scalar", bf16), ("preflight", preflight), ("identity", identity),
    ) if result["status"] != "PASS"]
    return {"status": status, "strict_scalar_status": strict["status"],
            "bf16_scalar_status": bf16["status"], "preflight": preflight,
            "identity": identity,
            "parameter_diagnostic": parameters, "strict_scalar": strict, "bf16_scalar": bf16,
            "max_loss_abs": _max_delta(bf16, "loss_max_abs"),
            "max_norm_abs": _max_delta(bf16, "norm_max_abs"),
            "failed_steps": failed_steps, "first_failed_step": failed_steps[0] if failed_steps else None,
            "failed_checks": failed_checks, "first_parameter_failure": first_parameter_failure}


def _load_ledger_context(ledger_path: Path) -> dict[str, Any]:
    """Load the frozen ledger, execution record, and current-input freshness."""
    ledger_bytes = ledger_path.read_bytes()
    ledger = json.loads(ledger_bytes)
    ledger_sha256 = hashlib.sha256(ledger_bytes).hexdigest()
    execution_path = ledger_path.with_name("execution.json")
    execution = json.loads(execution_path.read_text(encoding="utf-8")) if execution_path.exists() else {}
    repository = Path(ledger["source_checkout"]).resolve()
    current_source_sha256 = _source_sha256(repository, ledger["cases"])
    asset_names_match = set(ledger.get("asset_sha256", {})) == set(ledger.get("asset_paths", {}))
    changed_assets = _changed_assets(ledger) if asset_names_match else ["asset manifest"]
    current_inputs_ok = (bool(ledger.get("source_sha256"))
                         and bool(ledger.get("asset_sha256"))
                         and current_source_sha256 == ledger["source_sha256"]
                         and not changed_assets)
    return {"ledger": ledger, "ledger_sha256": ledger_sha256, "execution": execution,
            "current_inputs_ok": current_inputs_ok, "changed_assets": changed_assets}


def _judge_case(case: dict[str, Any], context: dict[str, Any],
                by_name: dict[str, Any], ready: dict[str, bool]) -> dict[str, Any]:
    """Judge one declared case against its evidence and comparison parent."""
    ledger = context["ledger"]
    name = case["case_id"]
    rows = load_jsonl(_runtime_path(case) / "metrics.jsonl")
    steps = [row.get("step") for row in rows]
    missing = _required_evidence(case)
    run = context["execution"].get(name, {})
    provenance_ok = (
        run.get("status") == "PASS" and run.get("source_sha256") == ledger.get("source_sha256")
        and run.get("ledger_sha256") == context["ledger_sha256"] and context["current_inputs_ok"]
    )
    ready[name] = steps == case["measured_steps"] and not missing and provenance_ok
    if run.get("status") in ("FAIL", "BLOCKED"):
        return {"status": run["status"], "execution": run,
                "missing_evidence": missing}
    if steps != case["measured_steps"] or missing:
        return {"status": "INCOMPLETE", "observed_steps": steps,
                "expected_steps": case["measured_steps"],
                "missing_evidence": missing}
    if not provenance_ok:
        return {"status": "BLOCKED", "reason": "execution provenance or current inputs changed",
                "observed_steps": steps, "changed_assets": context["changed_assets"]}
    if case["compare_to"] is None:
        return {"status": "PASS", "role": "panel_baseline", "steps": steps}
    parent = by_name[case["compare_to"]]
    if not ready.get(parent["case_id"], False):
        return {"status": "BLOCKED", "reason": "comparison parent is unavailable"}
    return _compare_case(case, parent, ledger["acceptance"])


def _judge_pairs(ledger: dict[str, Any], by_name: dict[str, Any],
                 ready: dict[str, bool]) -> dict[str, Any]:
    """Judge every declared extra pair comparison."""
    pair_results = {}
    for pair in ledger.get("pair_checks", []):
        left = pair["case_id"]
        right = pair["compare_to"]
        pair_name = f"{left}__vs__{right}"
        if left not in by_name or right not in by_name:
            pair_results[pair_name] = {"status": "BLOCKED", "reason": "unknown case ID"}
        elif not ready[left] or not ready[right]:
            pair_results[pair_name] = {"status": "INCOMPLETE", "reason": "pair evidence is unavailable"}
        else:
            pair_results[pair_name] = _compare_case(by_name[left], by_name[right], ledger["acceptance"])
    return pair_results


def compare_matrix(ledger_path: Path) -> dict[str, Any]:
    """Return an auditable result, preserving incomplete and failed rows.

    Args:
        ledger_path: Declared matrix cases and their evidence locations.
    """
    context = _load_ledger_context(ledger_path)
    ledger = context["ledger"]
    by_name = {case["case_id"]: case for case in ledger["cases"]}
    results = {}
    ready = {}
    for case in ledger["cases"]:
        results[case["case_id"]] = _judge_case(case, context, by_name, ready)
    pair_results = _judge_pairs(ledger, by_name, ready)
    statuses = [result["status"] for result in (*results.values(), *pair_results.values())]
    status = "FAIL" if "FAIL" in statuses else (
        "INCOMPLETE" if "INCOMPLETE" in statuses else "BLOCKED" if "BLOCKED" in statuses else "PASS"
    )
    counts = {value: statuses.count(value) for value in ("PASS", "FAIL", "BLOCKED", "INCOMPLETE")}
    return {"status": status, "case_count": len(results), "pair_count": len(pair_results),
            "counts": counts, "acceptance": ledger["acceptance"], "cases": results,
            "pair_checks": pair_results, "scope": ledger.get("scope"),
            "out_of_scope": ledger.get("out_of_scope", []),
            "excluded": ledger.get("excluded", [])}


def main() -> int:
    """Write JSON and compact Chinese coverage summary for the declared matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    arguments = parser.parse_args()
    ledger_path = arguments.ledger.resolve()
    result = compare_matrix(ledger_path)
    output = ledger_path.with_name("matrix-comparison.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [result["status"], "", "# VLM Encoder/SP 精度矩阵", "",
             f"计划 {result['case_count']} 个运行用例、{result['pair_count']} 项额外成对比较；"
             f"结果 {result['counts']}。", "",
             "| 用例 | 状态 | 严格标量 | BF16 标量 | 最大 loss 差 | 最大裁剪前 norm 差 | 首败步 | 标量失败诊断 |",
             "| --- | --- | --- | --- | ---: | ---: | ---: | --- |"]
    for name, row in result["cases"].items():
        lines.append(f"| `{name}` | {row['status']} | {row.get('strict_scalar_status', '-') } | "
                     f"{row.get('bf16_scalar_status', '-')} | {row.get('max_loss_abs', '-')} | "
                     f"{row.get('max_norm_abs', '-')} | {row.get('first_failed_step', '-')} | "
                     f"{', '.join(row.get('failed_checks', ())) or '-'} |")
    if result["pair_checks"]:
        lines.extend(("", "## 额外成对比较", "", "| 比较 | 状态 | 最大 loss 差 | 最大裁剪前 norm 差 |",
                      "| --- | --- | ---: | ---: |"))
        for name, row in result["pair_checks"].items():
            lines.append(f"| `{name}` | {row['status']} | {row.get('max_loss_abs', '-')} | "
                         f"{row.get('max_norm_abs', '-')} |")
    lines.extend(("", "正式判定只使用十步 loss、裁剪前 norm、配置和输入身份；参数探针仅在标量不对齐时诊断。",
                  "`INCOMPLETE` 表示计划用例缺少 10 步或必要证据；不会计为通过。",
                  "对照关系、固定阈值与排除项见 `matrix-ledger.json`；逐步结果见 `matrix-comparison.json`。"))
    ledger_path.with_name("matrix-summary.zh-CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
