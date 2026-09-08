#!/usr/bin/env python3
"""Create upload-safe V1-R.2J reports from the ignored runtime result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CONTRACTS = ("s0_old", "s0_transition", "s1_pb", "s1_world")
CONTRACT_LABELS = {
    "s0_old": "S0-old",
    "s0_transition": "S0-transition",
    "s1_pb": "S1-PB",
    "s1_world": "S1-world",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def compact_replay(replay: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "initial_state_match",
        "steps_executed",
        "base_steps",
        "tail_steps",
        "base_success",
        "success",
        "first_success_action_index",
        "first_state_divergence_action_index",
        "first_exact_state_divergence_action_index",
        "state_divergence_atol",
        "first_contact_action_index",
        "first_close_command_action_index",
        "first_grasp_action_index",
        "contact_observed",
        "grasp_observed",
        "failure_stage",
        "max_controller_goal_vs_captured_translation_error_m",
        "max_controller_goal_vs_captured_rotation_error_rad",
    )
    return {key: replay.get(key) for key in keep}


def build_summary(result: dict[str, Any], result_path: Path) -> dict[str, Any]:
    records = []
    for record in result["records"]:
        compact_contracts = {}
        for contract in CONTRACTS:
            item = record["contracts"][contract]
            compact_contracts[contract] = {
                "base": compact_replay(item["base"]),
                "post_success_tail_diagnostic": compact_replay(
                    item["post_success_tail_diagnostic"]
                ),
                "tail_prefix_action_max_abs_difference": item[
                    "tail_prefix_action_max_abs_difference"
                ],
                "base_actions_sha256": item["base_actions_sha256"],
                "tail_actions_sha256": item["tail_actions_sha256"],
            }
        records.append(
            {
                "layout": record["layout"],
                "demo_key": record["demo_key"],
                "base_action_count": record["base_action_count"],
                "source_action_count": record["source_action_count"],
                "tail_plan": record["tail_plan"],
                "source_action_prefix_match": record["source_action_prefix_match"],
                "delta_reference": record["delta_reference"],
                "contracts": compact_contracts,
                "exception": record["exception"],
            }
        )
    return {
        "stage": result["stage"],
        "latest_completed_stage": result["stage"],
        "status": result["status"],
        "formal_threshold": result["formal_threshold"],
        "runtime": result["runtime"],
        "contracts": result["contracts"],
        "post_success_tail_diagnostic": result["post_success_tail_diagnostic"],
        "state_divergence_atol": result["state_divergence_atol"],
        "source": result["source"],
        "checked": result["checked"],
        "base_passed": result["base_passed"],
        "post_success_tail_diagnostic_passed": result[
            "post_success_tail_diagnostic_passed"
        ],
        "per_layout": result["per_layout"],
        "invariants": result["invariants"],
        "decision": result["decision"],
        "records": records,
        "local_evidence": {
            "result": relative_path(result_path),
            "result_size_bytes": result_path.stat().st_size,
            "result_sha256": file_sha256(result_path),
            "step_telemetry_location": "records[*].contracts[*].{base,post_success_tail_diagnostic}.telemetry",
        },
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# V1-R.2J Executable Absolute-Action Label Contract",
        "",
        "## Scope",
        "",
        "- Same 20 V1-R.2I sequential-success demonstrations; no policy training or inference.",
        "- Runtime: robosuite `1.4.1`, MuJoCo `3.3.5`, `OSC_POSE`, 20 Hz.",
        "- Strict gate: `20/20`; base and post-success-tail results are separate.",
        "- Every absolute replay records its input action, controller goal, target error, delta-reference state error, first divergence, contact, close command, grasp diagnostic, success, and failure stage.",
        "",
        "## Base Gate",
        "",
        "| Layout | S0-old | S0-transition | S1-PB | S1-world |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for layout, item in summary["per_layout"].items():
        counts = item["base_passed"]
        lines.append(
            f"| {layout} | {counts['s0_old']}/5 | {counts['s0_transition']}/5 | "
            f"{counts['s1_pb']}/5 | {counts['s1_world']}/5 |"
        )
    base = summary["base_passed"]
    lines.extend(
        [
            f"| **Total** | **{base['s0_old']}/20** | **{base['s0_transition']}/20** | "
            f"**{base['s1_pb']}/20** | **{base['s1_world']}/20** |",
            "",
            "## Post-Success Tail Diagnostic",
            "",
            "The predeclared rule appends ten real remaining HDF5 delta transitions when available. "
            "When fewer than ten remain, it appends five repeats of each contract's final absolute target. "
            "The rule is selected from source length before replay outcomes are known.",
            "",
            "| Layout | S0-old | S0-transition | S1-PB | S1-world |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for layout, item in summary["per_layout"].items():
        counts = item["post_success_tail_diagnostic_passed"]
        lines.append(
            f"| {layout} | {counts['s0_old']}/5 | {counts['s0_transition']}/5 | "
            f"{counts['s1_pb']}/5 | {counts['s1_world']}/5 |"
        )
    tail = summary["post_success_tail_diagnostic_passed"]
    lines.extend(
        [
            f"| **Total** | **{tail['s0_old']}/20** | **{tail['s0_transition']}/20** | "
            f"**{tail['s1_pb']}/20** | **{tail['s1_world']}/20** |",
            "",
            "## Invariants",
            "",
            f"- Same demonstrations: `{summary['invariants']['same_20_v1r_2i_demonstrations']}`.",
            f"- HDF5 action-prefix matches: `{summary['invariants']['source_action_prefix_matches']}/20`.",
            f"- Exact initial-state matches: `{summary['invariants']['delta_initial_state_matches']}/20`.",
            f"- Delta reference successes: `{summary['invariants']['delta_base_successes']}/20`.",
            f"- Exact regenerated base-state sequences: `{summary['invariants']['base_delta_state_exact_matches']}/20`.",
            f"- 2I S0-old/S1-PB outcomes reproduced: `{summary['invariants']['s0_old_and_s1_pb_v1r_2i_outcomes_reproduced']}`.",
            f"- Simulator exceptions: `{summary['invariants']['simulator_exceptions']}`.",
            "",
            "## Decision",
            "",
            f"Decision case: `{summary['decision']['case']}`.",
            f"Selected label contract: `{summary['decision']['selected_label_contract']}`.",
            f"Next action: `{summary['decision']['next_action']}`.",
            "",
            "B0/B1 training, confirm rollouts, V2, and V3 remain unauthorized. "
            "Seed-0 authorization, if present, applies only after rebuilding data with the frozen contract; it does not authorize confirm, V2, or V3.",
            "",
            "Five-step stable grasp remains diagnostic and is not substituted for task success.",
            "",
            "## Local Evidence",
            "",
            f"- `{summary['local_evidence']['result']}`: `{summary['local_evidence']['result_sha256']}` "
            f"({summary['local_evidence']['result_size_bytes']} bytes; ignored, not uploaded).",
            "- Per-step telemetry remains in that local result. The tracked JSON contains compact per-trajectory summaries only.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def write_gate(summary: dict[str, Any], path: Path) -> None:
    decision = summary["decision"]
    lines = [
        "name: V1-R.2J executable absolute-action label contract reconstruction",
        "stage: V1-R.2J",
        f"status: {summary['status']}",
        f"decision_case: {decision['case']}",
        "formal_threshold: 20/20",
        "formal_runtime:",
        "  robosuite: 1.4.1",
        "  mujoco: 3.3.5",
        "  control_freq_hz: 20",
        "base_gates:",
    ]
    for contract in CONTRACTS:
        passed = summary["base_passed"][contract]
        lines.append(
            f"  {contract}: {{status: {'passed' if passed == 20 else 'failed'}, checked: 20, successes: {passed}}}"
        )
    lines.append("post_success_tail_diagnostic:")
    for contract in CONTRACTS:
        passed = summary["post_success_tail_diagnostic_passed"][contract]
        lines.append(
            f"  {contract}: {{status: {'passed' if passed == 20 else 'failed'}, checked: 20, successes: {passed}}}"
        )
    lines.extend(
        [
            f"selected_label_contract: {yaml_scalar(decision['selected_label_contract'])}",
            f"uniform_tail_required: {yaml_scalar(decision['uniform_tail_required'])}",
            f"b0_b1_training_authorized: {yaml_scalar(decision['b0_b1_training_authorized'])}",
            f"seed0_training_authorized: {yaml_scalar(decision['seed0_training_authorized'])}",
            f"confirm_rollouts_authorized: {yaml_scalar(decision['confirm_rollouts_authorized'])}",
            f"v2_formal_experiment_authorized: {yaml_scalar(decision['v2_formal_experiment_authorized'])}",
            f"v3_formal_experiment_authorized: {yaml_scalar(decision['v3_formal_experiment_authorized'])}",
            f"next_action: {decision['next_action']}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def artifact_rows(result_path: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "local_path": relative_path(result_path),
            "size_bytes": result_path.stat().st_size,
            "type": "generated_runtime_result",
            "description": "V1-R.2J full per-step telemetry for four base and four tail-assisted absolute contracts",
            "reason": "Raw runtime output remains under gitignored outputs; compact reports are tracked",
            "sha256": file_sha256(result_path),
        }
    ]
    seen = set()
    for record in summary["records"]:
        reference = record["delta_reference"]
        if not reference:
            continue
        local_path = reference["artifact"]
        if local_path in seen:
            continue
        seen.add(local_path)
        path = ROOT / local_path
        rows.append(
            {
                "local_path": local_path,
                "size_bytes": path.stat().st_size,
                "type": "sequential_tail_reference_bundle",
                "description": f"Layout {record['layout']} {record['demo_key']}: regenerated continuous delta states and selected real tail targets",
                "reason": "Generated NPZ remains local and is indexed by filename and hash",
                "sha256": file_sha256(path),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=Path,
        default=ROOT
        / "outputs"
        / "v1r"
        / "executable_absolute_contracts_2j"
        / "results.json",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "experiments" / "v1r" / "reports" / "executable_absolute_label_contracts.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "experiments" / "v1r" / "reports" / "executable_absolute_label_contracts.md",
    )
    parser.add_argument(
        "--gate-output",
        type=Path,
        default=ROOT / "experiments" / "v1r" / "reports" / "v1r_2j_executable_absolute_label_contract.yaml",
    )
    parser.add_argument(
        "--artifact-index",
        type=Path,
        default=ROOT / "experiments" / "v1r" / "manifests" / "executable_absolute_label_contract_artifact_index.csv",
    )
    args = parser.parse_args()
    result_path = args.result.resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    summary = build_summary(result, result_path)
    args.json_output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.json_output.resolve().write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(summary, args.markdown_output.resolve())
    write_gate(summary, args.gate_output.resolve())
    rows = artifact_rows(result_path, summary)
    args.artifact_index.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.artifact_index.resolve().open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "local_path",
                "size_bytes",
                "type",
                "description",
                "reason",
                "sha256",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "stage": summary["stage"],
                "base_passed": summary["base_passed"],
                "tail_passed": summary["post_success_tail_diagnostic_passed"],
                "decision": summary["decision"],
                "artifact_rows": len(rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
