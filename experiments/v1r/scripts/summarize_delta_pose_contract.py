#!/usr/bin/env python3
"""Create upload-safe V1-R.2J delta-pose contract reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record.get(key)
        for key in (
            "layout",
            "demo_key",
            "initial_state_match",
            "initial_state_max_abs_error",
            "steps_executed",
            "success",
            "first_success_action_index",
            "first_state_divergence_action_index",
            "state_divergence_atol",
            "roundtrip_max_abs_error",
            "roundtrip_passed",
            "gripper_sign_preserved",
            "controller_use_delta",
            "contact_observed",
            "grasp_observed",
        )
    }


def build_summary(result: dict[str, Any], result_path: Path) -> dict[str, Any]:
    records = [compact_record(record) for record in result["records"]]
    per_layout = {}
    for layout in (1, 2, 3, 4):
        items = [record for record in records if record["layout"] == layout]
        per_layout[str(layout)] = {
            "checked": len(items),
            "initial_state_matches": sum(item["initial_state_match"] for item in items),
            "normalization_roundtrip_passed": sum(
                item["roundtrip_passed"] for item in items
            ),
            "gripper_sign_preserved": sum(
                item["gripper_sign_preserved"] for item in items
            ),
            "official_delta_runtime": sum(
                item["controller_use_delta"] for item in items
            ),
            "execution_successes": sum(item["success"] for item in items),
        }
    failures = [
        {"layout": item["layout"], "demo_key": item["demo_key"]}
        for item in records
        if not item["success"]
    ]
    return {
        "stage": result["stage"],
        "latest_completed_stage": result["stage"],
        "status": "completed_delta_pose_contract_failed",
        "validation": result["validation"],
        "formal_threshold": result["formal_threshold"],
        "runtime": result["runtime"],
        "normalization": result["normalization"],
        "gates": result["gates"],
        "per_layout": per_layout,
        "failures": failures,
        "decision": {
            "case": "D",
            "selected_label_contract": None,
            "b0_b1_training_authorized": False,
            "seed0_training_authorized": False,
            "confirm_rollouts_authorized": False,
            "v2_formal_experiment_authorized": False,
            "v3_formal_experiment_authorized": False,
            "next_action": "diagnose_delta_pose_contract_failure_on_same_20_sequential_demos_without_training",
        },
        "invariants": {
            "same_20_v1r_2i_demonstrations": True,
            "accepted_demonstrations": 20,
            "initial_state_matches": sum(item["initial_state_match"] for item in records),
            "normalization_roundtrip_passed": sum(
                item["roundtrip_passed"] for item in records
            ),
            "gripper_sign_preserved": sum(
                item["gripper_sign_preserved"] for item in records
            ),
            "official_delta_runtime": sum(
                item["controller_use_delta"] for item in records
            ),
            "hdf5_action_prefix_matches": 20,
            "policy_training_or_inference": False,
            "intermediate_state_restore": False,
            "simulator_exceptions": 0,
        },
        "source": result["source"],
        "local_evidence": {
            "result": relative_path(result_path),
            "result_size_bytes": result_path.stat().st_size,
            "result_sha256": file_sha256(result_path),
            "telemetry_location": "local result records[*].telemetry; not uploaded",
        },
        "records": records,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# V1-R.2J Delta-Pose Contract Validation",
        "",
        "## Scope",
        "",
        "- Same 20 accepted V1-R.2I sequential-success demonstrations; no policy training or inference.",
        "- Runtime: robosuite `1.4.1`, MuJoCo `3.3.5`, official Point Bridge suite, delta OSC, 20 Hz.",
        "- Strict execution gate: `20/20`; normalization, gripper sign, runtime mode, and task execution are separate checks.",
        "- The raw 7-D actions are the commands issued during the continuous demonstrations; no intermediate state is restored.",
        "",
        "## Gates",
        "",
        "| Gate | Checked | Passed | Status |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, gate in summary["gates"].items():
        passed = gate["passed"]
        lines.append(
            f"| {name} | {gate['checked']} | {passed} | "
            f"{'passed' if passed == gate['checked'] else 'failed'} |"
        )
    lines.extend(
        [
            "",
            "## Per Layout",
            "",
            "| Layout | Initial state | Roundtrip | Gripper sign | Delta runtime | Execution |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for layout, item in summary["per_layout"].items():
        lines.append(
            f"| {layout} | {item['initial_state_matches']}/5 | "
            f"{item['normalization_roundtrip_passed']}/5 | "
            f"{item['gripper_sign_preserved']}/5 | "
            f"{item['official_delta_runtime']}/5 | "
            f"{item['execution_successes']}/5 |"
        )
    lines.extend(
        [
            "",
            "## Failures",
            "",
            "The strict execution failures were: "
            + ", ".join(
                f"layout {item['layout']} `{item['demo_key']}`"
                for item in summary["failures"]
            )
            + ".",
            "",
            "## Decision",
            "",
            "The delta-pose fallback is executable in the official runtime for 17/20 demonstrations, "
            "but it does not satisfy the predeclared 20/20 contract. It is retained as an explicit "
            "unvalidated interface patch; no label contract is selected and training remains blocked.",
            "",
            "## Invariants",
            "",
            f"- HDF5 action-prefix matches: `{summary['invariants']['hdf5_action_prefix_matches']}/20`.",
            f"- Initial-state matches: `{summary['invariants']['initial_state_matches']}/20`.",
            f"- Normalization roundtrips: `{summary['invariants']['normalization_roundtrip_passed']}/20`.",
            f"- Gripper signs preserved: `{summary['invariants']['gripper_sign_preserved']}/20`.",
            f"- Official delta runtime: `{summary['invariants']['official_delta_runtime']}/20`.",
            "- Policy training or inference: `false`.",
            "- Intermediate state restoration: `false`.",
            "- Simulator exceptions: `0`.",
            "",
            "## Local Evidence",
            "",
            f"- `{summary['local_evidence']['result']}`: `{summary['local_evidence']['result_sha256']}` "
            f"({summary['local_evidence']['result_size_bytes']} bytes; ignored, not uploaded).",
            "- Full per-step telemetry remains local; this tracked report contains compact summaries only.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gate(summary: dict[str, Any], path: Path) -> None:
    gates = summary["gates"]
    lines = [
        "name: V1-R.2J delta-pose fallback contract validation",
        "stage: V1-R.2J",
        "status: completed_delta_pose_contract_failed",
        "decision_case: D",
        "formal_threshold: 20/20",
        "formal_runtime:",
        "  robosuite: 1.4.1",
        "  mujoco: 3.3.5",
        "  control_freq_hz: 20",
        "  action_mode: delta_pose",
        "  action_shape: [7]",
        "  control_delta: true",
        "gates:",
    ]
    for name, gate in gates.items():
        lines.append(
            f"  {name}: {{status: {'passed' if gate['passed'] == gate['checked'] else 'failed'}, "
            f"checked: {gate['checked']}, passed: {gate['passed']}}}"
        )
    lines.extend(
        [
            "selected_label_contract: null",
            "b0_b1_training_authorized: false",
            "seed0_training_authorized: false",
            "confirm_rollouts_authorized: false",
            "v2_formal_experiment_authorized: false",
            "v3_formal_experiment_authorized: false",
            "next_action: diagnose_delta_pose_contract_failure_on_same_20_sequential_demos_without_training",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_artifact_index(result_path: Path, index_path: Path) -> None:
    row = {
        "local_path": relative_path(result_path),
        "size_bytes": result_path.stat().st_size,
        "type": "generated_runtime_result",
        "description": "V1-R.2J delta-pose normalization, official runtime, and execution replay telemetry",
        "reason": "Raw runtime output remains under gitignored outputs; compact report is tracked",
        "sha256": file_sha256(result_path),
    }
    existing = []
    if index_path.exists():
        with index_path.open("r", encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
    existing = [item for item in existing if item.get("local_path") != row["local_path"]]
    existing.append({key: str(value) for key, value in row.items()})
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("local_path", "size_bytes", "type", "description", "reason", "sha256"),
        )
        writer.writeheader()
        writer.writerows(existing)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=Path,
        default=ROOT / "outputs" / "v1r" / "delta_pose_contract_2j" / "results.json",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "experiments" / "v1r" / "reports" / "delta_pose_contract.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "experiments" / "v1r" / "reports" / "delta_pose_contract.md",
    )
    parser.add_argument(
        "--gate-output",
        type=Path,
        default=ROOT / "experiments" / "v1r" / "reports" / "v1r_2j_delta_pose_contract.yaml",
    )
    parser.add_argument(
        "--artifact-index",
        type=Path,
        default=ROOT
        / "experiments"
        / "v1r"
        / "manifests"
        / "executable_absolute_label_contract_artifact_index.csv",
    )
    args = parser.parse_args()
    result_path = args.result.resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    summary = build_summary(result, result_path)
    args.json_output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.json_output.resolve().write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, args.markdown_output.resolve())
    write_gate(summary, args.gate_output.resolve())
    update_artifact_index(result_path, args.artifact_index.resolve())
    print(json.dumps({"stage": summary["stage"], "gates": summary["gates"], "failures": summary["failures"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
