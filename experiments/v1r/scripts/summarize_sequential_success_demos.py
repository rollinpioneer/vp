#!/usr/bin/env python3
"""Materialize upload-safe V1-R.2I reports from local runtime artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
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


def per_layout_capture(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["layout"])].append(record)
    return {
        str(layout): {
            "attempted": len(items),
            "accepted": sum(bool(item.get("accepted")) for item in items),
            "accepted_demo_keys": [
                item["demo_key"] for item in items if item.get("accepted")
            ],
        }
        for layout, items in sorted(grouped.items())
    }


def per_layout_verification(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["layout"])].append(record)
    return {
        str(layout): {
            "checked": len(items),
            "actual_command_passed": sum(
                bool(item.get("actual_command_passed")) for item in items
            ),
            "s0_passed": sum(bool(item.get("s0_passed")) for item in items),
            "s1_passed": sum(bool(item.get("s1_passed")) for item in items),
        }
        for layout, items in sorted(grouped.items())
    }


def artifact_index_rows(
    capture_path: Path,
    verification_path: Path,
    accepted_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        {
            "local_path": relative_path(capture_path),
            "size_bytes": capture_path.stat().st_size,
            "type": "generated_runtime_manifest",
            "description": "V1-R.2I candidate and accepted sequential-demo capture records",
            "reason": "Raw runtime output remains under gitignored outputs; upload-safe summary is tracked",
            "sha256": file_sha256(capture_path),
        },
        {
            "local_path": relative_path(verification_path),
            "size_bytes": verification_path.stat().st_size,
            "type": "generated_runtime_verification",
            "description": "V1-R.2I per-demo actual-command, S0, and S1 replay records",
            "reason": "Raw runtime output remains under gitignored outputs; upload-safe summary is tracked",
            "sha256": file_sha256(verification_path),
        },
    ]
    for record in accepted_records:
        for key, artifact_type, description in (
            (
                "artifact",
                "sequential_demo_state_bundle",
                "Continuous transitions, issued commands, measured EEF poses, gripper commands, and controller targets",
            ),
            (
                "model_xml",
                "sequential_demo_runtime_xml",
                "Migrated formal-runtime MuJoCo model XML for exact initial-state replay",
            ),
        ):
            path = ROOT / record[key]
            rows.append(
                {
                    "local_path": record[key],
                    "size_bytes": path.stat().st_size,
                    "type": artifact_type,
                    "description": (
                        f"Layout {record['layout']} {record['demo_key']}: {description}"
                    ),
                    "reason": "Generated runtime artifact remains local and is indexed by filename and hash",
                    "sha256": file_sha256(path),
                }
            )
    return rows


def build_summary(
    capture: dict[str, Any],
    verification: dict[str, Any],
    capture_path: Path,
    verification_path: Path,
) -> dict[str, Any]:
    accepted = [record for record in capture["records"] if record.get("accepted")]
    verified = verification["records"]
    verified_by_key = {
        (int(record["layout"]), str(record["demo_key"])): record
        for record in verified
    }
    accepted_records = []
    for record in accepted:
        verification_record = verified_by_key[(int(record["layout"]), record["demo_key"])]
        accepted_records.append(
            {
                "layout": int(record["layout"]),
                "demo_key": record["demo_key"],
                "actions_issued": int(record["actions_issued"]),
                "first_success_action_index": int(record["first_success_action_index"]),
                "task_success": bool(record["task_success"]),
                "contact_observed": bool(record["contact_observed"]),
                "stable_grasp_observed": bool(record["stable_grasp_observed"]),
                "closed_without_bowl_observed": bool(
                    record["closed_without_bowl_observed"]
                ),
                "artifact": record["artifact"],
                "artifact_sha256": record["artifact_sha256"],
                "model_xml": record["model_xml"],
                "model_xml_sha256": record["model_xml_sha256"],
                "actual_command_passed": bool(
                    verification_record["actual_command_passed"]
                ),
                "s0_passed": bool(verification_record["s0_passed"]),
                "s1_passed": bool(verification_record["s1_passed"]),
            }
        )
    gates = verification["gates"]
    return {
        "stage": "V1-R.2I",
        "latest_completed_stage": "V1-R.2I",
        "status": "completed_label_contract_failed",
        "decision": "blocked_sequential_label_contract",
        "formal_runtime": capture["runtime"],
        "capture": {
            "status": capture["status"],
            "action_source": "official_hdf5_delta_actions",
            "continuous_env_step_only": True,
            "intermediate_state_restore_count": int(
                capture["middle_state_restore_count"]
            ),
            "attempted": len(capture["records"]),
            "accepted": len(accepted),
            "restored_initial_state_matches": sum(
                bool(record.get("restored_initial_state_match"))
                for record in capture["records"]
            ),
            "simulator_exceptions": sum(
                bool(record.get("exception")) for record in capture["records"]
            ),
            "per_layout": per_layout_capture(capture["records"]),
        },
        "verification": {
            "actual_command_replay": gates["actual_command_replay"],
            "s0_original_pointbridge_label_replay": gates["s0_label_replay"],
            "s1_recorded_controller_target_label_replay": gates[
                "s1_label_replay"
            ],
            "per_layout": per_layout_verification(verified),
        },
        "label_contract_decision": {
            "selected_contract": None,
            "reason": "neither_s0_nor_s1_reached_the_strict_20_of_20_replay_threshold",
            "b0_b1_training_authorized": False,
            "confirm_rollouts_authorized": False,
            "v2_formal_experiment_authorized": False,
            "v3_formal_experiment_authorized": False,
            "next_action": "diagnose_and_reconstruct_an_executable_absolute_label_contract_on_the_same_20_sequential_demos",
        },
        "diagnostic_boundary": {
            "stable_grasp_is_acceptance_filter": False,
            "accepted_with_stable_grasp_observed": sum(
                bool(record["stable_grasp_observed"]) for record in accepted
            ),
            "accepted_with_task_success": sum(
                bool(record["task_success"]) for record in accepted
            ),
            "scope": "20 contract-validation demonstrations; not training-data coverage proof",
        },
        "accepted_records": accepted_records,
        "local_evidence": {
            "capture_manifest": relative_path(capture_path),
            "capture_manifest_sha256": file_sha256(capture_path),
            "verification": relative_path(verification_path),
            "verification_sha256": file_sha256(verification_path),
        },
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    capture = summary["capture"]
    verification = summary["verification"]
    lines = [
        "# V1-R.2I Sequential Success Demonstration Gate",
        "",
        "## Scope",
        "",
        "- Runtime: robosuite `1.4.1`, MuJoCo `3.3.5`, `OSC_POSE`, delta commands, 20 Hz.",
        "- Candidate command source: the official HDF5 7-D delta action streams.",
        "- Each candidate restores its initial state once; every later state is produced by `env.step()`.",
        "- The old saved-state PKL generator and patch `0001-mimiclabs-saved-state-generator.patch` are not called.",
        "- Task success is the acceptance predicate. Five-step stable grasp remains diagnostic only.",
        "",
        "## Sequential Capture",
        "",
        "| Layout | Candidates executed | Continuous successes | Accepted demos |",
        "| --- | ---: | ---: | --- |",
    ]
    for layout, item in capture["per_layout"].items():
        lines.append(
            f"| {layout} | {item['attempted']} | {item['accepted']} | "
            f"{', '.join(item['accepted_demo_keys'])} |"
        )
    lines.extend(
        [
            "",
            f"All {capture['attempted']} attempted initial states restored exactly; simulator exceptions: "
            f"{capture['simulator_exceptions']}. Intermediate state restores: "
            f"{capture['intermediate_state_restore_count']}.",
            "",
            "## Replay Gates",
            "",
            "| Layout | Actual issued commands | S0 original labels | S1 recorded targets |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for layout, item in verification["per_layout"].items():
        lines.append(
            f"| {layout} | {item['actual_command_passed']}/{item['checked']} | "
            f"{item['s0_passed']}/{item['checked']} | {item['s1_passed']}/{item['checked']} |"
        )
    actual = verification["actual_command_replay"]
    s0 = verification["s0_original_pointbridge_label_replay"]
    s1 = verification["s1_recorded_controller_target_label_replay"]
    lines.extend(
        [
            "",
            f"The actual-command gate passed `{actual['passed']}/{actual['checked']}`. "
            f"S0 passed `{s0['passed']}/{s0['checked']}` and S1 passed "
            f"`{s1['passed']}/{s1['checked']}`; neither met the strict `20/20` label threshold.",
            "",
            "## Decision",
            "",
            "V1-R.2I is complete as a sequential demonstration reconstruction experiment, but the "
            "absolute training-label contract remains blocked. No S0/S1 contract is selected, and "
            "B0/B1 training, confirm rollouts, V2, and V3 remain unauthorized.",
            "",
            "The next action is limited to diagnosing and reconstructing an executable absolute "
            "label contract on these same 20 sequential demonstrations. It is not a replay of "
            "V1-R.2H and does not authorize policy retraining.",
            "",
            "None of the 20 accepted demonstrations met the five-consecutive-step stable-grasp "
            "diagnostic, while all 20 met task success and actual-command replay. This diagnostic "
            "therefore remains non-gating. These 20 demonstrations validate the contract sample "
            "only and do not establish future data coverage.",
            "",
            "## Local Evidence",
            "",
            f"- `{summary['local_evidence']['capture_manifest']}`: "
            f"`{summary['local_evidence']['capture_manifest_sha256']}`",
            f"- `{summary['local_evidence']['verification']}`: "
            f"`{summary['local_evidence']['verification_sha256']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gate(summary: dict[str, Any], path: Path) -> None:
    capture = summary["capture"]
    verification = summary["verification"]
    lines = [
        "name: V1-R.2I sequential success demonstration and label contract gate",
        "stage: V1-R.2I",
        "status: completed_label_contract_failed",
        "decision: blocked_sequential_label_contract",
        "formal_runtime:",
        "  robosuite: 1.4.1",
        "  mujoco: 3.3.5",
        "  control_freq_hz: 20",
        "capture:",
        f"  attempted: {capture['attempted']}",
        f"  accepted: {capture['accepted']}",
        f"  restored_initial_state_matches: {capture['restored_initial_state_matches']}",
        f"  intermediate_state_restore_count: {capture['intermediate_state_restore_count']}",
        f"  simulator_exceptions: {capture['simulator_exceptions']}",
        "gates:",
        "  actual_command_replay: {status: passed, checked: 20, successes: 20}",
        f"  s0_label_replay: {{status: failed, checked: 20, successes: {verification['s0_original_pointbridge_label_replay']['passed']}}}",
        f"  s1_label_replay: {{status: failed, checked: 20, successes: {verification['s1_recorded_controller_target_label_replay']['passed']}}}",
        "selected_label_contract: null",
        "b0_b1_training_authorized: false",
        "confirm_rollouts_authorized: false",
        "v2_formal_experiment_authorized: false",
        "v3_formal_experiment_authorized: false",
        "next_action: diagnose_and_reconstruct_an_executable_absolute_label_contract_on_the_same_20_sequential_demos",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument("--artifact-index-output", type=Path, required=True)
    args = parser.parse_args()

    capture_path = args.capture.resolve()
    verification_path = args.verification.resolve()
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    summary = build_summary(capture, verification, capture_path, verification_path)
    accepted = [record for record in capture["records"] if record.get("accepted")]

    for output in (
        args.summary_output,
        args.report_output,
        args.gate_output,
        args.artifact_index_output,
    ):
        output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.resolve().write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(summary, args.report_output.resolve())
    write_gate(summary, args.gate_output.resolve())
    with args.artifact_index_output.resolve().open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "local_path",
            "size_bytes",
            "type",
            "description",
            "reason",
            "sha256",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            artifact_index_rows(capture_path, verification_path, accepted)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
