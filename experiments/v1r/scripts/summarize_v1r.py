#!/usr/bin/env python3
"""Apply the predeclared V1-R gate logic and write handoff artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GATES = ("clean_baseline_gate.json", "perception_gate.json", "visibility_scientific_gate.json", "frequency_camera_gate.json", "context_gate.json")
AUDITS = {
    "initial_state_pairing_audit": "initial_state_report.json",
    "runner_parity_audit": "runner_parity_report.json",
    "cpu_cuda_parity_audit": "cpu_cuda_parity.json",
    "expert_replay_audit": "expert_replay_report.json",
    "per_layout_failure_analysis": "layout_failure_report.json",
}
AUDIT_FIELDS = {
    "initial_state_pairing_audit": (
        "status",
        "clean_states",
        "runner_parity_states",
        "splits",
        "runtime_repeats",
        "seed_only_reproduction",
    ),
    "runner_parity_audit": (
        "status",
        "device",
        "scenarios",
        "passed_scenarios",
        "source_old_outcomes",
        "path_success",
    ),
    "cpu_cuda_parity_audit": (
        "status",
        "scenarios_requested",
        "deployment_device",
        "deployment_device_protocol_frozen",
        "formal_evaluation_device_constraint",
        "blockers",
    ),
    "expert_replay_audit": (
        "status",
        "checked",
        "passed",
        "per_layout",
        "restored_initial_state_matches",
        "failed_layouts",
        "failed_records_diverging_at_step_1",
        "failure_localization",
    ),
    "per_layout_failure_analysis": (
        "status",
        "rows",
        "layouts",
        "failure_stages",
        "termination_reasons",
        "rollout_csv_sha256",
    ),
}


def load_gate(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"status": "blocked", "blockers": [f"missing gate file: {path}"]}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_audit(name: str, payload: dict[str, object]) -> dict[str, object]:
    return {key: payload[key] for key in AUDIT_FIELDS[name] if key in payload}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=Path("experiments/v1r/reports"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/v1r"))
    args = parser.parse_args()
    gates = {name.removesuffix(".json"): load_gate(args.reports_dir / name) for name in GATES}
    audits = {
        name: summarize_audit(name, load_gate(args.reports_dir / filename))
        for name, filename in AUDITS.items()
    }
    for name, payload in gates.items():
        path = args.reports_dir / f"{name}.json"
        if not path.is_file():
            path.write_text(json.dumps({"stage": "V1-R", "status": "blocked", **payload}, indent=2) + "\n", encoding="utf-8")
    if not (args.reports_dir / "targeted_occlusion_report.md").is_file():
        (args.reports_dir / "targeted_occlusion_report.md").write_text("# V1-R.4 定向遮挡\n\n状态：`blocked`。缺少 clean reference rollout 和冻结的目标投影 mask schedule。\n", encoding="utf-8")
    if not (args.reports_dir / "context_alias_report.md").is_file():
        (args.reports_dir / "context_alias_report.md").write_text("# V1-R.6 上下文别名场景\n\n状态：`blocked`。尚未提供 LEFT_BLOCK/RIGHT_OPEN 等成对环境和动作证据。\n", encoding="utf-8")
    clean = gates["clean_baseline_gate"]
    visibility = gates["visibility_scientific_gate"]
    frequency = gates["frequency_camera_gate"]
    context = gates["context_gate"]
    if clean.get("status") != "passed":
        decision = "blocked_clean_baseline"
    elif visibility.get("status") != "passed":
        decision = "authorize_v3_fixed_budget_context_selection" if context.get("status") == "passed" else "no_go_current_vico_point_direction"
    elif frequency.get("status") == "passed":
        decision = "no_go_v2_high_rate_or_multiview_is_sufficient"
    else:
        decision = "authorize_v2_moving_object_point_belief"
    summary = {"stage": "V1-R.7", "decision": decision, "gates": gates, "audits": audits, "v2_formal_experiment_authorized": decision in {"authorize_v2_moving_object_point_belief", "authorize_v2_then_v3_sequentially"}, "v3_formal_experiment_authorized": decision in {"authorize_v3_fixed_budget_context_selection", "authorize_v2_then_v3_sequentially"}}
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "v1r_gate_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    audit_lines = "\n".join(
        f"- {name}: `{payload.get('status', 'blocked')}`"
        for name, payload in audits.items()
    )
    clean_rate = clean.get("pooled_success_rate", clean.get("success_rate"))
    (args.reports_dir / "v1r_final_report.md").write_text(
        "# V1-R 执行结果\n\n"
        f"- decision: `{decision}`\n"
        f"- clean baseline success rate: `{clean_rate}`\n"
        "- scientific claim status: 未授权；未通过或不可执行的门槛保持 blocked/failed。\n"
        "- legacy V1 结果未被覆盖，confirm 未用于覆盖 dev 失败。\n\n"
        "## V1-R.2F audits\n\n"
        + audit_lines
        + "\n",
        encoding="utf-8",
    )
    runtime_repairs_passed = (
        audits["initial_state_pairing_audit"].get("status") == "passed"
        and audits["runner_parity_audit"].get("status") == "passed"
    )
    clean_scientific_status = (
        "passed" if clean.get("status") == "passed" else "blocked"
    )
    deployment_device = audits["cpu_cuda_parity_audit"].get(
        "deployment_device", "unresolved"
    )
    args.output_dir.joinpath("v1r_decision.yaml").write_text(
        f"decision: {decision}\n"
        f"v1r_runtime_repairs: {'passed' if runtime_repairs_passed else 'incomplete'}\n"
        f"clean_baseline_scientific_gate: {clean_scientific_status}\n"
        f"initial_state_pairing_audit: {audits['initial_state_pairing_audit'].get('status', 'blocked')}\n"
        f"runner_parity_audit: {audits['runner_parity_audit'].get('status', 'blocked')}\n"
        f"cpu_cuda_parity_audit: {audits['cpu_cuda_parity_audit'].get('status', 'blocked')}\n"
        f"deployment_device: {deployment_device}\n"
        f"expert_replay_audit: {audits['expert_replay_audit'].get('status', 'blocked')}\n"
        f"per_layout_failure_analysis: {audits['per_layout_failure_analysis'].get('status', 'blocked')}\n"
        f"v2_formal_experiment_authorized: {str(summary['v2_formal_experiment_authorized']).lower()}\n"
        f"v3_formal_experiment_authorized: {str(summary['v3_formal_experiment_authorized']).lower()}\n"
        "next_stage: v1r_2f_clean_baseline_root_cause_repair\n",
        encoding="utf-8",
    )
    args.output_dir.joinpath("v2_handoff.yaml").write_text("authorized: false\nreason: V1-R clean baseline and/or scientific gates are not passed\n", encoding="utf-8")
    args.output_dir.joinpath("v3_handoff.yaml").write_text("authorized: false\nreason: V1-R context alias gate is not passed\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "v2": summary["v2_formal_experiment_authorized"], "v3": summary["v3_formal_experiment_authorized"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
