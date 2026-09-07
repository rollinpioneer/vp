#!/usr/bin/env python3
"""Apply the predeclared V1-R gate logic and write handoff artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GATES = ("clean_baseline_gate.json", "perception_gate.json", "visibility_scientific_gate.json", "frequency_camera_gate.json", "context_gate.json")


def load_gate(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"status": "blocked", "blockers": [f"missing gate file: {path}"]}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=Path("experiments/v1r/reports"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/v1r"))
    args = parser.parse_args()
    gates = {name.removesuffix(".json"): load_gate(args.reports_dir / name) for name in GATES}
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
    summary = {"stage": "V1-R.7", "decision": decision, "gates": gates, "v2_formal_experiment_authorized": decision in {"authorize_v2_moving_object_point_belief", "authorize_v2_then_v3_sequentially"}, "v3_formal_experiment_authorized": decision in {"authorize_v3_fixed_budget_context_selection", "authorize_v2_then_v3_sequentially"}}
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "v1r_gate_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.reports_dir / "v1r_final_report.md").write_text("# V1-R 执行结果\n\n" + f"- decision: `{decision}`\n" + "- scientific claim status: 未授权，所有缺少真实输入的门槛保持 blocked/unresolved。\n" + "- legacy V1 结果未被覆盖。\n", encoding="utf-8")
    args.output_dir.joinpath("v1r_decision.yaml").write_text(f"decision: {decision}\nv2_formal_experiment_authorized: {str(summary['v2_formal_experiment_authorized']).lower()}\nv3_formal_experiment_authorized: {str(summary['v3_formal_experiment_authorized']).lower()}\n", encoding="utf-8")
    args.output_dir.joinpath("v2_handoff.yaml").write_text("authorized: false\nreason: V1-R clean baseline and/or scientific gates are not passed\n", encoding="utf-8")
    args.output_dir.joinpath("v3_handoff.yaml").write_text("authorized: false\nreason: V1-R context alias gate is not passed\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "v2": summary["v2_formal_experiment_authorized"], "v3": summary["v3_formal_experiment_authorized"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
