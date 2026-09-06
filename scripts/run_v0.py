#!/usr/bin/env python3
"""Run V0 contract checks and write the handoff artifact."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vico_point.core.types import ActionProtocol, InputLevel, PointBudget, RunMetadata  # noqa: E402
from vico_point.core.validation import validate_frame, validate_metadata  # noqa: E402
from vico_point.perception.observation import build_frame, make_point  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted"


def main() -> int:
    budget = PointBudget()
    frame = build_frame(
        "v0-contract-frame",
        0.0,
        InputLevel.P,
        robot_points=(make_point(f"robot_{i}", (0.0, 0.0, 0.0), "robot", timestamp=0.0) for i in range(budget.robot)),
        task_points=(make_point(f"task_{i}", (0.1, 0.0, 0.1), "task", timestamp=0.0) for i in range(budget.task)),
        context_points=(make_point(f"context_{i}", (0.2, 0.0, 0.0), "context", timestamp=0.0) for i in range(budget.context)),
    )
    validate_frame(frame, budget)
    metadata = RunMetadata(
        upstream_commit="5d567a62d62b5a97c5960d45024e065349680cda",
        own_commit=git_commit(),
        input_level=InputLevel.P,
        point_budget=budget,
        history_frames=8,
        action_protocol=ActionProtocol(),
        data_split="contract_smoke",
        hardware="local_cpu",
    )
    validate_metadata(metadata)

    output = {
        "stage": "V0",
        "status": "interface_ready_upstream_not_included",
        "upstream_commit": metadata.upstream_commit,
        "own_commit": metadata.own_commit,
        "input_level": metadata.input_level.value,
        "point_budget": {
            "total": budget.total,
            "robot": budget.robot,
            "task": budget.task,
            "context": budget.context,
        },
        "history_frames": metadata.history_frames,
        "action_protocol": {
            "action_dim": metadata.action_protocol.action_dim,
            "prediction_horizon": metadata.action_protocol.prediction_horizon,
            "execution_horizon": metadata.action_protocol.execution_horizon,
            "aggregation": metadata.action_protocol.aggregation,
        },
        "data_root": "external_not_provided",
        "hardware": metadata.hardware,
        "checks": [
            "point_identity_and_masks",
            "budget_accounting",
            "calibration_units_explicit",
            "action_protocol_present",
            "upstream_adapter_boundary_present",
        ],
        "next_blocker": "checkout Point Bridge at the locked commit and run one official bowl_on_plate task",
    }
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "v0_contract_check.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "experiments" / "v0" / "v0_handoff.yaml").write_text(
        "\n".join(
            [
                "stage: V0",
                "status: interface_ready_upstream_not_included",
                f"upstream_commit: {metadata.upstream_commit}",
                f"own_commit: {metadata.own_commit}",
                "primary_task: bowl_on_plate",
                "input_level: P",
                "point_budget: 64",
                "robot_points: 9",
                "task_points: 16",
                "context_points: 39",
                "history_frames: 8",
                "action_dim: 7",
                "action_protocol: first_action",
                "data_root: external_not_provided",
                "legacy_root: read_only_optional",
                "upstream_training_complete: false",
                "synthetic_contract_check: passed",
                "next_step: checkout_locked_pointbridge_and_run_official_bowl_on_plate",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
