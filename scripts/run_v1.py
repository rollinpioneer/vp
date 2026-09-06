#!/usr/bin/env python3
"""Run the deterministic V1 four-condition diagnostic smoke experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vico_point.envs.scenarios import TASKS, generate_scenarios, write_registry  # noqa: E402
from vico_point.envs.synthetic import METHODS, evaluate_scenario  # noqa: E402
from vico_point.evaluation.metrics import paired_differences, write_results  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios-per-condition", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scenarios = generate_scenarios(
        scenarios_per_condition=args.scenarios_per_condition,
        tasks=TASKS,
        seed=args.seed,
    )
    registry_path = ROOT / "manifests" / "scenario_registry.csv"
    write_registry(scenarios, registry_path)

    rows = [
        evaluate_scenario(scenario, method, training_seed)
        for training_seed in (0, 1, 2)
        for scenario in scenarios
        for method in METHODS
    ]
    output_dir = ROOT / "outputs" / "v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_results(rows, output_dir / "diagnostic_results.csv", output_dir / "summary.json")

    paired = paired_differences(rows, "budget_matched_full_scene", "sparse_task_points")
    report = {
        "stage": "V1",
        "status": "diagnostic_smoke_complete",
        "scientific_status": "not_a_pointbridge_result",
        "tasks": list(TASKS),
        "conditions": ["E00", "E10", "E01", "E11"],
        "training_seeds": [0, 1, 2],
        "scenarios_per_condition_per_task": args.scenarios_per_condition,
        "rows": len(rows),
        "paired_comparison": {
            "method_a": "budget_matched_full_scene",
            "method_b": "sparse_task_points",
            "mean_success_difference_by_condition": {
                condition: sum(values) / len(values) if values else 0.0
                for condition, values in paired.items()
            },
        },
        "interpretation": [
            "This runner validates scenario generation, pairing and metric plumbing.",
            "Its synthetic outcomes must not be reported as upstream or scientific evidence.",
            "Replace evaluate_scenario with a Point Bridge adapter before drawing V1 conclusions.",
        ],
    }
    (output_dir / "diagnostic_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
