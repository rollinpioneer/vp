#!/usr/bin/env python3
"""Summarize clean baseline success and failure stages by layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    with args.rollouts.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        parser.error("rollout CSV is empty")
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    failure_stages: Counter[str] = Counter()
    termination_reasons: Counter[str] = Counter()
    for row in rows:
        layout = int(str(row["layout"]).rsplit("_", 1)[-1])
        grouped[layout].append(row)
        if int(row["success"]):
            continue
        stage = row.get("failure_stage") or (
            "timeout" if int(row.get("steps", "0")) >= 300 else "unknown"
        )
        failure_stages[stage] += 1
        termination = row.get("termination_reason") or (
            "timeout" if int(row.get("steps", "0")) >= 300 else "unknown"
        )
        termination_reasons[termination] += 1
    layouts = []
    for layout in range(1, 5):
        values = grouped.get(layout, [])
        successes = sum(int(row["success"]) for row in values)
        layouts.append(
            {
                "layout": layout,
                "scenarios": len(values),
                "successes": successes,
                "success_rate": successes / len(values) if values else None,
            }
        )
    result = {
        "stage": "V1-R.2F",
        "audit": "per_layout_failure_analysis",
        "status": "complete",
        "rollout_csv": str(args.rollouts),
        "rollout_csv_sha256": sha256(args.rollouts),
        "rows": len(rows),
        "layouts": layouts,
        "failure_stages": dict(sorted(failure_stages.items())),
        "termination_reasons": dict(sorted(termination_reasons.items())),
        "phase_thresholds_m": {
            "approach": 0.12,
            "bowl_to_plate": 0.12,
        },
        "fine_phase_telemetry_available": "failure_stage" in rows[0],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    table = [
        "| layout | scenarios | successes | success rate |",
        "|---:|---:|---:|---:|",
    ]
    for item in layouts:
        rate = "n/a" if item["success_rate"] is None else f"{item['success_rate']:.3f}"
        table.append(
            f"| {item['layout']} | {item['scenarios']} | {item['successes']} | {rate} |"
        )
    stage_lines = "\n".join(
        f"- `{stage}`: {count}" for stage, count in failure_stages.items()
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# V1-R.2F Layout Failure Analysis\n\n"
        + "\n".join(table)
        + "\n\n失败阶段：\n"
        + stage_lines
        + "\n\n"
        + f"原始 CSV SHA-256：`{result['rollout_csv_sha256']}`。\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
