#!/usr/bin/env python3
"""Aggregate causal perception audit rows without fabricating missing evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


REQUIRED = {"episode_id", "point_mode", "object_motion_regime", "error_3d_m", "source", "identity_switch", "confidence", "visible_pred", "visible_gt", "age_seconds"}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return values[index]


def run(input_path: Path | None, per_point: Path, per_episode: Path, gate: Path, report: Path) -> dict[str, object]:
    blockers: list[str] = []
    rows: list[dict[str, str]] = []
    if input_path is None or not input_path.is_file():
        blockers.append("no recorded RGB-D/GT perception audit CSV was supplied")
    else:
        with input_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        missing = REQUIRED.difference(rows[0] if rows else ())
        if missing:
            blockers.append(f"input is missing columns: {sorted(missing)}")
    per_point.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else sorted(REQUIRED)
    with per_point.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summaries: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["point_mode"], row["object_motion_regime"])].append(row)
    for (mode, regime), values in sorted(grouped.items()):
        errors = [float(row["error_3d_m"]) for row in values]
        stale = sum(row["source"] in {"previous_fallback", "last_reliable_hold"} for row in values) / len(values)
        identity = sum(int(row["identity_switch"]) for row in values) / len(values)
        summaries.append({
            "point_mode": mode,
            "object_motion_regime": regime,
            "n": len(values),
            "error_3d_median": percentile(errors, 0.50),
            "error_3d_p90": percentile(errors, 0.90),
            "error_gt_2cm_fraction": sum(error >= 0.02 for error in errors) / len(errors),
            "stale_or_fallback_fraction": stale,
            "identity_switch_fraction": identity,
            "mean_age_seconds": sum(float(row["age_seconds"]) for row in values) / len(values),
        })
    with per_episode.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["point_mode", "object_motion_regime", "n", "error_3d_median", "error_3d_p90", "error_gt_2cm_fraction", "stale_or_fallback_fraction", "identity_switch_fraction", "mean_age_seconds"])
        writer.writeheader()
        writer.writerows(summaries)
    status = "blocked" if blockers else "unresolved"
    payload = {
        "stage": "V1-R.3",
        "status": status,
        "perception_reliability_problem_observed": None,
        "failure_relevance": "unresolved",
        "criteria_any": {
            "moving_object_p90_3d_error_m": 0.02,
            "stale_or_fallback_fraction": 0.05,
            "identity_switch_fraction": 0.01,
            "low_visibility_points_with_error_gt_2cm_fraction": 0.05,
        },
        "blockers": blockers,
        "groups": summaries,
    }
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report.write_text("# V1-R.3 感知审计\n\n" + ("状态：`blocked`。\n\n" if blockers else "状态：`unresolved`，等待失败回合/成功回合相关性证据。\n\n") + "\n".join(f"- {item}" for item in blockers) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--per-point", type=Path, default=Path("outputs/v1r/perception_audit/per_point.csv"))
    parser.add_argument("--per-episode", type=Path, default=Path("outputs/v1r/perception_audit/per_episode.csv"))
    parser.add_argument("--gate", type=Path, default=Path("experiments/v1r/reports/perception_gate.json"))
    parser.add_argument("--report", type=Path, default=Path("experiments/v1r/reports/perception_audit.md"))
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.per_point, args.per_episode, args.gate, args.report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
