#!/usr/bin/env python3
"""Summarize 5/10/20 Hz perception comparisons from rollout records."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("experiments/v1r/reports/frequency_camera_gate.json"))
    args = parser.parse_args()
    blockers: list[str] = []
    rows: list[dict[str, str]] = []
    if args.input is None or not args.input.is_file():
        blockers.append("no paired frequency/camera rollout CSV was supplied")
    else:
        with args.input.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        required = {"perception_update_hz", "camera_mode", "success", "mean_latency_ms", "p95_latency_ms"}
        missing = required.difference(rows[0] if rows else ())
        if missing:
            blockers.append(f"input is missing columns: {sorted(missing)}")
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["perception_update_hz"], row["camera_mode"])].append(row)
    summary = [{"perception_update_hz": hz, "camera_mode": camera, "n": len(values), "success_rate": sum(int(row["success"]) for row in values) / len(values), "mean_latency_ms": sum(float(row["mean_latency_ms"]) for row in values) / len(values), "p95_latency_ms": max(float(row["p95_latency_ms"]) for row in values)} for (hz, camera), values in sorted(groups.items())]
    payload = {"stage": "V1-R.5", "status": "blocked" if blockers else "unresolved", "best_simple": None, "oracle_minus_best_simple": None, "blockers": blockers, "groups": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text("# V1-R.5 简单替代方案\n\n" + "\n".join(f"- {item}" for item in blockers) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
