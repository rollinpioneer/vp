#!/usr/bin/env python3
"""Build frozen target-mask metadata from reference projection boxes.

The script requires offline reference boxes. It never reads a candidate policy
trajectory, so a missing input is reported as a blocker rather than filled with
a synthetic target location.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v1r/mask_schedules"))
    parser.add_argument("--index", type=Path, default=Path("experiments/v1r/manifests/mask_schedule_index.csv"))
    args = parser.parse_args()
    rows: list[dict[str, str]] = []
    blockers: list[str] = []
    if args.input is None or not args.input.is_file():
        blockers.append("no clean reference projection-box table was supplied")
    else:
        with args.input.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        required = {"base_scenario_id", "phase", "step", "target_x0", "target_y0", "target_x1", "target_y1"}
        missing = required.difference(rows[0] if rows else ())
        if missing:
            blockers.append(f"input is missing columns: {sorted(missing)}")
    args.index.parent.mkdir(parents=True, exist_ok=True)
    fields = ["base_scenario_id", "phase", "schedule_path", "schedule_sha256", "status", "actual_reference_hidden_fraction"]
    entries: list[dict[str, object]] = []
    if not blockers:
        grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault((row["base_scenario_id"], row["phase"]), []).append(row)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for (scenario, phase), values in sorted(grouped.items()):
            payload = {"base_scenario_id": scenario, "phase": phase, "frames": values, "nominal_target_hidden_fraction": 0.60}
            raw = json.dumps(payload, sort_keys=True).encode("utf-8")
            path = args.output_dir / f"{scenario}_{phase}.json"
            path.write_bytes(raw)
            entries.append({"base_scenario_id": scenario, "phase": phase, "schedule_path": str(path), "schedule_sha256": hashlib.sha256(raw).hexdigest(), "status": "requires_mask_rasterization", "actual_reference_hidden_fraction": ""})
    if not entries:
        entries.append({"base_scenario_id": "", "phase": "", "schedule_path": "", "schedule_sha256": "", "status": "blocked", "actual_reference_hidden_fraction": ""})
    with args.index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(entries)
    print(json.dumps({"status": "blocked" if blockers else "unresolved", "blockers": blockers, "schedules": len(entries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
