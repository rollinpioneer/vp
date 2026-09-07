#!/usr/bin/env python3
"""Identify reference phases from an offline clean rollout table."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def identify(input_path: Path, output_path: Path, static_speed: float, moving_speed: float) -> dict[str, object]:
    with input_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"episode_id", "step", "object_speed_mps", "object_attached", "gripper_closed", "reference_success"}
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"input is missing columns: {sorted(missing)}")
    by_episode: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode_id"]].append(row)
    output: list[dict[str, object]] = []
    for episode_id, values in sorted(by_episode.items()):
        values.sort(key=lambda row: int(row["step"]))
        pre = next((row for row in values if not int(row["object_attached"]) and float(row["object_speed_mps"]) <= static_speed and int(row["gripper_closed"]) == 0), None)
        transport = next((row for row in values if int(row["object_attached"]) and float(row["object_speed_mps"]) >= moving_speed), None)
        output.append({
            "episode_id": episode_id,
            "pregrasp_start_step": int(pre["step"]) if pre else "",
            "transport_start_step": int(transport["step"]) if transport else "",
            "object_speed_at_start": float(transport["object_speed_mps"]) if transport else "",
            "reference_success": int(values[0]["reference_success"]),
            "phase_complete": bool(pre and transport and int(values[0]["reference_success"])),
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output[0]) if output else ["episode_id", "pregrasp_start_step", "transport_start_step", "object_speed_at_start", "reference_success", "phase_complete"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    return {"status": "passed", "episodes": len(output), "phase_complete": sum(bool(row["phase_complete"]) for row in output)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--static-speed", type=float, default=0.01)
    parser.add_argument("--moving-speed", type=float, default=0.05)
    args = parser.parse_args()
    print(identify(args.input, args.output, args.static_speed, args.moving_speed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
