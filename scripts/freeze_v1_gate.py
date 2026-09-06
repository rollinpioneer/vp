#!/usr/bin/env python3
"""Freeze the 3 x 50 paired E00/E10 Point Bridge evaluation scenarios."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "manifests" / "v1_e00_e10_scenarios.csv"


def stable_seed(training_seed: int, index: int) -> int:
    payload = f"bowl_on_plate|{training_seed}|{index}|2026-09-06".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def main() -> int:
    rows = []
    for training_seed in (0, 1, 2):
        for index in range(50):
            scenario_id = f"pb_bowl_s{training_seed}_{index:03d}"
            simulator_seed = stable_seed(training_seed, index)
            duration_steps = (5, 10)[index % 2]
            start_step = 20 + 5 * (index % 4)
            for condition in ("E00", "E10"):
                rows.append(
                    {
                        "training_seed": training_seed,
                        "scenario_id": scenario_id,
                        "condition": condition,
                        "task": "bowl_on_plate",
                        "simulator_seed": simulator_seed,
                        "initial_state_pair_key": f"{scenario_id}_state",
                        "object_point_identity_key": f"{scenario_id}_points128",
                        "occlusion_start_step": start_step if condition == "E10" else "",
                        "occlusion_duration_steps": duration_steps if condition == "E10" else 0,
                        "control_hz": 20,
                        "split": "test",
                    }
                )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
