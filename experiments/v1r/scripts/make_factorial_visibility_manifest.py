#!/usr/bin/env python3
"""Create the crossed V1-R visibility factors without layout-duration confounding."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

TRAINING_SEEDS = (0, 1, 2)
LAYOUTS = (1, 2, 3, 4)
PHASES = ("pregrasp_static", "transport_moving")
DURATIONS_STEPS = (5, 10, 20)
TARGET_HIDDEN_FRACTIONS = (0.60,)
SCENARIOS_PER_CELL = 10


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in TRAINING_SEEDS:
        for layout in LAYOUTS:
            for phase in PHASES:
                for index in range(SCENARIOS_PER_CELL):
                    base_id = f"v1r_vis_s{seed}_l{layout}_{phase}_{index:02d}"
                    simulator_seed = stable_seed("v1r_visibility", seed, layout, phase, index)
                    start_step = 20 + 5 * ((index + layout + (phase == "transport_moving")) % 4)
                    schedule_key = f"mask_{seed}_{layout}_{phase}_{index:02d}"
                    common = {
                        "training_seed": seed,
                        "base_scenario_id": base_id,
                        "layout": layout,
                        "phase": phase,
                        "task": "bowl_on_plate",
                        "simulator_seed": simulator_seed,
                        "initial_state_key": f"{base_id}_state",
                        "fixed_object_point_identity_key": f"{base_id}_points128",
                        "occlusion_start_step": start_step,
                        "control_hz": 20,
                        "nominal_hidden_fraction": TARGET_HIDDEN_FRACTIONS[0],
                        "occlusion_phase": phase,
                        "occlusion_target": "primary_task_object",
                        "occluder_type": "precomputed_reference_mask_sequence",
                        "mask_schedule_key": schedule_key,
                        "split": "test",
                    }
                    rows.append({**common, "branch": "E00_CLEAN", "oracle": False, "occlusion_duration_steps": 0, "occlusion_duration_seconds": 0.0})
                    for duration in DURATIONS_STEPS:
                        duration_fields = {
                            "occlusion_duration_steps": duration,
                            "occlusion_duration_seconds": duration / 20.0,
                        }
                        rows.append({**common, **duration_fields, "branch": "E10", "oracle": False})
                        rows.append({**common, **duration_fields, "branch": "E10_ORACLE_HIDDEN_GT", "oracle": True})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_rows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
