#!/usr/bin/env python3
"""Create disjoint clean dev/confirmation scenario lists for V1-R.2."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def stable_seed(split: str, layout: int, index: int) -> int:
    payload = f"v1r-clean|{split}|{layout}|{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def rows(split: str, per_layout: int) -> list[dict[str, object]]:
    output = []
    for layout in range(1, 5):
        for index in range(per_layout):
            simulator_seed = stable_seed(split, layout, index)
            base_id = f"v1r_clean_{split}_l{layout}_{index:02d}"
            for training_seed in (0, 1, 2):
                output.append(
                    {
                        "training_seed": training_seed,
                        "scenario_id": base_id,
                        "task": "bowl_on_plate",
                        "layout": layout,
                        "simulator_seed": simulator_seed,
                        "initial_state_key": f"{base_id}_state",
                        "condition": "E00_CLEAN",
                        "occlusion_duration_steps": 0,
                        "control_hz": 20,
                        "split": split,
                    }
                )
    return output


def write(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write(args.output_dir / "clean_baseline_dev.csv", rows("dev", 10))
    write(args.output_dir / "clean_baseline_confirm.csv", rows("confirm", 25))
    print("wrote clean dev (120 rows) and confirm (300 rows) manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
