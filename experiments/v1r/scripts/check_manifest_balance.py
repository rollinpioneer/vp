#!/usr/bin/env python3
"""Validate the V1-R visibility manifest's crossed factors and pair keys."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def check(path: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "training_seed", "layout", "phase", "occlusion_duration_steps",
        "base_scenario_id", "branch", "simulator_seed", "initial_state_key",
        "fixed_object_point_identity_key", "mask_schedule_key",
    }
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    cell_counts = Counter(
        (row["training_seed"], row["layout"], row["phase"], row["occlusion_duration_steps"])
        for row in rows
        if row["branch"] != "E00_CLEAN"
    )
    if len(set(cell_counts.values())) != 1:
        raise ValueError(f"factor cells are unbalanced: {cell_counts}")
    by_base: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_base[row["base_scenario_id"]].append(row)
    expected = {
        "E00_CLEAN",
        "E10",
        "E10_ORACLE_HIDDEN_GT",
    }
    for base_id, group in by_base.items():
        branches = {row["branch"] for row in group}
        if branches != expected or sum(row["branch"] != "E00_CLEAN" for row in group) != 6:
            raise ValueError(f"{base_id} has branches {sorted(branches)}")
        durations = {
            (row["branch"], row["occlusion_duration_steps"])
            for row in group
            if row["branch"] != "E00_CLEAN"
        }
        if durations != {
            (branch, duration)
            for branch in ("E10", "E10_ORACLE_HIDDEN_GT")
            for duration in ("5", "10", "20")
        }:
            raise ValueError(f"{base_id} does not have all duration branches")
        keys = {(row["simulator_seed"], row["initial_state_key"], row["fixed_object_point_identity_key"], row["mask_schedule_key"]) for row in group}
        if len(keys) != 1:
            raise ValueError(f"pair keys differ for {base_id}")
    duration_by_layout = defaultdict(set)
    for row in rows:
        if row["branch"] != "E00_CLEAN":
            duration_by_layout[row["layout"]].add(row["occlusion_duration_steps"])
    if any(set(Durations) != {"5", "10", "20"} for Durations in duration_by_layout.values()):
        raise ValueError("duration is not fully crossed within each layout")
    return {"rows": len(rows), "base_scenarios": len(by_base), "factor_cells": len(cell_counts), "cell_count": next(iter(cell_counts.values()))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(check(args.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
