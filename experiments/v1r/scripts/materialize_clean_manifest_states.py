#!/usr/bin/env python3
"""Freeze full simulator states for all clean dev and confirm scenarios."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPSTREAM = ROOT / "third_party" / "pointbridge"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from state_utils import file_sha256, raw_array_sha256


def load_legacy_runner():
    path = ROOT / "scripts" / "evaluate_pointbridge_paired.py"
    spec = importlib.util.spec_from_file_location("vico_pointbridge_state_materializer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"clean manifest is empty: {path}")
    unique: dict[str, dict[str, str]] = {}
    for row in rows:
        scenario_id = row["scenario_id"]
        if scenario_id in unique:
            prior = unique[scenario_id]
            for key in ("layout", "simulator_seed", "initial_state_key", "split"):
                if prior[key] != row[key]:
                    raise ValueError(f"scenario {scenario_id} changes {key} across training seeds")
        else:
            unique[scenario_id] = row
    return list(unique.values())


def source_revision(upstream: Path) -> str:
    lock = upstream / ".vico_source_lock"
    if lock.is_file():
        for line in lock.read_text(encoding="utf-8").splitlines():
            if line.startswith("commit="):
                return line.split("=", 1)[1].strip()
    return "unavailable"


def write_augmented_manifest(
    path: Path, index: dict[str, dict[str, Any]]
) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        original_fields = list(rows[0])
    added_fields = [
        "state_path",
        "saved_state_sha256",
        "restored_state_sha256",
        "state_file_sha256",
    ]
    for row in rows:
        state = index[row["scenario_id"]]
        row.update(
            {
                "state_path": state["state_path"],
                "saved_state_sha256": state["state_sha256"],
                "restored_state_sha256": state["restored_state_sha256"],
                "state_file_sha256": state["state_file_sha256"],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=original_fields + added_fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--state-dir", type=Path, default=ROOT / "outputs/v1r/clean_state_store")
    parser.add_argument("--index", type=Path, default=ROOT / "experiments/v1r/manifests/clean_state_index.csv")
    parser.add_argument("--update-manifests", action="store_true")
    args = parser.parse_args()

    manifests = [path.resolve() for path in args.manifest]
    scenarios: dict[str, dict[str, str]] = {}
    for manifest in manifests:
        for row in read_manifest(manifest):
            if row["scenario_id"] in scenarios:
                raise ValueError(f"duplicate scenario across manifests: {row['scenario_id']}")
            scenarios[row["scenario_id"]] = row

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TENSORBOARD_NO_TF", "1")
    checkpoint = args.checkpoint.resolve()
    upstream = args.upstream.resolve()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    legacy = load_legacy_runner()
    legacy._add_paths(upstream)
    eval_module = legacy._load_eval_module(upstream)
    cfg = legacy._compose_config(upstream, checkpoint, 0, device="cpu")
    cfg.save_video = False
    cfg.use_tb = False
    old_cwd = Path.cwd()
    records: list[dict[str, Any]] = []
    os.chdir(upstream)
    try:
        workspace = eval_module.Workspace(cfg)
        for index, (scenario_id, scenario) in enumerate(scenarios.items()):
            seed = int(scenario["simulator_seed"])
            random.seed(seed)
            np.random.seed(seed % (2**32))
            try:
                import torch

                torch.manual_seed(seed)
            except ImportError:
                pass
            env = workspace.env[int(scenario["layout"]) - 1]
            env.reset()
            initial = np.asarray(env.sim.get_state().flatten()).copy()
            env.sim.set_state_from_flattened(initial)
            env.sim.forward()
            saved_state = np.asarray(env.sim.get_state().flatten()).copy()
            object_points = {
                str(key): np.asarray(value).copy()
                for key, value in env.object_points.items()
            }
            env.sim.set_state_from_flattened(saved_state)
            env.sim.forward()
            restored_hash = raw_array_sha256(env.sim.get_state().flatten())
            path = args.state_dir / f"{scenario_id}.npz"
            keys = sorted(object_points)
            payload: dict[str, np.ndarray] = {
                "sim_state": saved_state,
                "object_point_keys": np.asarray(keys, dtype="U"),
            }
            for object_index, key in enumerate(keys):
                payload[f"object_points_{object_index}"] = object_points[key]
            np.savez_compressed(path, **payload)
            records.append(
                {
                    "scenario_id": scenario_id,
                    "layout": int(scenario["layout"]),
                    "simulator_seed": seed,
                    "initial_state_key": scenario["initial_state_key"],
                    "split": scenario["split"],
                    "state_path": str(path.relative_to(ROOT)),
                    "state_sha256": raw_array_sha256(saved_state),
                    "restored_state_sha256": restored_hash,
                    "state_file_sha256": file_sha256(path),
                    "state_length": saved_state.size,
                    "object_point_keys": ";".join(keys),
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "upstream_git_revision": source_revision(upstream),
                }
            )
            print(
                f"{index + 1}/{len(scenarios)} {scenario_id} split={scenario['split']} layout={scenario['layout']}",
                flush=True,
            )
    finally:
        if "workspace" in locals():
            for env in workspace.env:
                env.close()
        os.chdir(old_cwd)

    args.index.parent.mkdir(parents=True, exist_ok=True)
    with args.index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(records[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
    by_id = {str(record["scenario_id"]): record for record in records}
    if args.update_manifests:
        for manifest in manifests:
            write_augmented_manifest(manifest, by_id)
    result = {
        "stage": "V1-R.2F",
        "status": "complete",
        "states": len(records),
        "dev_states": sum(record["split"] == "dev" for record in records),
        "confirm_states": sum(record["split"] == "confirm" for record in records),
        "training_seed_sharing": "one_state_file_per_scenario_shared_by_seeds_0_1_2",
        "confirm_rollouts_executed": False,
        "index": str(args.index.resolve().relative_to(ROOT)),
        "checkpoint_sha256": records[0]["checkpoint_sha256"],
        "upstream_git_revision": records[0]["upstream_git_revision"],
    }
    args.index.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
