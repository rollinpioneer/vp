#!/usr/bin/env python3
"""Materialize ten complete V1-R audit states from legacy V1 E00 scenarios."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPSTREAM = ROOT / "third_party" / "pointbridge"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from state_utils import file_sha256, raw_array_sha256


def load_legacy_runner():
    path = ROOT / "scripts" / "evaluate_pointbridge_paired.py"
    spec = importlib.util.spec_from_file_location("vico_pointbridge_legacy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_scenarios(paired: Path, limit_per_class: int) -> list[dict[str, str]]:
    with paired.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["training_seed"] == "0" and row["branch"] == "E00"]
    chosen: list[dict[str, str]] = []
    for success in ("1", "0"):
        candidates = [row for row in rows if row["success"] == success]
        if len(candidates) < limit_per_class:
            raise ValueError(f"legacy E00 has fewer than {limit_per_class} rows for success={success}")
        chosen.extend(candidates[:limit_per_class])
    return chosen


def git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unavailable"


def upstream_revision(path: Path) -> str:
    lock = path / ".vico_source_lock"
    if lock.is_file():
        for line in lock.read_text(encoding="utf-8").splitlines():
            if line.startswith("commit="):
                return line.split("=", 1)[1].strip()
    return git_revision(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-rollouts", type=Path, default=ROOT / "outputs/v1/pointbridge_seed0/paired_rollouts.csv")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--state-dir", type=Path, default=ROOT / "outputs/v1r/state_store")
    parser.add_argument("--index", type=Path, default=ROOT / "experiments/v1r/manifests/runner_parity_state_index.csv")
    parser.add_argument("--limit-per-class", type=int, default=5)
    args = parser.parse_args()
    if not args.paired_rollouts.is_file():
        parser.error(f"paired rollout CSV does not exist: {args.paired_rollouts}")
    checkpoint = args.checkpoint.resolve()
    upstream = args.upstream.resolve()
    args.state_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TENSORBOARD_NO_TF", "1")
    runner = load_legacy_runner()
    runner._add_paths(upstream)
    eval_module = runner._load_eval_module(upstream)
    cfg = runner._compose_config(upstream, checkpoint, 0, device="cpu")
    cfg.save_video = False
    cfg.use_tb = False
    selected = choose_scenarios(args.paired_rollouts.resolve(), args.limit_per_class)
    old_cwd = Path.cwd()
    rows: list[dict[str, object]] = []
    os.chdir(upstream)
    try:
        workspace = eval_module.Workspace(cfg)
        for index, source in enumerate(selected):
            simulator_seed = int(source["simulator_seed"])
            random.seed(simulator_seed)
            np.random.seed(simulator_seed % (2**32))
            try:
                import torch

                torch.manual_seed(simulator_seed)
            except ImportError:
                pass
            layout = int(source["layout"].rsplit("_", 1)[-1])
            env = workspace.env[layout - 1]
            env.reset()
            # Canonicalize quaternions exactly as every audited restore does.
            reset_state = np.asarray(env.sim.get_state().flatten()).copy()
            env.sim.set_state_from_flattened(reset_state)
            env.sim.forward()
            state = np.asarray(env.sim.get_state().flatten()).copy()
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            restored_state_sha256 = raw_array_sha256(
                env.sim.get_state().flatten()
            )
            object_points = {
                str(key): np.asarray(value).copy() for key, value in env.object_points.items()
            }
            path = args.state_dir / f"{source['scenario_id']}.npz"
            keys = sorted(object_points)
            payload: dict[str, np.ndarray] = {
                "sim_state": state,
                "object_point_keys": np.asarray(keys, dtype="U"),
            }
            for object_index, key in enumerate(keys):
                payload[f"object_points_{object_index}"] = object_points[key]
            np.savez_compressed(path, **payload)
            rows.append(
                {
                    "scenario_id": source["scenario_id"],
                    "layout": layout,
                    "simulator_seed": simulator_seed,
                    "initial_state_key": f"{source['scenario_id']}_state",
                    "source_old_success": source["success"],
                    "source_old_initial_state_sha256": source["initial_state_sha256"],
                    "state_path": str(path.relative_to(ROOT)),
                    "state_sha256": raw_array_sha256(state),
                    "restored_state_sha256": restored_state_sha256,
                    "state_file_sha256": file_sha256(path),
                    "state_length": state.size,
                    "object_point_keys": ";".join(keys),
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "upstream_git_revision": upstream_revision(upstream),
                    "repo_git_revision": git_revision(ROOT),
                }
            )
            print(f"{index + 1}/{len(selected)} {source['scenario_id']} layout={layout} success={source['success']}", flush=True)
    finally:
        if "workspace" in locals():
            for env in workspace.env:
                env.close()
        os.chdir(old_cwd)

    args.index.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "stage": "V1-R.2F",
        "status": "complete",
        "states": len(rows),
        "success_states": sum(row["source_old_success"] == "1" for row in rows),
        "failure_states": sum(row["source_old_success"] == "0" for row in rows),
        "index": str(args.index.resolve().relative_to(ROOT)),
        "state_dir": str(args.state_dir.resolve().relative_to(ROOT)),
        "checkpoint_sha256": rows[0]["checkpoint_sha256"],
        "upstream_git_revision": rows[0]["upstream_git_revision"],
        "state_hashes_match_legacy": sum(
            row["source_old_initial_state_sha256"] == row["state_sha256"] for row in rows
        ),
    }
    output = args.index.with_suffix(".json")
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
