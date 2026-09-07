#!/usr/bin/env python3
"""Run causal, unoccluded Point Bridge rollouts for the V1-R clean gate.

This runner uses the independent V1-R manifest and the pinned Point Bridge
checkpoint. It records a complete rollout row even when an episode fails due to
an exception, so the clean gate cannot silently drop simulator failures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPSTREAM = ROOT / "third_party" / "pointbridge"


def load_legacy_runner():
    path = ROOT / "scripts" / "evaluate_pointbridge_paired.py"
    spec = importlib.util.spec_from_file_location("vico_point_paired_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def load_snapshot(workspace: Any, checkpoint: Path, device: str) -> None:
    """Load CUDA-trained weights on CPU without changing the checkpoint."""

    import torch

    if device != "cpu":
        workspace.load_snapshot({"bc": checkpoint})
        return
    with checkpoint.open("rb") as handle:
        payload = torch.load(handle, map_location=torch.device("cpu"), weights_only=False)
    agent_payload = {key: value for key, value in payload.items() if key not in workspace.__dict__}
    workspace.agent.load_snapshot(agent_payload, eval=True)
    workspace.stats = payload["stats"]


def load_rows(path: Path, seed: int, split: str | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["training_seed"]) == seed and row["condition"] == "E00_CLEAN"]
    if split is not None:
        rows = [row for row in rows if row.get("split") == split]
    return rows


def run_episode(workspace: Any, env: Any, row: dict[str, str], seed: int) -> dict[str, object]:
    import torch
    from point_bridge import utils

    set_seed(int(row["simulator_seed"]))
    start_time = time.monotonic()
    result: dict[str, object] = {
        "training_seed": seed,
        "checkpoint_sha256": "",
        "scenario_id": row["scenario_id"],
        "base_scenario_id": row["scenario_id"],
        "branch": "E00_CLEAN",
        "layout": row["layout"],
        "simulator_seed": row["simulator_seed"],
        "initial_state_sha256": "",
        "point_identity_sha256": "",
        "mask_schedule_sha256": "",
        "point_mode": "gt_mesh_points",
        "camera_mode": "EXTERNAL_ONLY",
        "perception_update_hz": 20,
        "control_hz": 20,
        "occluder_type": "none",
        "occlusion_target": "none",
        "occlusion_phase": "none",
        "occlusion_duration_steps": 0,
        "occlusion_duration_seconds": 0.0,
        "nominal_hidden_fraction": 0.0,
        "actual_hidden_fraction_mean": 0.0,
        "actual_hidden_fraction_peak": 0.0,
        "object_motion_regime": "clean",
        "object_displacement_m": 0.0,
        "object_rotation_deg": 0.0,
        "memory_mode": "none",
        "point_budget_total": 0,
        "point_budget_task": 0,
        "point_budget_context": 0,
        "success": 0,
        "collision": 0,
        "steps": 0,
        "mean_point_age_seconds": 0.0,
        "p95_point_age_seconds": 0.0,
        "mean_point_error_m": 0.0,
        "p95_point_error_m": 0.0,
        "identity_switch_count": 0,
        "previous_fallback_count": 0,
        "oracle_hidden_truth_read_count": 0,
        "simulator_exception": 0,
        "action_decode_error": 0,
        "wall_clock_seconds": 0.0,
        "error_type": "",
    }
    try:
        time_step = env.reset()
        result["initial_state_sha256"] = hashlib.sha256(np.asarray(env.sim.get_state().flatten()).tobytes()).hexdigest()
        point_key = f"{env._object_points_key}_3d"
        if point_key in time_step.observation:
            points = np.asarray(time_step.observation[point_key])
            result["point_identity_sha256"] = array_sha256(points)
            result["point_budget_task"] = int(points.reshape(-1, 3).shape[0])
            result["point_budget_total"] = int(points.reshape(-1, 3).shape[0])
        workspace.agent.buffer_reset()
        steps = 0
        while not time_step.last():
            try:
                with torch.no_grad(), utils.eval_mode(workspace.agent):
                    action = workspace.agent.act(time_step.observation, workspace.stats, steps, workspace.global_step)
                action = np.asarray(action)
                if not np.isfinite(action).all():
                    raise ValueError("policy action contains non-finite values")
            except Exception as exc:
                result["action_decode_error"] = 1
                result["error_type"] = f"action:{type(exc).__name__}:{exc}\n{traceback.format_exc()[-2000:]}"
                break
            try:
                time_step = env.step(action)
            except Exception as exc:
                result["simulator_exception"] = 1
                result["error_type"] = f"simulator:{type(exc).__name__}:{exc}\n{traceback.format_exc()[-2000:]}"
                break
            steps += 1
        result["steps"] = steps
        result["success"] = int(bool(time_step.observation.get("goal_achieved", False)))
    except Exception as exc:
        result["simulator_exception"] = 1
        result["error_type"] = f"reset:{type(exc).__name__}:{exc}\n{traceback.format_exc()[-3000:]}"
    result["wall_clock_seconds"] = time.monotonic() - start_time
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--split", choices=("dev", "confirm"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    checkpoint = args.checkpoint.resolve()
    upstream = args.upstream.resolve()
    rows = load_rows(args.manifest, args.training_seed, args.split)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        parser.error("manifest selection is empty")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TENSORBOARD_NO_TF", "1")
    runner = load_legacy_runner()
    runner._add_paths(upstream)
    eval_module = runner._load_eval_module(upstream)
    cfg = runner._compose_config(upstream, checkpoint, args.training_seed, device=args.device)
    cfg.save_video = False
    cfg.use_tb = False
    old_cwd = Path.cwd()
    output_rows: list[dict[str, object]] = []
    os.chdir(upstream)
    try:
        workspace = eval_module.Workspace(cfg)
        load_snapshot(workspace, checkpoint, args.device)
        workspace.agent.train(False)
        for index, row in enumerate(rows):
            layout_index = int(row["layout"]) - 1
            if layout_index < 0 or layout_index >= len(workspace.env):
                raise ValueError(
                    f"manifest layout {row['layout']} is outside the loaded environment set"
                )
            env = workspace.env[layout_index]
            result = run_episode(workspace, env, row, args.training_seed)
            result["checkpoint_sha256"] = sha256(checkpoint)
            output_rows.append(result)
            print(f"{index + 1}/{len(rows)} {row['scenario_id']} success={result['success']} exception={result['simulator_exception']} action_error={result['action_decode_error']}", flush=True)
    finally:
        if "workspace" in locals():
            for env in workspace.env:
                try:
                    env.close()
                except Exception:
                    pass
        os.chdir(old_cwd)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output_rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "stage": "V1-R.2",
        "status": "complete" if len(output_rows) == len(load_rows(args.manifest, args.training_seed, args.split)) else "partial",
        "training_seed": args.training_seed,
        "rows": len(output_rows),
        "success_rate": sum(int(row["success"]) for row in output_rows) / len(output_rows),
        "simulator_exception_count": sum(int(row["simulator_exception"]) for row in output_rows),
        "action_decode_error_count": sum(int(row["action_decode_error"]) for row in output_rows),
        "checkpoint_sha256": sha256(checkpoint),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
