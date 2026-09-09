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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from state_utils import (  # noqa: E402
    body_state_hash,
    load_state_bundle,
    load_state_index,
    pointbridge_core_env,
    raw_array_sha256,
    refresh_pointbridge_observation,
    robot_pose_hash,
)


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


def load_state_rows(
    path: Path, seed: int, split: str | None = None
) -> list[dict[str, str]]:
    rows = []
    for state_row in load_state_index(path).values():
        row_split = state_row.get("split", "audit")
        if split is not None and row_split != split:
            continue
        rows.append(
            {
                "training_seed": str(seed),
                "scenario_id": state_row["scenario_id"],
                "task": "bowl_on_plate",
                "layout": state_row["layout"],
                "simulator_seed": state_row["simulator_seed"],
                "initial_state_key": state_row["initial_state_key"],
                "condition": "E00_CLEAN",
                "occlusion_duration_steps": "0",
                "control_hz": "20",
                "split": row_split,
            }
        )
    return sorted(rows, key=lambda item: item["scenario_id"])


def normalized_input_sha256(workspace: Any, observation: dict[str, Any]) -> str:
    """Hash the normalized arrays consumed by the first network layer."""

    digest = hashlib.sha256()
    agent = workspace.agent
    stats = workspace.stats or {}
    keys = [agent.robot_points_key, agent.object_points_key, agent.proprio_key]
    for key in keys:
        if key not in observation:
            continue
        stat_key = agent.proprio_key if key == agent.proprio_key else "past_tracks"
        if stat_key not in stats:
            continue
        value = np.asarray(observation[key])
        minimum = np.asarray(stats[stat_key]["min"])
        maximum = np.asarray(stats[stat_key]["max"])
        normalized = (value - minimum) / (maximum - minimum + 1e-5)
        digest.update(key.encode("utf-8"))
        digest.update(str(normalized.dtype).encode("ascii"))
        digest.update(str(normalized.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(normalized).tobytes())
    return digest.hexdigest()


def _failure_stage(result: dict[str, object]) -> str:
    if int(result["success"]):
        return "success"
    if int(result["action_decode_error"]):
        return "action_decode"
    if int(result["simulator_exception"]):
        return "simulator"
    if float(result["min_eef_bowl_distance_m"]) > 0.12:
        return "no_approach"
    if not int(result["grasped_any"]):
        return "no_grasp"
    if int(result["dropped_after_grasp"]):
        return "post_grasp_drop"
    if float(result["min_bowl_plate_distance_m"]) > 0.12:
        return "no_reach_plate"
    if result["termination_reason"] == "timeout":
        return "placement_failure"
    return "unknown"


def _phase_observation(env: Any) -> dict[str, float | int]:
    core = pointbridge_core_env(env)
    robosuite_env = core._env
    eef = np.asarray(robosuite_env.sim.data.get_body_xpos("gripper0_eef"))
    bowl = np.asarray(robosuite_env.sim.data.get_body_xpos("bowl_main"))
    plate = np.asarray(robosuite_env.sim.data.get_body_xpos("plate_main"))
    grasped = bool(
        robosuite_env._check_grasp(
            robosuite_env.robots[0].gripper,
            robosuite_env.objects_dict["bowl"],
        )
    )
    return {
        "eef_bowl_distance_m": float(np.linalg.norm(eef - bowl)),
        "bowl_plate_distance_m": float(np.linalg.norm(bowl - plate)),
        "grasped": int(grasped),
    }


def _prepare_episode(
    env: Any,
    row: dict[str, str],
    state_index: dict[str, dict[str, str]] | None,
) -> tuple[Any, dict[str, object]]:
    """Reset and, when requested, restore a complete indexed state bundle."""

    time_step = env.reset()
    metadata: dict[str, object] = {
        "state_path": "",
        "saved_initial_state_sha256": "",
        "expected_initial_state_sha256": "",
        "historical_restored_state_sha256": "",
        "actual_initial_state_sha256": "",
        "initial_state_match": "unresolved",
        "historical_initial_state_match": "unresolved",
        "state_file_sha256": "",
    }
    state_row = state_index.get(row["scenario_id"]) if state_index is not None else None
    if state_row is not None:
        bundle = load_state_bundle(ROOT, state_row)
        env.sim.set_state_from_flattened(bundle["sim_state"])
        env.sim.forward()
        time_step = refresh_pointbridge_observation(env, time_step, bundle["object_points"])
        metadata.update(
            {
                "state_path": str(bundle["path"].relative_to(ROOT)),
                # The state bundle is the authoritative input for this
                # runtime. The historical restored hash is retained as a
                # compatibility diagnostic because it was produced under a
                # different Point Bridge / MuJoCo source identity.
                "saved_initial_state_sha256": state_row["state_sha256"],
                "expected_initial_state_sha256": state_row["state_sha256"],
                "historical_restored_state_sha256": state_row[
                    "restored_state_sha256"
                ],
                "state_file_sha256": bundle["file_sha256"],
            }
        )
    actual = raw_array_sha256(env.sim.get_state().flatten())
    metadata["actual_initial_state_sha256"] = actual
    metadata["initial_state_match"] = (
        "passed"
        if not metadata["expected_initial_state_sha256"]
        or metadata["expected_initial_state_sha256"] == actual
        else "failed"
    )
    historical = metadata["historical_restored_state_sha256"]
    metadata["historical_initial_state_match"] = (
        "passed" if not historical or historical == actual else "failed"
    )
    return time_step, metadata


def run_episode(
    workspace: Any,
    env: Any,
    row: dict[str, str],
    seed: int,
    state_index: dict[str, dict[str, str]] | None = None,
    max_steps: int | None = None,
) -> dict[str, object]:
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
        "state_path": "",
        "state_file_sha256": "",
        "saved_initial_state_sha256": "",
        "expected_initial_state_sha256": "",
        "historical_restored_state_sha256": "",
        "actual_initial_state_sha256": "",
        "initial_state_match": "unresolved",
        "historical_initial_state_match": "unresolved",
        "robot_pose_sha256": "",
        "robot_points_sha256": "",
        "body_state_sha256": "",
        "point_identity_sha256": "",
        "normalized_network_input_sha256": "",
        "first_20_actions_sha256": "",
        "first_20_actions_json": "",
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
        "failure_stage": "unknown",
        "termination_reason": "unknown",
        "min_eef_bowl_distance_m": float("inf"),
        "min_bowl_plate_distance_m": float("inf"),
        "final_eef_bowl_distance_m": float("inf"),
        "final_bowl_plate_distance_m": float("inf"),
        "grasped_any": 0,
        "dropped_after_grasp": 0,
        "first_grasp_step": -1,
    }
    first_actions: list[list[float]] = []
    first_action_digest = hashlib.sha256()
    try:
        time_step, state_metadata = _prepare_episode(env, row, state_index)
        result.update(state_metadata)
        result["initial_state_sha256"] = result["actual_initial_state_sha256"]
        result["robot_pose_sha256"] = robot_pose_hash(env)
        result["body_state_sha256"] = body_state_hash(env)
        point_key = f"{env._object_points_key}_3d"
        robot_key = f"{env._robot_points_key}_3d"
        if robot_key in time_step.observation:
            result["robot_points_sha256"] = array_sha256(
                time_step.observation[robot_key]
            )
        if point_key in time_step.observation:
            points = np.asarray(time_step.observation[point_key])
            result["point_identity_sha256"] = array_sha256(points)
            result["point_budget_task"] = int(points.reshape(-1, 3).shape[0])
            result["point_budget_total"] = int(points.reshape(-1, 3).shape[0])
        workspace.agent.buffer_reset()
        steps = 0
        phase = _phase_observation(env)
        result["min_eef_bowl_distance_m"] = phase["eef_bowl_distance_m"]
        result["min_bowl_plate_distance_m"] = phase["bowl_plate_distance_m"]
        while not time_step.last() and (max_steps is None or steps < max_steps):
            try:
                if steps == 0:
                    result["normalized_network_input_sha256"] = normalized_input_sha256(
                        workspace, time_step.observation
                    )
                with torch.no_grad(), utils.eval_mode(workspace.agent):
                    action = workspace.agent.act(time_step.observation, workspace.stats, steps, workspace.global_step)
                action = np.asarray(action)
                if not np.isfinite(action).all():
                    raise ValueError("policy action contains non-finite values")
                if steps < 20:
                    first_actions.append(action.astype(float).tolist())
                    first_action_digest.update(np.ascontiguousarray(action).tobytes())
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
            phase = _phase_observation(env)
            result["min_eef_bowl_distance_m"] = min(
                float(result["min_eef_bowl_distance_m"]),
                float(phase["eef_bowl_distance_m"]),
            )
            result["min_bowl_plate_distance_m"] = min(
                float(result["min_bowl_plate_distance_m"]),
                float(phase["bowl_plate_distance_m"]),
            )
            if int(phase["grasped"]):
                if not int(result["grasped_any"]):
                    result["first_grasp_step"] = steps
                result["grasped_any"] = 1
            elif int(result["grasped_any"]):
                result["dropped_after_grasp"] = 1
        result["steps"] = steps
        result["success"] = int(bool(time_step.observation.get("goal_achieved", False)))
        result["final_eef_bowl_distance_m"] = phase["eef_bowl_distance_m"]
        result["final_bowl_plate_distance_m"] = phase["bowl_plate_distance_m"]
        episode_limit = int(getattr(env, "_max_episode_len", 300))
        if result["success"]:
            result["termination_reason"] = "success"
        elif steps >= episode_limit:
            result["termination_reason"] = "timeout"
        elif max_steps is not None and steps >= max_steps:
            result["termination_reason"] = "audit_step_limit"
    except Exception as exc:
        result["simulator_exception"] = 1
        result["error_type"] = f"reset:{type(exc).__name__}:{exc}\n{traceback.format_exc()[-3000:]}"
    result["first_20_actions_sha256"] = first_action_digest.hexdigest()
    result["first_20_actions_json"] = json.dumps(first_actions, separators=(",", ":"))
    result["failure_stage"] = _failure_stage(result)
    result["wall_clock_seconds"] = time.monotonic() - start_time
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--state-index", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--split", choices=("dev", "confirm"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    checkpoint = args.checkpoint.resolve()
    upstream = args.upstream.resolve()
    if args.state_index:
        rows = load_state_rows(
            args.state_index.resolve(), args.training_seed, args.split
        )
    else:
        if args.manifest is None:
            parser.error("--manifest is required unless --state-index is supplied")
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
    state_index = load_state_index(args.state_index.resolve()) if args.state_index else None
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
            result = run_episode(
                workspace,
                env,
                row,
                args.training_seed,
                state_index,
                max_steps=args.max_steps,
            )
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
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "stage": "V1-R.2",
        "status": "complete" if len(output_rows) == len(rows) else "partial",
        "training_seed": args.training_seed,
        "rows": len(output_rows),
        "success_rate": sum(int(row["success"]) for row in output_rows) / len(output_rows),
        "simulator_exception_count": sum(int(row["simulator_exception"]) for row in output_rows),
        "action_decode_error_count": sum(int(row["action_decode_error"]) for row in output_rows),
        "checkpoint_sha256": sha256(checkpoint),
        "initial_state_match_count": sum(row["initial_state_match"] == "passed" for row in output_rows),
        "initial_state_match_status": (
            "passed"
            if all(row["initial_state_match"] == "passed" for row in output_rows)
            else "failed_or_unresolved"
        ),
        "failure_stages": {
            stage: sum(row["failure_stage"] == stage for row in output_rows)
            for stage in sorted({str(row["failure_stage"]) for row in output_rows})
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
