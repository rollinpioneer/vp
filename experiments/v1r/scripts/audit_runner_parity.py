#!/usr/bin/env python3
"""Audit legacy, V1-R, and official Point Bridge clean runner parity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPSTREAM = ROOT / "third_party" / "pointbridge"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_clean_pointbridge as clean_runner
from state_utils import (
    body_state_hash,
    load_state_bundle,
    load_state_index,
    raw_array_sha256,
    refresh_pointbridge_observation,
    robot_pose_hash,
)


def load_legacy_runner():
    path = ROOT / "scripts" / "evaluate_pointbridge_paired.py"
    spec = importlib.util.spec_from_file_location("vico_pointbridge_legacy_parity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_hash(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def official_frozen_episode(
    workspace: Any,
    env: Any,
    state_row: dict[str, str],
    bundle: dict[str, Any],
    max_steps: int | None,
) -> dict[str, Any]:
    """Run the policy loop used by Point Bridge Workspace.eval on a frozen state."""

    import torch
    from point_bridge import utils

    time_step = env.reset()
    env.sim.set_state_from_flattened(bundle["sim_state"])
    env.sim.forward()
    time_step = refresh_pointbridge_observation(env, time_step, bundle["object_points"])
    point_key = f"{env._object_points_key}_3d"
    robot_key = f"{env._robot_points_key}_3d"
    result: dict[str, Any] = {
        "initial_state_sha256": raw_array_sha256(env.sim.get_state().flatten()),
        "initial_state_match": False,
        "robot_pose_sha256": robot_pose_hash(env),
        "robot_points_sha256": raw_hash(time_step.observation[robot_key]),
        "object_points_sha256": raw_hash(time_step.observation[point_key]),
        "body_state_sha256": body_state_hash(env),
        "normalized_network_input_sha256": clean_runner.normalized_input_sha256(
            workspace, time_step.observation
        ),
        "first_20_actions": [],
        "steps": 0,
        "success": 0,
        "exception": "",
    }
    result["initial_state_match"] = (
        result["initial_state_sha256"] == state_row["restored_state_sha256"]
    )
    workspace.agent.buffer_reset()
    step = 0
    while not time_step.last() and (max_steps is None or step < max_steps):
        with torch.no_grad(), utils.eval_mode(workspace.agent):
            action = workspace.agent.act(
                time_step.observation,
                workspace.stats,
                step,
                workspace.global_step,
            )
        action = np.asarray(action)
        if step < 20:
            result["first_20_actions"].append(action.astype(float).tolist())
        time_step = env.step(action)
        step += 1
    result["steps"] = step
    result["success"] = int(bool(time_step.observation.get("goal_achieved", False)))
    return result


def new_frozen_episode(
    workspace: Any,
    env: Any,
    state_row: dict[str, str],
    state_index: dict[str, dict[str, str]],
    max_steps: int | None,
) -> dict[str, Any]:
    row = {
        "training_seed": "0",
        "scenario_id": state_row["scenario_id"],
        "layout": state_row["layout"],
        "simulator_seed": state_row["simulator_seed"],
    }
    record = clean_runner.run_episode(
        workspace,
        env,
        row,
        0,
        state_index,
        max_steps=max_steps,
    )
    return {
        "initial_state_sha256": record["actual_initial_state_sha256"],
        "initial_state_match": record["initial_state_match"] == "passed",
        "robot_pose_sha256": record["robot_pose_sha256"],
        "robot_points_sha256": record["robot_points_sha256"],
        "object_points_sha256": record["point_identity_sha256"],
        "body_state_sha256": record["body_state_sha256"],
        "normalized_network_input_sha256": record[
            "normalized_network_input_sha256"
        ],
        "first_20_actions": json.loads(str(record["first_20_actions_json"])),
        "steps": record["steps"],
        "success": record["success"],
        "exception": record["error_type"],
    }


def legacy_frozen_episode(
    legacy: Any,
    workspace: Any,
    env: Any,
    state_row: dict[str, str],
    bundle: dict[str, Any],
    max_steps: int | None,
) -> dict[str, Any]:
    return legacy._run_frozen_clean_branch(
        workspace,
        env,
        state=bundle["sim_state"],
        object_points=bundle["object_points"],
        expected_state_sha256=state_row["restored_state_sha256"],
        max_steps=max_steps,
    )


def action_difference(left: list[Any], right: list[Any]) -> float | None:
    if len(left) != len(right):
        return None
    if not left:
        return 0.0
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        return None
    return float(np.max(np.abs(left_array - right_array)))


def parity_record(
    scenario_id: str,
    state_row: dict[str, str],
    paths: dict[str, dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    pairs = (("legacy", "new"), ("legacy", "official"), ("new", "official"))
    differences = {
        f"{left}_vs_{right}": action_difference(
            paths[left]["first_20_actions"], paths[right]["first_20_actions"]
        )
        for left, right in pairs
    }
    exact_fields = (
        "initial_state_sha256",
        "robot_pose_sha256",
        "robot_points_sha256",
        "object_points_sha256",
        "body_state_sha256",
        "normalized_network_input_sha256",
    )
    field_equal = {
        field: len({str(paths[name][field]) for name in paths}) == 1
        for field in exact_fields
    }
    action_equal = all(
        difference is not None and difference <= tolerance
        for difference in differences.values()
    )
    success_equal = len({int(paths[name]["success"]) for name in paths}) == 1
    exception_free = all(not paths[name].get("exception") for name in paths)
    return {
        "scenario_id": scenario_id,
        "layout": int(state_row["layout"]),
        "source_old_success": int(state_row["source_old_success"]),
        "expected_restored_state_sha256": state_row["restored_state_sha256"],
        "paths": paths,
        "field_equal": field_equal,
        "action_max_abs_diff": differences,
        "action_equal": action_equal,
        "success_equal": success_equal,
        "exception_free": exception_free,
        "passed": (
            all(field_equal.values())
            and action_equal
            and success_equal
            and exception_free
            and all(bool(paths[name]["initial_state_match"]) for name in paths)
        ),
    }


def path_success_summary(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Summarize final success markers for each compared execution path."""

    summary: dict[str, dict[str, float | int]] = {}
    for name in ("legacy", "new", "official"):
        successes = sum(int(record["paths"][name]["success"]) for record in records)
        summary[name] = {
            "successes": successes,
            "scenarios": len(records),
            "success_rate": successes / len(records) if records else 0.0,
        }
    return summary


def write_report(path: Path, result: dict[str, Any]) -> None:
    records = result["records"]
    max_diff = max(
        (
            difference
            for record in records
            for difference in record["action_max_abs_diff"].values()
            if difference is not None
        ),
        default=0.0,
    )
    path_table = [
        "| runner | successes | scenarios | success rate |",
        "|---|---:|---:|---:|",
    ]
    for name, values in result["path_success"].items():
        path_table.append(
            f"| {name} | {values['successes']} | {values['scenarios']} | "
            f"{values['success_rate']:.3f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# V1-R.2F Runner Parity\n\n"
        f"状态：`{result['status']}`。设备：`{result['device']}`。"
        f"完整回合：`{result['full_episode']}`。\n\n"
        f"旧 runner 来源构成：成功 {result['source_old_outcomes']['success']}；"
        f"失败 {result['source_old_outcomes']['failure']}。\n\n"
        f"冻结场景：{len(records)}；通过：{result['passed_scenarios']}；"
        f"前 20 步最大动作绝对差：`{max_diff}`。\n\n"
        + "\n".join(path_table)
        + "\n\n"
        "比较字段：完整初态、首帧机器人点、首帧对象点、机器人位姿、"
        "body state、归一化网络输入、前 20 步动作与最终成功标记。\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--action-tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TENSORBOARD_NO_TF", "1")
    upstream = args.upstream.resolve()
    checkpoint = args.checkpoint.resolve()
    state_index = load_state_index(args.state_index.resolve())
    if args.limit is not None:
        state_index = dict(list(state_index.items())[: args.limit])
    legacy = load_legacy_runner()
    legacy._add_paths(upstream)
    eval_module = legacy._load_eval_module(upstream)
    cfg = legacy._compose_config(upstream, checkpoint, 0, device=args.device)
    cfg.save_video = False
    cfg.use_tb = False
    old_cwd = Path.cwd()
    records: list[dict[str, Any]] = []
    os.chdir(upstream)
    try:
        workspace = eval_module.Workspace(cfg)
        clean_runner.load_snapshot(workspace, checkpoint, args.device)
        workspace.agent.train(False)
        for index, (scenario_id, state_row) in enumerate(state_index.items()):
            bundle = load_state_bundle(ROOT, state_row)
            env = workspace.env[int(state_row["layout"]) - 1]
            paths: dict[str, dict[str, Any]] = {}
            for name, function in (
                ("legacy", legacy_frozen_episode),
                ("new", new_frozen_episode),
                ("official", official_frozen_episode),
            ):
                legacy._set_seed(int(state_row["simulator_seed"]))
                try:
                    if name == "legacy":
                        value = function(
                            legacy, workspace, env, state_row, bundle, args.max_steps
                        )
                    elif name == "new":
                        value = function(
                            workspace, env, state_row, state_index, args.max_steps
                        )
                    else:
                        value = function(
                            workspace, env, state_row, bundle, args.max_steps
                        )
                    value.setdefault("exception", "")
                    paths[name] = value
                except Exception as exc:
                    paths[name] = {
                        "initial_state_sha256": "",
                        "initial_state_match": False,
                        "robot_pose_sha256": "",
                        "robot_points_sha256": "",
                        "object_points_sha256": "",
                        "body_state_sha256": "",
                        "normalized_network_input_sha256": "",
                        "first_20_actions": [],
                        "steps": 0,
                        "success": 0,
                        "exception": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-3000:]}",
                    }
            record = parity_record(
                scenario_id, state_row, paths, args.action_tolerance
            )
            records.append(record)
            print(
                f"{index + 1}/{len(state_index)} {scenario_id} passed={record['passed']} "
                f"diff={record['action_max_abs_diff']}",
                flush=True,
            )
    finally:
        if "workspace" in locals():
            for env in workspace.env:
                env.close()
        os.chdir(old_cwd)

    full_episode = args.max_steps is None
    passed = all(record["passed"] for record in records)
    result = {
        "stage": "V1-R.2F",
        "audit": "runner_parity",
        "status": "passed" if passed and full_episode else "smoke_passed" if passed else "failed",
        "device": args.device,
        "full_episode": full_episode,
        "max_steps": args.max_steps,
        "action_tolerance": args.action_tolerance,
        "scenarios": len(records),
        "passed_scenarios": sum(record["passed"] for record in records),
        "source_old_outcomes": {
            "success": sum(int(record["source_old_success"]) for record in records),
            "failure": sum(not int(record["source_old_success"]) for record in records),
        },
        "path_success": path_success_summary(records),
        "checkpoint_sha256": clean_runner.sha256(checkpoint),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(args.report, result)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
