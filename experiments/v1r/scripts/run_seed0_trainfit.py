#!/usr/bin/env python3
"""Run D1 E1: closed-loop rollouts from the 20 frozen training initial states."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from capture_sequential_success_demos import (  # noqa: E402
    phase_flags,
    synchronize_runtime_state,
)
from run_clean_b1_2k_20_seed0 import load_snapshot  # noqa: E402
from run_clean_pointbridge import _phase_observation, normalized_input_sha256  # noqa: E402
from seed0_d1_common import (  # noqa: E402
    DEFAULT_UPSTREAM,
    RESOLVED_CONFIG,
    configure_runtime,
    read_csv,
    sha256,
    validate_preflight,
)
from state_utils import pointbridge_core_env, raw_array_sha256, refresh_pointbridge_observation  # noqa: E402


def _load_legacy(upstream: Path):
    import importlib.util

    path = ROOT / "scripts/evaluate_pointbridge_paired.py"
    spec = importlib.util.spec_from_file_location("d1_legacy_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._add_paths(upstream)
    return module, module._load_eval_module(upstream)


def _episode(
    workspace: Any,
    env: Any,
    row: dict[str, str],
    checkpoint: Path,
    root: Path,
) -> dict[str, Any]:
    import torch
    from point_bridge import utils
    from point_bridge.robot_utils.common.camera_utils import add_camera_with_offset

    started = time.monotonic()
    result: dict[str, Any] = {
        "evaluation_device": str(workspace.device),
        "control_hz": 20,
        "action_shape": "(7,)",
        "action_dtype": "float64_after_float32_policy_decode",
        "checkpoint_step": int(checkpoint.stem),
        "checkpoint_sha256": sha256(checkpoint),
        "scenario_id": row["scenario_id"],
        "layout": int(row["layout"]),
        "episode_key": row["episode_key"],
        "initial_state_sha256": "",
        "expected_initial_state_sha256": row["initial_state_sha256"],
        "source_artifact_sha256": row["source_artifact_sha256"],
        "source_pkl_sha256": row["source_pkl_sha256"],
        "model_xml_sha256": row["model_xml_sha256"],
        "object_template_sha256": row["object_template_sha256"],
        "initial_state_match": "unresolved",
        "success": 0,
        "steps": 0,
        "failure_stage": "unknown",
        "termination_reason": "unknown",
        "simulator_exception": 0,
        "action_decode_error": 0,
        "contact_observed": 0,
        "grasped_any": 0,
        "dropped_after_grasp": 0,
        "first_contact_step": -1,
        "first_grasp_step": -1,
        "min_eef_bowl_distance_m": float("inf"),
        "min_bowl_plate_distance_m": float("inf"),
        "first_20_actions_sha256": "",
        "error_type": "",
        "wall_clock_seconds": 0.0,
    }
    action_digest = __import__("hashlib").sha256()
    first_actions: list[list[float]] = []
    try:
        artifact = root / row["source_artifact"]
        xml = (root / row["model_xml"]).read_text(encoding="utf-8")
        with np.load(artifact, allow_pickle=False) as bundle:
            state = np.asarray(bundle["initial_state"], dtype=np.float64)
        import pickle
        pkl = pickle.load((root / row["source_pkl"]).open("rb"))
        template = pkl["object_point_templates"][int(row["episode_index"])]
        timestep = env.reset()
        core = pointbridge_core_env(env)
        core._env.reset_to({"states": state, "model": xml})
        # ``reset_to`` replaces the MuJoCo model with the frozen source XML.
        # Recreate the same runtime camera that Point Bridge adds after its
        # normal reset, then restore the exact state once more because adding
        # the camera rebuilds the simulator model.
        camera_added = add_camera_with_offset(
            core._env,
            "agentviewleft",
            "agentview",
            np.array([0.0, -0.12, 0.0], dtype=np.float64),
        )
        if not camera_added:
            raise RuntimeError("failed to recreate Point Bridge agentviewleft camera")
        core._env.sim.reset()
        core._env.sim.set_state_from_flattened(state)
        core._env.sim.forward()
        synchronize_runtime_state(core._env)
        timestep = refresh_pointbridge_observation(env, timestep, template, gripper_state=-1.0)
        actual = raw_array_sha256(env.sim.get_state().flatten())
        result["initial_state_sha256"] = actual
        result["initial_state_match"] = "passed" if actual == row["initial_state_sha256"] else "failed"
        if result["initial_state_match"] != "passed":
            raise RuntimeError("training initial state mismatch")
        workspace.agent.buffer_reset()
        phase = _phase_observation(env)
        result["min_eef_bowl_distance_m"] = phase["eef_bowl_distance_m"]
        result["min_bowl_plate_distance_m"] = phase["bowl_plate_distance_m"]
        step = 0
        while not timestep.last():
            try:
                if step == 0:
                    result["normalized_network_input_sha256"] = normalized_input_sha256(workspace, timestep.observation)
                with torch.no_grad(), utils.eval_mode(workspace.agent):
                    action = np.asarray(workspace.agent.act(timestep.observation, workspace.stats, step, workspace.global_step))
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise ValueError(f"expected finite action shape (7,), got {action.shape}")
                if step < 20:
                    first_actions.append(action.astype(float).tolist())
                    action_digest.update(np.ascontiguousarray(action).tobytes())
            except Exception as exc:
                result["action_decode_error"] = 1
                result["error_type"] = f"action:{type(exc).__name__}:{exc}"
                break
            try:
                timestep = env.step(action)
            except Exception as exc:
                result["simulator_exception"] = 1
                result["error_type"] = f"simulator:{type(exc).__name__}:{exc}"
                break
            step += 1
            phase = _phase_observation(env)
            result["min_eef_bowl_distance_m"] = min(result["min_eef_bowl_distance_m"], phase["eef_bowl_distance_m"])
            result["min_bowl_plate_distance_m"] = min(result["min_bowl_plate_distance_m"], phase["bowl_plate_distance_m"])
            contact, grasp, _ = phase_flags(pointbridge_core_env(env)._env)
            if contact:
                result["contact_observed"] = 1
                if result["first_contact_step"] < 0:
                    result["first_contact_step"] = step
            if grasp:
                if not result["grasped_any"]:
                    result["first_grasp_step"] = step
                result["grasped_any"] = 1
            elif result["grasped_any"]:
                result["dropped_after_grasp"] = 1
        result["steps"] = step
        result["success"] = int(bool(timestep.observation.get("goal_achieved", False)))
        if result["success"]:
            result["failure_stage"] = "success"
            result["termination_reason"] = "success"
        elif result["simulator_exception"]:
            result["failure_stage"] = "simulator"
        elif result["action_decode_error"]:
            result["failure_stage"] = "action_decode"
        elif result["min_eef_bowl_distance_m"] > 0.12:
            result["failure_stage"] = "no_approach"
        elif not result["grasped_any"]:
            result["failure_stage"] = "no_grasp"
        elif result["dropped_after_grasp"]:
            result["failure_stage"] = "post_grasp_drop"
        else:
            result["failure_stage"] = "unknown"
        if not result["success"] and result["steps"] >= int(getattr(env, "_max_episode_len", 300)):
            result["termination_reason"] = "timeout"
    except Exception as exc:
        result["simulator_exception"] = 1
        result["failure_stage"] = "simulator"
        result["error_type"] = f"reset:{type(exc).__name__}:{exc}\n{traceback.format_exc()[-1200:]}"
    result["first_20_actions_sha256"] = action_digest.hexdigest()
    result["wall_clock_seconds"] = time.monotonic() - started
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, default=RESOLVED_CONFIG)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    frozen = validate_preflight(args.upstream.resolve())
    checkpoint = args.checkpoint.resolve()
    if int(checkpoint.stem) not in (100000, 200000, 300000):
        raise ValueError("D1 E1 accepts only the three frozen checkpoints")
    rows = read_csv(args.manifest.resolve())
    if len(rows) != 20:
        raise ValueError(f"expected 20 training rows, got {len(rows)}")
    configure_runtime(args.upstream.resolve())
    runner, eval_module = _load_legacy(args.upstream.resolve())
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.resolved_config.resolve())
    cfg.eval = True; cfg.device = args.device; cfg.save_video = False; cfg.use_tb = False
    cfg.bc_weight = str(checkpoint); cfg.suite.num_eval_episodes = 1; cfg.expert_dataset = cfg.dataloader.bc_dataset
    old_cwd = Path.cwd(); os.chdir(args.upstream.resolve())
    output_rows: list[dict[str, Any]] = []
    try:
        workspace = eval_module.Workspace(cfg)
        load_snapshot(workspace, checkpoint, args.device)
        workspace.agent.train(False)
        for i, row in enumerate(rows, 1):
            result = _episode(workspace, workspace.env[int(row["layout"]) - 1], row, checkpoint, ROOT)
            output_rows.append(result)
            print(f"{i}/20 {row['scenario_id']} success={result['success']} stage={result['failure_stage']}", flush=True)
    finally:
        if "workspace" in locals():
            for env in workspace.env:
                try: env.close()
                except Exception: pass
        os.chdir(old_cwd)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(output_rows)
    summary = {
        "stage": "V1-R.2K.seed0-D1.E1", "status": "complete", "diagnostic_only": True,
        "checkpoint_step": int(checkpoint.stem), "rollouts": 20,
        "successes": sum(int(row["success"]) for row in output_rows),
        "successes_per_layout": {str(l): sum(int(row["success"]) for row in output_rows if int(row["layout"]) == l) for l in range(1,5)},
        "initial_state_matches": sum(row["initial_state_match"] == "passed" for row in output_rows),
        "simulator_exceptions": sum(int(row["simulator_exception"]) for row in output_rows),
        "action_decode_errors": sum(int(row["action_decode_error"]) for row in output_rows),
        "failure_stage_counts": {stage: sum(row["failure_stage"] == stage for row in output_rows) for stage in sorted({row["failure_stage"] for row in output_rows})},
        "checkpoint_sha256": sha256(checkpoint), "dataset_manifest_sha256": frozen["sha256"]["dataset_manifest"],
        "resolved_config_sha256": sha256(args.resolved_config.resolve()), "gate_impact": "diagnostic_only_not_a_clean_dev_gate",
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
