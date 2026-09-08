#!/usr/bin/env python3
"""Verify the V1-R.2J fallback native 7-D delta-pose contract.

This is a validation-only path.  It uses the same 20 V1-R.2I artifacts,
computes one shared min/max action range, applies the patched Point Bridge
``delta_pose`` normalization and deployment inverse, and executes the
resulting actions through the official Point Bridge environment in delta OSC
mode.  It does not train a policy or create PKL files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPSTREAM = ROOT / "third_party" / "pointbridge"
DEFAULT_MANIFEST = (
    ROOT / "outputs" / "v1r" / "sequential_success_demos_2i" / "manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "v1r" / "delta_pose_contract_2j" / "results.json"
)
REQUIRED_ROBOSUITE = "1.4.1"
REQUIRED_MUJOCO = "3.3.5"
CONTROL_FREQ_HZ = 20
ROUNDTRIP_ATOL = 1e-6
STATE_DIVERGENCE_ATOL = 1e-9

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_sequential_success_demos import (  # noqa: E402
    add_upstream_paths,
    controller_for,
    file_sha256,
    raw_array_sha256,
)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def text_sha256(value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def locate_core_environment(env: Any) -> Any:
    core = env
    while hasattr(core, "_env"):
        # Outer Point Bridge wrappers forward unknown attributes through
        # __getattr__, so inspect the instance dictionary to avoid selecting
        # a forwarding wrapper by accident.
        if "_action_mode" in vars(core) and "robot_base" in vars(core):
            return core
        core = core._env
    raise RuntimeError("could not locate Point Bridge RGBArrayAsObservationWrapper")


class _PrecomputedLanguageEncoder:
    def encode(self, _: str) -> np.ndarray:
        return np.zeros(384, dtype=np.float32)


def shared_action_stats(action_arrays: list[np.ndarray]) -> dict[str, np.ndarray]:
    values = np.concatenate(action_arrays, axis=0).astype(np.float64)
    minimum = np.min(values, axis=0)
    maximum = np.max(values, axis=0)
    scale = np.maximum(maximum - minimum, 1e-12)
    return {"min": minimum, "max": maximum, "scale": scale}


def normalize_delta_actions(
    actions: Any, stats: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, float]:
    raw = np.asarray(actions, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 7:
        raise ValueError(f"delta_pose actions must have shape (T, 7), got {raw.shape}")
    normalized_float32 = np.asarray(
        (raw - stats["min"]) / stats["scale"], dtype=np.float32
    )
    deployment = normalized_float32.astype(np.float64) * stats["scale"] + stats["min"]
    max_abs_error = float(np.max(np.abs(deployment - raw)))
    return normalized_float32, deployment, max_abs_error


def synchronized_reset(env: Any, initial_state: np.ndarray, model_xml: str) -> Any:
    core = locate_core_environment(env)
    robosuite_env = core._env
    robosuite_env.reset_to({"states": initial_state, "model": model_xml})
    gripper = robosuite_env.robots[0].gripper
    gripper.current_action = np.zeros(gripper.dof, dtype=np.float64)
    controller = controller_for(robosuite_env)
    controller.update(force=True)
    controller.update_initial_joints(controller.joint_pos)
    # ``reset_to`` is intentionally used here so the replay starts from the
    # captured simulator state without the wrapper's ten-step dummy reset.
    # Recreate only the wrapper caches that ``RGBArrayAsObservationWrapper``
    # normally initializes before its first real action.
    core.fixed_points = False
    core.object_points = {}
    core._step = 0
    core.robot_actions = None
    core.prev_gripper_state = float(-1.0)
    core.prev_gripper = deque(maxlen=5)
    pos = np.asarray(
        robosuite_env.sim.data.get_body_xpos(core._bodynames[0]), dtype=np.float64
    )
    ori = np.asarray(
        robosuite_env.sim.data.get_body_xmat(core._bodynames[0]), dtype=np.float64
    )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3], pose[:3, :3] = pos, ori
    from point_bridge.suite.mimiclabs import T_gripper, matrix_to_rotation_6d

    pose = np.linalg.inv(core.robot_base) @ pose @ T_gripper
    core.robot_base_orientation = pose[:3, :3].copy()
    core._current_pose = np.concatenate(
        [pose[:3, 3], matrix_to_rotation_6d(pose[:3, :3]), [core.prev_gripper_state]]
    )
    # The first points observation normally samples object points while the
    # wrapper is at ``_step == 0``.  Do that cache-only part explicitly; it
    # reads the restored geometry and does not advance the simulator.
    core.base_robot_points, _ = core.get_gt_points()
    return core


def phase_flags(robosuite_env: Any) -> tuple[bool, bool, bool]:
    gripper = robosuite_env.robots[0].gripper
    bowl = robosuite_env.objects_dict["bowl"]
    return (
        bool(robosuite_env.check_contact(gripper, bowl)),
        bool(robosuite_env._check_grasp(gripper, bowl)),
        bool(robosuite_env._check_success()),
    )


def replay_one(
    env: Any,
    initial_state: np.ndarray,
    model_xml: str,
    raw_actions: np.ndarray,
    normalized_actions: np.ndarray,
    deployment_actions: np.ndarray,
    reference_states_after: np.ndarray,
    roundtrip_error: float,
) -> dict[str, Any]:
    started = time.monotonic()
    core = synchronized_reset(env, initial_state, model_xml)
    robosuite_env = core._env
    controller = controller_for(robosuite_env)
    actual_initial = np.asarray(robosuite_env.sim.get_state().flatten(), dtype=np.float64)
    telemetry: list[dict[str, Any]] = []
    actual_states: list[np.ndarray] = []
    for action_index, (raw, normalized, action) in enumerate(
        zip(raw_actions, normalized_actions, deployment_actions)
    ):
        time_step = env.step(action)
        actual_state = np.asarray(robosuite_env.sim.get_state().flatten(), dtype=np.float64)
        actual_states.append(actual_state.copy())
        contact, grasped, success = phase_flags(robosuite_env)
        state_error = float(
            np.max(np.abs(actual_state - reference_states_after[action_index]))
        )
        telemetry.append(
            {
                "action_index": action_index,
                "raw_delta_action": raw,
                "normalized_delta_action_float32": normalized,
                "deployment_delta_action": action,
                "controller_use_delta": bool(controller.use_delta),
                "controller_goal_pos_m": np.asarray(controller.goal_pos),
                "controller_goal_ori_matrix": np.asarray(controller.goal_ori),
                "state_max_abs_error_vs_continuous_reference": state_error,
                "state_diverged": bool(state_error > STATE_DIVERGENCE_ATOL),
                "contact": contact,
                "close_command": bool(action[-1] > 0.0),
                "grasped": grasped,
                "task_success": success,
                "time_step_last": bool(getattr(time_step, "last", lambda: False)()),
            }
        )
    success_indices = [row["action_index"] for row in telemetry if row["task_success"]]
    divergence_indices = [
        row["action_index"] for row in telemetry if row["state_diverged"]
    ]
    return {
        "initial_state_match": bool(np.array_equal(actual_initial, initial_state)),
        "initial_state_max_abs_error": float(np.max(np.abs(actual_initial - initial_state))),
        "steps_executed": len(deployment_actions),
        "success": bool(success_indices),
        "first_success_action_index": success_indices[0] if success_indices else None,
        "first_state_divergence_action_index": (
            divergence_indices[0] if divergence_indices else None
        ),
        "state_divergence_atol": STATE_DIVERGENCE_ATOL,
        "roundtrip_max_abs_error": roundtrip_error,
        "contact_observed": any(row["contact"] for row in telemetry),
        "grasp_observed": any(row["grasped"] for row in telemetry),
        "controller_use_delta": bool(controller.use_delta),
        "telemetry": telemetry,
        "wall_clock_seconds": time.monotonic() - started,
    }


def validate_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = [record for record in manifest["records"] if record.get("accepted")]
    by_layout = {layout: [] for layout in (1, 2, 3, 4)}
    for record in records:
        by_layout[int(record["layout"])].append(record)
    if len(records) != 20 or any(len(items) != 5 for items in by_layout.values()):
        raise ValueError("delta_pose validation requires the same five accepted demos per layout")
    return [record for layout in by_layout for record in by_layout[layout]]


def make_delta_environment(pb_suite: Any, upstream: Path, task_name: str) -> Any:
    pb_suite.init_models = lambda: _PrecomputedLanguageEncoder()
    bddl_dir = (
        upstream
        / "third_party"
        / "mimiclabs"
        / "mimiclabs"
        / "mimiclabs"
        / "task_suites"
        / "new_task_suite"
    )
    envs, _ = pb_suite.make(
        bddl_dir=str(bddl_dir),
        task_names=[task_name],
        action_repeat=1,
        height=128,
        width=128,
        seed=0,
        max_episode_len=300,
        max_state_dim=100,
        eval=True,
        pixel_keys=["pixels_right"],
        num_robot_points=8,
        num_points_per_obj=128,
        robot_points_key="robot_tracks",
        object_points_key="object_tracks",
        obs_type=["points"],
        action_mode="delta_pose",
        use_vlm_points=False,
        vlm_mode="segment_depth",
        depth_type="gt",
        add_camera_from_extrinsics=False,
        camera_extrinsics_file="",
        temporal_agg_strategy="exponential_average",
        visualize_3d=False,
        use_full_scene_pcd=False,
        full_scene_num_points=512,
        max_depth_meters=2.0,
        min_z_robot_frame=0.0,
        full_scene_camera_name="cam_8_left",
    )
    if len(envs) != 1:
        raise RuntimeError(f"expected one delta_pose environment, got {len(envs)}")
    return envs[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    upstream = args.upstream.resolve()
    add_upstream_paths(upstream)
    os.environ.setdefault("MUJOCO_GL", "egl")

    import h5py
    import mujoco
    import robosuite
    from mimiclabs.mimiclabs.envs.problems import (  # noqa: F401
        MimicLabs_Lab1_Tabletop_Manipulation,
    )
    import point_bridge.suite.mimiclabs as pb_suite

    if str(getattr(robosuite, "__version__", "unknown")) != REQUIRED_ROBOSUITE:
        raise RuntimeError(f"expected robosuite {REQUIRED_ROBOSUITE}")
    if str(getattr(mujoco, "__version__", "unknown")) != REQUIRED_MUJOCO:
        raise RuntimeError(f"expected MuJoCo {REQUIRED_MUJOCO}")

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = validate_records(manifest)
    arrays_by_key: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    action_arrays: list[np.ndarray] = []
    for record in records:
        artifact = ROOT / record["artifact"]
        if file_sha256(artifact) != record["artifact_sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {artifact}")
        with np.load(artifact, allow_pickle=False) as bundle:
            arrays = {key: np.asarray(bundle[key]).copy() for key in bundle.files}
        arrays_by_key[(int(record["layout"]), str(record["demo_key"]))] = arrays
        action_arrays.append(arrays["issued_actions"])
    stats = shared_action_stats(action_arrays)

    records_out: list[dict[str, Any]] = []
    for layout in (1, 2, 3, 4):
        task_name = f"bowl_on_plate_{layout}"
        hdf5_path = (
            upstream
            / "data"
            / "mimicgen_data"
            / "bowl_on_plate"
            / task_name
            / "demo"
            / "demo.hdf5"
        )
        env = make_delta_environment(pb_suite, upstream, task_name)
        core = locate_core_environment(env)
        if core._action_mode != "delta_pose":
            raise RuntimeError(f"Point Bridge action mode is {core._action_mode!r}")
        if tuple(env.action_spec().shape) != (7,):
            raise RuntimeError(f"delta_pose action spec is {env.action_spec().shape}")
        if not bool(controller_for(core._env).use_delta):
            raise RuntimeError("delta_pose environment did not enable delta OSC")
        with h5py.File(hdf5_path, "r") as handle:
            for record in [item for item in records if int(item["layout"]) == layout]:
                key = (layout, str(record["demo_key"]))
                arrays = arrays_by_key[key]
                raw_actions = arrays["issued_actions"].astype(np.float64)
                normalized, deployment, roundtrip_error = normalize_delta_actions(
                    raw_actions, stats
                )
                source_actions = np.asarray(
                    handle["data"][record["demo_key"]]["actions"], dtype=np.float64
                )
                if not np.array_equal(source_actions[: len(raw_actions)], raw_actions):
                    raise RuntimeError(f"HDF5 action prefix mismatch for {key}")
                model_xml = (ROOT / record["model_xml"]).read_text(encoding="utf-8")
                replay = replay_one(
                    env,
                    arrays["initial_state"],
                    model_xml,
                    raw_actions,
                    normalized,
                    deployment,
                    arrays["states_after"],
                    roundtrip_error,
                )
                replay.update(
                    {
                        "layout": layout,
                        "demo_key": record["demo_key"],
                        "artifact": record["artifact"],
                        "raw_actions_sha256": raw_array_sha256(raw_actions),
                        "normalized_actions_sha256": raw_array_sha256(normalized),
                        "deployment_actions_sha256": raw_array_sha256(deployment),
                        "roundtrip_passed": bool(roundtrip_error <= ROUNDTRIP_ATOL),
                        "gripper_sign_preserved": bool(
                            np.array_equal(
                                np.sign(raw_actions[:, -1]),
                                np.sign(deployment[:, -1]),
                            )
                        ),
                    }
                )
                records_out.append(jsonable(replay))
                print(
                    f"layout={layout} demo={record['demo_key']} "
                    f"roundtrip={replay['roundtrip_max_abs_error']:.3e} "
                    f"success={replay['success']}",
                    flush=True,
                )
        env.close()

    execution_passed = sum(
        bool(record["initial_state_match"] and record["success"])
        for record in records_out
    )
    roundtrip_passed = sum(bool(record["roundtrip_passed"]) for record in records_out)
    sign_passed = sum(bool(record["gripper_sign_preserved"]) for record in records_out)
    controller_delta_passed = sum(bool(record["controller_use_delta"]) for record in records_out)
    result = {
        "stage": "V1-R.2J",
        "validation": "delta_pose_normalization_execution_roundtrip",
        "status": "passed" if execution_passed == 20 else "failed",
        "formal_threshold": {"checked": 20, "required_passed": 20},
        "runtime": {
            "robosuite": REQUIRED_ROBOSUITE,
            "mujoco": REQUIRED_MUJOCO,
            "control_freq_hz": CONTROL_FREQ_HZ,
            "official_entrypoint": "point_bridge.suite.mimiclabs.make",
            "action_mode": "delta_pose",
            "action_shape": [7],
            "control_delta": True,
        },
        "normalization": {
            "source": "same_20_v1r_2i_issued_actions",
            "dtype_before_model": "float32",
            "deployment_dtype": "float64",
            "formula": "(action - shared_min) / max(shared_max - shared_min, 1e-12)",
            "inverse": "normalized_float32 * scale + shared_min",
            "shared_min": stats["min"],
            "shared_max": stats["max"],
            "shared_scale": stats["scale"],
            "roundtrip_atol": ROUNDTRIP_ATOL,
        },
        "gates": {
            "normalization_roundtrip": {"checked": 20, "passed": roundtrip_passed},
            "gripper_sign_preservation": {"checked": 20, "passed": sign_passed},
            "official_delta_runtime": {"checked": 20, "passed": controller_delta_passed},
            "execution_replay": {"checked": 20, "passed": execution_passed},
        },
        "training_run": False,
        "records": records_out,
        "source": {
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": file_sha256(manifest_path),
        },
        "decision": (
            "delta_pose_contract_validated_for_followup_seed_only"
            if execution_passed == 20 and roundtrip_passed == 20 and sign_passed == 20
            else "delta_pose_contract_not_validated"
        ),
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(jsonable(result), indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            jsonable({key: value for key, value in result.items() if key != "records"}),
            indent=2,
        )
    )
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
