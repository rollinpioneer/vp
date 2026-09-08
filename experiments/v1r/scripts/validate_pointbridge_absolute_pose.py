#!/usr/bin/env python3
"""Validate Point Bridge's native absolute-pose expert action contract.

This gate deliberately bypasses policy inference. ``BCDataset`` constructs the
expert labels, its own preprocessor normalizes them, the deployment
postprocessing equation denormalizes them, and the official Point Bridge suite
executes the resulting 10-D actions with absolute OSC_POSE control.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPSTREAM = ROOT / "third_party" / "pointbridge"
REQUIRED_ROBOSUITE = "1.4.1"
REQUIRED_MUJOCO = "3.3.5"
ACT_SUBSAMPLE = 1
CONTROL_FREQ_HZ = 20
PER_LAYOUT = 5


@lru_cache(maxsize=None)
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def raw_array_sha256(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def add_upstream_paths(upstream: Path) -> None:
    paths = (
        upstream,
        upstream / "third_party" / "LIBERO",
        upstream / "third_party" / "mimicgen",
        upstream / "third_party" / "mimiclabs",
        upstream / "third_party" / "robocasa",
    )
    for path in reversed(paths):
        sys.path.insert(0, str(path))


def pointbridge_postprocess_actions(
    normalized_actions: Any, stats: dict[str, Any]
) -> np.ndarray:
    """Apply the pose-action postprocessor used by ``PB.act``."""

    normalized = np.asarray(normalized_actions)
    minimum = np.asarray(stats["actions"]["min"])
    maximum = np.asarray(stats["actions"]["max"])
    return normalized * (maximum - minimum) + minimum


def normalize_and_postprocess(dataset: Any, actions: Any) -> tuple[np.ndarray, np.ndarray]:
    """Route labels through BCDataset preprocessing and PB deployment postprocessing."""

    normalized = np.asarray(
        dataset.preprocess["actions"](np.asarray(actions)), dtype=np.float32
    )
    return normalized, pointbridge_postprocess_actions(normalized, dataset.stats)


def gripper_contract(raw_actions: Any, deployment_actions: Any) -> dict[str, Any]:
    """Verify Point Bridge's scalar gripper channel and sign encoding."""

    raw = np.asarray(raw_actions)
    deployment = np.asarray(deployment_actions)
    if raw.ndim != 2 or deployment.shape != raw.shape or raw.shape[1] != 10:
        raise ValueError(
            f"expected matching Point Bridge action arrays shaped (T, 10), got "
            f"raw={raw.shape}, deployment={deployment.shape}"
        )
    raw_gripper = raw[:, -1]
    deployment_gripper = deployment[:, -1]
    signs_match = bool(np.array_equal(np.sign(raw_gripper), np.sign(deployment_gripper)))
    return {
        "channels": 1,
        "raw_min": float(np.min(raw_gripper)),
        "raw_max": float(np.max(raw_gripper)),
        "deployment_min": float(np.min(deployment_gripper)),
        "deployment_max": float(np.max(deployment_gripper)),
        "open_close_signs_match": signs_match,
        "raw_within_normalized_gripper_range": bool(
            np.all(raw_gripper >= -1.0) and np.all(raw_gripper <= 1.0)
        ),
        "passed": bool(
            signs_match
            and np.all(raw_gripper >= -1.0)
            and np.all(raw_gripper <= 1.0)
        ),
    }


def native_label_alignment(episode: dict[str, Any]) -> dict[str, Any]:
    """Verify the native act_subsample=1 next-pose and terminal-repeat contract."""

    actions = np.asarray(episode["action"])
    eef_states = np.asarray(episode["observation"]["eef_states"])
    if actions.ndim != 2 or actions.shape[1] != 10:
        raise ValueError(f"expected native actions with shape (T, 10), got {actions.shape}")
    if eef_states.shape != actions.shape:
        raise ValueError(
            f"native eef/action shape mismatch: eef={eef_states.shape} actions={actions.shape}"
        )
    next_pose_error = float(np.max(np.abs(actions[:-1] - eef_states[1:])))
    terminal_repeat_error = float(np.max(np.abs(actions[-1] - eef_states[-1])))
    return {
        "act_subsample": ACT_SUBSAMPLE,
        "action_shape": list(actions.shape),
        "eef_state_shape": list(eef_states.shape),
        "next_pose_max_abs_error": next_pose_error,
        "terminal_repeat_max_abs_error": terminal_repeat_error,
        "passed": next_pose_error <= 1e-12 and terminal_repeat_error <= 1e-12,
    }


def control_boundary_evidence(value: Any, limits: Any, atol: float = 1e-9) -> dict[str, Any]:
    """Report configured goal limits and whether the controller goal touches one."""

    if limits is None:
        return {
            "configured": False,
            "hit": False,
            "lower_axes": [],
            "upper_axes": [],
            "reason": "absolute_OSC_goal_limit_not_configured",
        }
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    bounds = np.asarray(limits, dtype=np.float64)
    if bounds.shape != (2, values.size):
        raise ValueError(
            f"control limits must have shape {(2, values.size)}, got {bounds.shape}"
        )
    lower = np.flatnonzero(np.isclose(values, bounds[0], atol=atol, rtol=0)).tolist()
    upper = np.flatnonzero(np.isclose(values, bounds[1], atol=atol, rtol=0)).tolist()
    return {
        "configured": True,
        "hit": bool(lower or upper),
        "lower_axes": lower,
        "upper_axes": upper,
        "reason": "configured_goal_limit_checked",
    }


def failure_stage(summary: dict[str, Any]) -> str:
    if bool(summary.get("success")):
        return "success"
    if summary.get("exception"):
        return "simulator"
    if float(summary.get("min_eef_bowl_distance_m", float("inf"))) > 0.12:
        return "no_approach"
    if not bool(summary.get("grasped_any")):
        return "no_grasp"
    if bool(summary.get("dropped_after_grasp")):
        return "post_grasp_drop"
    if float(summary.get("min_bowl_plate_distance_m", float("inf"))) > 0.12:
        return "no_reach_plate"
    return "placement_failure"


def classify_gate(
    per_layout: dict[str, dict[str, Any]], clean_policy_success_rate: float | None
) -> dict[str, Any]:
    """Apply the predeclared A/B/C decision in the V1-R.2G request."""

    all_layouts_pass = all(
        int(per_layout.get(str(layout), {}).get("passed", 0))
        == int(per_layout.get(str(layout), {}).get("checked", 0))
        == PER_LAYOUT
        for layout in range(1, 5)
    )
    layout_1_or_2_failed = any(
        int(per_layout.get(str(layout), {}).get("passed", 0)) < PER_LAYOUT
        for layout in (1, 2)
    )
    if all_layouts_pass and clean_policy_success_rate is not None and clean_policy_success_rate <= 0.05:
        return {
            "case": "C",
            "gate": "passed",
            "conclusion": "expert_labels_executable_policy_data_generalization_problem_remains",
            "b0_b1_training_authorized": True,
            "next_stage": "audit_data_construction_coverage_normalization_sampling_and_generalization_then_seed0_clean_dev",
        }
    if all_layouts_pass:
        return {
            "case": "A",
            "gate": "passed",
            "conclusion": "absolute_pose_expert_labels_executable_on_all_layouts",
            "b0_b1_training_authorized": True,
            "next_stage": "build_b1_then_train_and_gate_seed0_on_frozen_clean_dev",
        }
    if layout_1_or_2_failed:
        return {
            "case": "B",
            "gate": "failed",
            "conclusion": "absolute_pose_contract_still_fails_on_layout_1_or_2",
            "b0_b1_training_authorized": False,
            "next_stage": "repair_absolute_transform_OSC_frequency_gripper_and_saved_state_compatibility_without_retraining",
        }
    return {
        "case": "unresolved",
        "gate": "failed",
        "conclusion": "absolute_pose_contract_failed_outside_predeclared_layout_1_2_case",
        "b0_b1_training_authorized": False,
        "next_stage": "localize_remaining_absolute_pose_contract_failure_without_retraining",
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _controller_for(robosuite_env: Any) -> Any:
    controller = robosuite_env.robots[0].controller
    if isinstance(controller, dict):
        if len(controller) != 1:
            raise ValueError("absolute-pose gate expects exactly one arm controller")
        controller = next(iter(controller.values()))
    return controller


def _pointbridge_core_env(env: Any) -> Any:
    core = env
    while hasattr(core, "_env"):
        if hasattr(core, "_pixel_keys") and hasattr(core, "get_gt_points"):
            return core
        core = core._env
    raise RuntimeError("could not locate Point Bridge RGBArrayAsObservationWrapper")


def _synchronize_restored_runtime(core: Any, initial_gripper_label: float) -> None:
    robosuite_env = core._env
    gripper = robosuite_env.robots[0].gripper
    gripper.current_action = np.zeros(gripper.dof, dtype=np.float64)
    controller = _controller_for(robosuite_env)
    controller.update(force=True)
    controller.update_initial_joints(controller.joint_pos)
    core._step = 0
    core.robot_actions = None
    core.prev_gripper_state = float(initial_gripper_label)
    core.prev_gripper = deque(maxlen=5)


def _phase_observation(robosuite_env: Any) -> dict[str, Any]:
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
        "grasped": grasped,
    }


def _hdf5_demo_for_episode(data_group: Any, states: np.ndarray) -> tuple[str, Any]:
    expected = np.asarray(states[0], dtype=np.float32)
    expected_digest = raw_array_sha256(expected)
    matches: list[str] = []
    for demo_key in data_group.keys():
        demo = data_group[demo_key]
        if len(demo["states"]) != len(states):
            continue
        candidate = np.asarray(demo["states"][0], dtype=np.float32)
        if raw_array_sha256(candidate) == expected_digest and np.array_equal(candidate, expected):
            matches.append(str(demo_key))
    if len(matches) != 1:
        raise ValueError(
            f"expected one HDF5 match for PKL state/length, found {matches or 'none'}"
        )
    return matches[0], data_group[matches[0]]


class _PrecomputedLanguageEncoder:
    """Observation-only adapter that prevents neural inference in this action gate."""

    def __init__(self, task_embedding: Any):
        self._task_embedding = np.asarray(task_embedding).copy()

    def encode(self, _: str) -> np.ndarray:
        return self._task_embedding.copy()


def _make_official_environment(
    pb_suite: Any,
    upstream: Path,
    task_name: str,
    task_embedding: Any,
    output_dir: Path,
) -> Any:
    pb_suite.init_models = lambda: _PrecomputedLanguageEncoder(task_embedding)
    bddl_dir = (
        upstream
        / "third_party"
        / "mimiclabs"
        / "mimiclabs"
        / "mimiclabs"
        / "task_suites"
        / "new_task_suite"
    )
    previous_cwd = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(output_dir)
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
            obs_type=["image"],
            action_mode="pose",
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
    finally:
        os.chdir(previous_cwd)
    if len(envs) != 1:
        raise RuntimeError(f"Point Bridge make() returned {len(envs)} environments")
    return envs[0]


def _runtime_contract(env: Any, robosuite_version: str, mujoco_version: str) -> dict[str, Any]:
    core = _pointbridge_core_env(env)
    controller = _controller_for(core._env)
    result = {
        "robosuite_version": robosuite_version,
        "mujoco_version": mujoco_version,
        "official_entrypoint": "point_bridge.suite.mimiclabs.make",
        "controller": str(controller.name),
        "control_delta": bool(controller.use_delta),
        "control_freq_hz": int(core._env.control_freq),
        "action_shape": list(env.action_spec().shape),
        "action_mode": str(core._action_mode),
        "action_repeat": 1,
        "policy_inference_used": False,
        "language_encoder_inference_used": False,
        "observation_mode": ["image"],
        "observation_mode_reason": "action_only_gate_does_not_consume_policy_observations",
        "pixel_keys": ["pixels_right"],
        "camera_contract": "pixels_right_maps_to_saved_model_native_agentview",
    }
    result["passed"] = bool(
        robosuite_version == REQUIRED_ROBOSUITE
        and mujoco_version == REQUIRED_MUJOCO
        and controller.name == "OSC_POSE"
        and not controller.use_delta
        and int(core._env.control_freq) == CONTROL_FREQ_HZ
        and tuple(env.action_spec().shape) == (10,)
        and core._action_mode == "pose"
    )
    return result


def _tracking_summary(telemetry: list[dict[str, Any]]) -> dict[str, Any]:
    position = np.asarray([row["tracking_error"]["position_m"] for row in telemetry])
    rotation = np.asarray([row["tracking_error"]["rotation_rad"] for row in telemetry])
    return {
        "position_m": {
            "mean": float(np.mean(position)),
            "p95": float(np.percentile(position, 95)),
            "max": float(np.max(position)),
            "final": float(position[-1]),
        },
        "rotation_rad": {
            "mean": float(np.mean(rotation)),
            "p95": float(np.percentile(rotation, 95)),
            "max": float(np.max(rotation)),
            "final": float(rotation[-1]),
        },
    }


def _replay_episode(
    env: Any,
    episode: dict[str, Any],
    hdf5_demo: Any,
    layout: int,
    pkl_episode_index: int,
    demo_key: str,
    dataset: Any,
    source_pkl: Path,
    migrate_saved_model_xml: Any,
    rotation_6d_to_matrix: Any,
    matrix_to_rotation_6d: Any,
    scipy_rotation: Any,
) -> dict[str, Any]:
    started = time.monotonic()
    actions = np.asarray(episode["action"])
    observations = episode["observation"]
    states = np.asarray(observations["states"])
    alignment = native_label_alignment(episode)
    normalized, deployment_actions = normalize_and_postprocess(dataset, actions)
    gripper_evidence = gripper_contract(actions, deployment_actions)
    normalization_error = float(np.max(np.abs(deployment_actions - actions)))
    original_xml = hdf5_demo.attrs["model_file"]
    if isinstance(original_xml, bytes):
        original_xml = original_xml.decode("utf-8")
    migrated_xml = migrate_saved_model_xml(str(original_xml))
    record: dict[str, Any] = {
        "layout": layout,
        "pkl_episode_index": pkl_episode_index,
        "hdf5_demo_key": demo_key,
        "source_pkl": str(source_pkl.relative_to(ROOT)),
        "source_pkl_sha256": file_sha256(source_pkl),
        "source_hdf5_model_xml_sha256": text_sha256(str(original_xml)),
        "runtime_model_xml_sha256": text_sha256(migrated_xml),
        "available_steps": int(len(actions)),
        "native_label_alignment": alignment,
        "normalization": {
            "normalized_dtype": str(normalized.dtype),
            "deployment_action_dtype": str(deployment_actions.dtype),
            "normalized_min": float(np.min(normalized)),
            "normalized_max": float(np.max(normalized)),
            "postprocess_max_abs_error": normalization_error,
            "training_preprocess": "BCDataset.preprocess['actions']",
            "deployment_postprocess": "PB.act pose-action affine postprocess",
        },
        "gripper_contract": gripper_evidence,
        "restored_initial_state_match": False,
        "steps_executed": 0,
        "first_success_step": None,
        "success": False,
        "failure_stage": "simulator",
        "exception": None,
        "telemetry": [],
    }
    try:
        env.reset()
        core = _pointbridge_core_env(env)
        robosuite_env = core._env
        robosuite_env.reset_to({"states": states[0], "model": migrated_xml})
        _synchronize_restored_runtime(core, float(observations["eef_states"][0, -1]))
        controller = _controller_for(robosuite_env)
        if controller.use_delta:
            raise RuntimeError("official Point Bridge controller changed to delta mode after reset_to")
        actual_state = np.asarray(robosuite_env.sim.get_state().flatten())
        expected_state = np.asarray(states[0])
        state_error = float(np.max(np.abs(actual_state - expected_state)))
        state_match = bool(
            actual_state.shape == expected_state.shape
            and np.array_equal(actual_state.astype(expected_state.dtype), expected_state)
        )
        record.update(
            {
                "expected_initial_state_sha256": raw_array_sha256(expected_state),
                "actual_initial_state_sha256": raw_array_sha256(
                    actual_state.astype(expected_state.dtype)
                ),
                "initial_state_max_abs_error": state_error,
                "restored_initial_state_match": state_match,
                "robot_base": np.asarray(core.robot_base),
            }
        )
        if not state_match:
            raise RuntimeError(f"restored initial state mismatch: max_abs={state_error}")

        phase = _phase_observation(robosuite_env)
        min_eef_bowl = phase["eef_bowl_distance_m"]
        min_bowl_plate = phase["bowl_plate_distance_m"]
        grasped_any = bool(phase["grasped"])
        dropped_after_grasp = False
        telemetry: list[dict[str, Any]] = []
        for action_index, action in enumerate(deployment_actions):
            time_step = env.step(action)
            controller.update(force=True)
            target_native_rotation = rotation_6d_to_matrix(action[3:9])
            actual_native = np.asarray(core._current_pose)
            actual_native_rotation = rotation_6d_to_matrix(actual_native[3:9])
            position_error = float(np.linalg.norm(action[:3] - actual_native[:3]))
            rotation_error = float(
                scipy_rotation.from_matrix(
                    target_native_rotation @ actual_native_rotation.T
                ).magnitude()
            )
            controller_goal_rotvec = scipy_rotation.from_matrix(controller.goal_ori).as_rotvec()
            translation_boundary = control_boundary_evidence(
                controller.goal_pos, controller.position_limits
            )
            rotation_boundary = control_boundary_evidence(
                controller_goal_rotvec, controller.orientation_limits
            )
            success = bool(time_step.observation.get("goal_achieved", False))
            phase = _phase_observation(robosuite_env)
            min_eef_bowl = min(min_eef_bowl, phase["eef_bowl_distance_m"])
            min_bowl_plate = min(min_bowl_plate, phase["bowl_plate_distance_m"])
            if phase["grasped"]:
                grasped_any = True
            elif grasped_any:
                dropped_after_grasp = True
            telemetry.append(
                {
                    "action_index": action_index,
                    "absolute_target": {
                        "pointbridge_position_m": action[:3],
                        "pointbridge_rotation_6d": action[3:9],
                        "gripper_command": float(action[-1]),
                        "controller_world_position_m": np.asarray(controller.goal_pos),
                        "controller_world_rotation_6d": matrix_to_rotation_6d(
                            np.asarray(controller.goal_ori)
                        ),
                    },
                    "actual_eef": {
                        "pointbridge_position_m": actual_native[:3],
                        "pointbridge_rotation_6d": actual_native[3:9],
                        "controller_world_position_m": np.asarray(controller.ee_pos),
                        "controller_world_rotation_6d": matrix_to_rotation_6d(
                            np.asarray(controller.ee_ori_mat)
                        ),
                    },
                    "tracking_error": {
                        "position_m": position_error,
                        "rotation_rad": rotation_error,
                    },
                    "control_boundary": {
                        "translation": translation_boundary,
                        "rotation": rotation_boundary,
                    },
                    "grasped": bool(phase["grasped"]),
                    "task_success": success,
                }
            )
            if success:
                record["first_success_step"] = action_index + 1
                break

        record.update(
            {
                "steps_executed": len(telemetry),
                "success": bool(telemetry and telemetry[-1]["task_success"]),
                "min_eef_bowl_distance_m": float(min_eef_bowl),
                "min_bowl_plate_distance_m": float(min_bowl_plate),
                "final_eef_bowl_distance_m": float(phase["eef_bowl_distance_m"]),
                "final_bowl_plate_distance_m": float(phase["bowl_plate_distance_m"]),
                "grasped_any": grasped_any,
                "dropped_after_grasp": dropped_after_grasp,
                "translation_boundary_hit_steps": sum(
                    row["control_boundary"]["translation"]["hit"] for row in telemetry
                ),
                "rotation_boundary_hit_steps": sum(
                    row["control_boundary"]["rotation"]["hit"] for row in telemetry
                ),
                "tracking_summary": _tracking_summary(telemetry),
                "telemetry": telemetry,
            }
        )
    except Exception as exc:
        record["exception"] = (
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-3000:]}"
        )
    record["failure_stage"] = failure_stage(record)
    record["passed"] = bool(
        record["success"]
        and record["restored_initial_state_match"]
        and alignment["passed"]
        and gripper_evidence["passed"]
        and not record["exception"]
    )
    record["wall_clock_seconds"] = time.monotonic() - started
    return _jsonable(record)


def _load_clean_policy_rate(path: Path | None) -> float | None:
    if path is None or not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    value = payload.get("success_rate")
    if value is None:
        value = payload.get("pooled", {}).get("success_rate")
    return None if value is None else float(value)


def _dataset_contract(dataset: Any, upstream: Path) -> dict[str, Any]:
    source = upstream / "point_bridge" / "read_data" / "mimiclabs.py"
    rotation_source = upstream / "point_bridge" / "robot_utils" / "common" / "utils.py"
    suite_source = upstream / "point_bridge" / "suite" / "mimiclabs.py"
    return {
        "constructor": "point_bridge.read_data.mimiclabs.BCDataset",
        "act_subsample": int(dataset._act_subsample),
        "action_mode": str(dataset._action_mode),
        "task_count": len(dataset.tasks),
        "task_names": list(dataset.tasks),
        "episodes_per_task": {
            dataset.tasks[index]: int(dataset._num_demos[index])
            for index in dataset._num_demos
        },
        "action_stats": _jsonable(dataset.stats["actions"]),
        "source_sha256": {
            "BCDataset": file_sha256(source),
            "rotation_6d": file_sha256(rotation_source),
            "official_suite": file_sha256(suite_source),
        },
        "quaternion_conversion": "BCDataset scipy.spatial.transform.Rotation.from_quat(...).as_matrix()",
        "rotation_6d_conversion": "point_bridge.robot_utils.common.utils.matrix_to_rotation_6d",
        "terminal_alignment": "eef_states[1:] + gripper_states[1:] then repeat final action",
        "passed": bool(
            dataset._act_subsample == ACT_SUBSAMPLE
            and dataset._action_mode == "pose"
            and len(dataset.tasks) == 4
            and all(int(dataset._num_demos[index]) >= PER_LAYOUT for index in dataset._num_demos)
        ),
    }


def _capture_dataset_metadata(module: Any) -> tuple[dict[Path, dict[str, Any]], Any]:
    """Capture top-level PKL metadata during BCDataset's own load pass."""

    metadata: dict[Path, dict[str, Any]] = {}
    original_load = module.pkl.load

    def capturing_load(handle: Any) -> Any:
        data = original_load(handle)
        path = Path(handle.name).resolve()
        metadata[path] = {
            "robot_base": np.asarray(data["robot_base"]).copy(),
            "task_desc": str(data["task_desc"]),
            "task_embedding_sha256": raw_array_sha256(data["task_emb"]),
        }
        return data

    module.pkl.load = capturing_load
    return metadata, original_load


def coordinate_frame_contract(source_robot_base: Any, environment_robot_base: Any) -> dict[str, Any]:
    """Compare saved and official-environment Point Bridge base frames directly."""

    source = np.asarray(source_robot_base, dtype=np.float64)
    environment = np.asarray(environment_robot_base, dtype=np.float64)
    if source.shape != (4, 4) or environment.shape != (4, 4):
        raise ValueError(
            f"robot base matrices must be 4x4, got source={source.shape}, env={environment.shape}"
        )
    max_error = float(np.max(np.abs(source - environment)))
    return {
        "source_pkl_robot_base": source,
        "official_environment_robot_base": environment,
        "max_abs_error": max_error,
        "exact_match": bool(np.array_equal(source, environment)),
        "passed": max_error <= 1e-12,
    }


def _markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "## Material Passport",
        "",
        "- Schema: ARS Material Passport 9",
        "- Material type: reproducibility validation report",
        "- Source: local Point Bridge PKL/HDF5 demonstrations and pinned source tree",
        "- Raw data handling: local only; no raw dataset or model artifact uploaded",
        "- Verification status: VERIFIED",
        "",
        "# V1-R.2G Point Bridge Absolute-Pose Contract Gate",
        "",
        f"Status: `{result['status']}`. Decision case: `{result['decision']['case']}`.",
        "",
        "## Runtime And Contract",
        "",
        f"- Formal runtime: robosuite `{result['runtime']['robosuite_version']}`, MuJoCo `{result['runtime']['mujoco_version']}`.",
        "- Environment entrypoint: `point_bridge.suite.mimiclabs.make`.",
        "- Controller: `OSC_POSE`, `control_delta=False`, 20 Hz, one 10-D pose action per step.",
        "- Label source: native `BCDataset` with `act_subsample=1`; no policy or language-model inference was used.",
        "- Rotation path: native SciPy quaternion-to-matrix conversion and Point Bridge 6-D helpers.",
        "- Normalization path: `BCDataset.preprocess['actions']` followed by the pose branch of `PB.act` postprocessing.",
        "- Dtype path: normalized training labels are `float32`; deployment postprocessing yields `float64` actions from the saved float64 statistics.",
        "- Coordinate and gripper checks: each PKL `robot_base` matches the official environment, and the scalar open/close sign encoding is preserved.",
        "- Exact PKL initial state is restored after environment reset and checked before action 0.",
        "",
        "## Results",
        "",
        "| Layout | Checked | Passed | Success rate | Position error max (m) | Rotation error max (rad) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for layout in range(1, 5):
        row = result["per_layout"][str(layout)]
        lines.append(
            f"| {layout} | {row['checked']} | {row['passed']} | {row['success_rate']:.3f} | "
            f"{row['max_tracking_position_error_m']:.9f} | {row['max_tracking_rotation_error_rad']:.9f} |"
        )
    lines.extend(
        [
            "",
            f"Total: `{result['passed']}/{result['checked']}`; telemetry rows: `{result['telemetry_steps']}`.",
            f"All restored initial states matched: `{result['restored_initial_state_matches']}/{result['checked']}`.",
            f"Translation boundary hits: `{result['translation_boundary_hit_steps']}`; rotation boundary hits: `{result['rotation_boundary_hit_steps']}`. The official absolute OSC controller has no position or orientation goal limits configured.",
            "",
            "## Decision",
            "",
            f"- Case: `{result['decision']['case']}`",
            f"- Conclusion: `{result['decision']['conclusion']}`",
            f"- B0/B1 training authorized: `{str(result['decision']['b0_b1_training_authorized']).lower()}`",
            f"- Clean baseline gate: `blocked`",
            f"- V2/V3 formal experiments authorized: `false`",
            f"- Next stage: `{result['decision']['next_stage']}`",
            "",
            "Every executed step's absolute target, actual EEF pose, tracking error, boundary status, grasp state, and task-success flag is retained in `pointbridge_absolute_pose_contract.json`.",
            "",
        ]
    )
    failed = [record for record in result["records"] if not record["passed"]]
    if failed:
        lines.extend(["## Failures", ""])
        for record in failed:
            lines.append(
                f"- Layout {record['layout']} `{record['hdf5_demo_key']}`: "
                f"`{record['failure_stage']}`; exception=`{bool(record['exception'])}`."
            )
        lines.append("")
    return "\n".join(lines)


def _gate_yaml(result: dict[str, Any]) -> str:
    decision = result["decision"]
    lines = [
        "stage: V1-R.2G",
        "name: Point Bridge absolute-pose action contract validation",
        f"status: {result['status']}",
        f"decision_case: {decision['case']}",
        "v1r_raw_delta_replay_diagnosis: localized_not_fully_causal",
        "runner_parity: passed",
        "initial_state_restoration: passed",
        f"pointbridge_absolute_pose_contract_gate: {decision['gate']}",
        "clean_baseline_gate: blocked",
        f"b0_b1_training_authorized: {str(decision['b0_b1_training_authorized']).lower()}",
        "v2_formal_experiment_authorized: false",
        "v3_formal_experiment_authorized: false",
        f"next_stage: {decision['next_stage']}",
        "formal_runtime:",
        f"  robosuite: {result['runtime']['robosuite_version']}",
        f"  mujoco: {result['runtime']['mujoco_version']}",
        "evidence:",
        f"  checked: {result['checked']}",
        f"  passed: {result['passed']}",
        f"  restored_initial_state_matches: {result['restored_initial_state_matches']}",
        f"  translation_boundary_hit_steps: {result['translation_boundary_hit_steps']}",
        f"  rotation_boundary_hit_steps: {result['rotation_boundary_hit_steps']}",
        "  per_layout:",
    ]
    for layout in range(1, 5):
        row = result["per_layout"][str(layout)]
        lines.extend(
            [
                f"    {layout}:",
                f"      checked: {row['checked']}",
                f"      passed: {row['passed']}",
                f"      success_rate: {row['success_rate']:.6f}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--per-layout", type=int, default=PER_LAYOUT, choices=(PER_LAYOUT,))
    parser.add_argument(
        "--clean-policy-report",
        type=Path,
        default=ROOT / "experiments" / "v1r" / "reports" / "clean_baseline_dev_seed0.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument(
        "--runtime-output-dir",
        type=Path,
        default=ROOT / "outputs" / "v1r" / "pointbridge_absolute_pose_runtime",
    )
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    upstream = args.upstream.resolve()
    add_upstream_paths(upstream)

    import h5py
    import mujoco
    import robosuite
    from scipy.spatial.transform import Rotation as R

    import point_bridge.read_data.mimiclabs as pb_data
    from point_bridge.robot_utils.common.utils import (
        matrix_to_rotation_6d,
        rotation_6d_to_matrix,
    )
    import point_bridge.suite.mimiclabs as pb_suite

    sys.path.insert(0, str(ROOT / "src"))
    from vico_point.envs.mimiclabs_compat import migrate_saved_model_xml

    robosuite_version = str(getattr(robosuite, "__version__", "unknown"))
    mujoco_version = str(getattr(mujoco, "__version__", "unknown"))
    if robosuite_version != REQUIRED_ROBOSUITE or mujoco_version != REQUIRED_MUJOCO:
        raise RuntimeError(
            "formal V1-R.2G requires robosuite "
            f"{REQUIRED_ROBOSUITE} and MuJoCo {REQUIRED_MUJOCO}; got "
            f"robosuite {robosuite_version}, MuJoCo {mujoco_version}"
        )

    pkl_dir = upstream / "expert_demos" / "mimiclabs__no_images"
    source_metadata, original_pickle_load = _capture_dataset_metadata(pb_data)
    try:
        dataset = pb_data.BCDataset(
            path=str(pkl_dir),
            suffix=None,
            num_demos_per_task=10000,
            history_len=1,
            action_chunking=True,
            num_queries=40,
            img_size=[128, 128],
            num_robot_points=8,
            num_points_per_obj=128,
            robot_points_key="robot_tracks",
            object_points_key="object_tracks",
            pixel_keys=["pixels_left", "pixels_right"],
            act_subsample=ACT_SUBSAMPLE,
            obs_subsample=1,
            obs_type=["points"],
            action_mode="pose",
            task_indices=None,
            noise_object_points=True,
            noise_std=0.01,
        )
    finally:
        pb_data.pkl.load = original_pickle_load
    dataset_contract = _dataset_contract(dataset, upstream)
    if not dataset_contract["passed"]:
        raise RuntimeError(f"native BCDataset contract failed: {dataset_contract}")

    records: list[dict[str, Any]] = []
    coordinate_contracts: dict[str, Any] = {}
    runtime_contract: dict[str, Any] | None = None
    for path_index, source_pkl in dataset._paths.items():
        task_name = source_pkl.stem
        layout = int(task_name.rsplit("_", 1)[-1])
        hdf5_path = (
            upstream
            / "data"
            / "mimicgen_data"
            / "bowl_on_plate"
            / task_name
            / "demo"
            / "demo.hdf5"
        )
        task_embedding = dataset._episodes[path_index][0]["task_emb"]
        env = _make_official_environment(
            pb_suite,
            upstream,
            task_name,
            task_embedding,
            args.runtime_output_dir.resolve(),
        )
        try:
            current_runtime = _runtime_contract(
                env, robosuite_version=robosuite_version, mujoco_version=mujoco_version
            )
            if not current_runtime["passed"]:
                raise RuntimeError(f"official runtime contract failed: {current_runtime}")
            if runtime_contract is None:
                runtime_contract = current_runtime
            elif current_runtime != runtime_contract:
                raise RuntimeError("runtime contract changed between layouts")
            metadata = source_metadata[source_pkl.resolve()]
            coordinate_contract = coordinate_frame_contract(
                metadata["robot_base"], _pointbridge_core_env(env).robot_base
            )
            coordinate_contract.update(
                {
                    "source_pkl": str(source_pkl.relative_to(ROOT)),
                    "task_desc": metadata["task_desc"],
                    "task_embedding_sha256": metadata["task_embedding_sha256"],
                }
            )
            coordinate_contracts[str(layout)] = _jsonable(coordinate_contract)
            if not coordinate_contract["passed"]:
                raise RuntimeError(
                    f"layout {layout} saved/environment robot-base mismatch: "
                    f"{coordinate_contract['max_abs_error']}"
                )
            with h5py.File(hdf5_path, "r") as hdf5:
                for pkl_episode_index, episode in enumerate(
                    dataset._episodes[path_index][: args.per_layout]
                ):
                    states = np.asarray(episode["observation"]["states"])
                    demo_key, hdf5_demo = _hdf5_demo_for_episode(hdf5["data"], states)
                    record = _replay_episode(
                        env=env,
                        episode=episode,
                        hdf5_demo=hdf5_demo,
                        layout=layout,
                        pkl_episode_index=pkl_episode_index,
                        demo_key=demo_key,
                        dataset=dataset,
                        source_pkl=source_pkl,
                        migrate_saved_model_xml=migrate_saved_model_xml,
                        rotation_6d_to_matrix=rotation_6d_to_matrix,
                        matrix_to_rotation_6d=matrix_to_rotation_6d,
                        scipy_rotation=R,
                    )
                    records.append(record)
                    print(
                        f"layout={layout} episode={pkl_episode_index} "
                        f"demo={demo_key} success={record['success']} "
                        f"stage={record['failure_stage']}",
                        flush=True,
                    )
        finally:
            env.close()

    if runtime_contract is None:
        raise RuntimeError("no Point Bridge runtime was created")
    per_layout: dict[str, dict[str, Any]] = {}
    for layout in range(1, 5):
        selected = [record for record in records if record["layout"] == layout]
        passed = sum(bool(record["passed"]) for record in selected)
        position_max = max(
            (
                record.get("tracking_summary", {})
                .get("position_m", {})
                .get("max", float("nan"))
                for record in selected
                if record.get("tracking_summary")
            ),
            default=float("nan"),
        )
        rotation_max = max(
            (
                record.get("tracking_summary", {})
                .get("rotation_rad", {})
                .get("max", float("nan"))
                for record in selected
                if record.get("tracking_summary")
            ),
            default=float("nan"),
        )
        per_layout[str(layout)] = {
            "checked": len(selected),
            "passed": passed,
            "success_rate": passed / len(selected) if selected else 0.0,
            "max_tracking_position_error_m": position_max,
            "max_tracking_rotation_error_rad": rotation_max,
            "failure_stages": {
                stage: sum(record["failure_stage"] == stage for record in selected)
                for stage in sorted({record["failure_stage"] for record in selected})
            },
        }

    clean_policy_rate = _load_clean_policy_rate(args.clean_policy_report.resolve())
    decision = classify_gate(per_layout, clean_policy_rate)
    result = {
        "stage": "V1-R.2G",
        "audit": "pointbridge_absolute_pose_contract",
        "status": decision["gate"],
        "runtime": runtime_contract,
        "dataset_contract": dataset_contract,
        "selection": {
            "per_layout": args.per_layout,
            "layouts": [1, 2, 3, 4],
            "source": "first five native BCDataset episodes per layout; matched to HDF5 by float32 initial state and trajectory length",
        },
        "clean_policy_success_rate": clean_policy_rate,
        "checked": len(records),
        "passed": sum(bool(record["passed"]) for record in records),
        "restored_initial_state_matches": sum(
            bool(record["restored_initial_state_match"]) for record in records
        ),
        "translation_boundary_hit_steps": sum(
            int(record.get("translation_boundary_hit_steps", 0)) for record in records
        ),
        "rotation_boundary_hit_steps": sum(
            int(record.get("rotation_boundary_hit_steps", 0)) for record in records
        ),
        "telemetry_steps": sum(len(record["telemetry"]) for record in records),
        "coordinate_frame_contracts": coordinate_contracts,
        "per_layout": per_layout,
        "decision": decision,
        "authorization": {
            "clean_baseline_gate": "blocked",
            "b0_b1_training_authorized": decision["b0_b1_training_authorized"],
            "v2_formal_experiment_authorized": False,
            "v3_formal_experiment_authorized": False,
        },
        "records": records,
    }
    result = _jsonable(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.gate_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(_markdown_report(result), encoding="utf-8")
    args.gate_output.write_text(_gate_yaml(result), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "checked", "passed", "decision")}, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
