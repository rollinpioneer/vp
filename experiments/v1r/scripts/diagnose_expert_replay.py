#!/usr/bin/env python3
"""Diagnose expert replay on two failed and one successful trajectory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPSTREAM = ROOT / "third_party" / "pointbridge"
DEFAULT_CASES = ((1, "demo_0"), (2, "demo_0"), (3, "demo_3"))
sys.path.insert(0, str(ROOT / "src"))

from vico_point.envs.mimiclabs_compat import migrate_saved_model_xml


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


def as_list(value: Any) -> list[float]:
    return np.asarray(value, dtype=np.float64).tolist()


def orientation_error_rad(left: Any, right: Any) -> float:
    relative = np.asarray(left) @ np.asarray(right).T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def l2_error(left: Any, right: Any) -> float:
    return float(np.linalg.norm(np.asarray(left) - np.asarray(right)))


def controller_for(env: Any) -> Any:
    controller = env.robots[0].controller
    if isinstance(controller, dict):
        if len(controller) != 1:
            raise ValueError("diagnostic expects exactly one arm controller")
        controller = next(iter(controller.values()))
    return controller


def direct_telemetry(env: Any) -> dict[str, Any]:
    import robosuite.utils.transform_utils as transform_utils

    controller = controller_for(env)
    site_id = env.sim.model.site_name2id(controller.eef_name)
    eef_orientation = np.asarray(env.sim.data.site_xmat[site_id]).reshape(3, 3)
    bowl_body = env.objects_dict["bowl"].root_body
    bowl_orientation = np.asarray(env.sim.data.get_body_xmat(bowl_body)).reshape(3, 3)
    robot = env.robots[0]
    gripper_qpos = np.asarray(env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes])
    return {
        "eef_position_m": as_list(env.sim.data.site_xpos[site_id]),
        "eef_orientation_xyzw": as_list(transform_utils.mat2quat(eef_orientation)),
        "eef_orientation_matrix": np.asarray(eef_orientation).tolist(),
        "eef_linear_velocity_m_per_s": as_list(
            env.sim.data.get_site_xvelp(controller.eef_name)
        ),
        "eef_angular_velocity_rad_per_s": as_list(
            env.sim.data.get_site_xvelr(controller.eef_name)
        ),
        "gripper_qpos_m": as_list(gripper_qpos),
        "bowl_position_m": as_list(env.sim.data.get_body_xpos(bowl_body)),
        "bowl_orientation_xyzw": as_list(transform_utils.mat2quat(bowl_orientation)),
        "bowl_orientation_matrix": np.asarray(bowl_orientation).tolist(),
        "flattened_state": as_list(env.sim.get_state().flatten()),
    }


def controller_cache(env: Any) -> dict[str, Any]:
    import robosuite.utils.transform_utils as transform_utils

    controller = controller_for(env)
    gripper = env.robots[0].gripper
    return {
        "new_update": bool(controller.new_update),
        "cached_eef_position_m": as_list(controller.ee_pos),
        "cached_eef_orientation_xyzw": as_list(
            transform_utils.mat2quat(controller.ee_ori_mat)
        ),
        "goal_position_m": as_list(controller.goal_pos),
        "goal_orientation_xyzw": as_list(
            transform_utils.mat2quat(controller.goal_ori)
        ),
        "initial_joint_position_rad": as_list(controller.initial_joint),
        "gripper_current_action": as_list(gripper.current_action),
    }


def synchronize_controller(env: Any) -> None:
    controller = controller_for(env)
    controller.update(force=True)
    controller.update_initial_joints(controller.joint_pos)


def reset_gripper_cache(env: Any, initial_action: Any = None) -> None:
    """Reset robosuite's stateful gripper command to the trajectory start."""
    gripper = env.robots[0].gripper
    action = np.zeros(gripper.dof) if initial_action is None else np.asarray(initial_action)
    if action.shape != (gripper.dof,):
        raise ValueError(
            f"expected gripper cache shape {(gripper.dof,)}, got {action.shape}"
        )
    gripper.current_action = action.astype(np.float64, copy=True)


def synchronize_runtime_state(env: Any, initial_gripper_action: Any = None) -> None:
    """Align non-MuJoCo caches with a state restored from a trajectory."""
    reset_gripper_cache(env, initial_gripper_action)
    synchronize_controller(env)


def state_errors(actual: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    return {
        "eef_position_l2_m": l2_error(
            actual["eef_position_m"], reference["eef_position_m"]
        ),
        "eef_orientation_angle_rad": orientation_error_rad(
            actual["eef_orientation_matrix"], reference["eef_orientation_matrix"]
        ),
        "eef_linear_velocity_l2_m_per_s": l2_error(
            actual["eef_linear_velocity_m_per_s"],
            reference["eef_linear_velocity_m_per_s"],
        ),
        "eef_angular_velocity_l2_rad_per_s": l2_error(
            actual["eef_angular_velocity_rad_per_s"],
            reference["eef_angular_velocity_rad_per_s"],
        ),
        "gripper_qpos_l2_m": l2_error(
            actual["gripper_qpos_m"], reference["gripper_qpos_m"]
        ),
        "bowl_position_l2_m": l2_error(
            actual["bowl_position_m"], reference["bowl_position_m"]
        ),
        "bowl_orientation_angle_rad": orientation_error_rad(
            actual["bowl_orientation_matrix"], reference["bowl_orientation_matrix"]
        ),
        "flattened_state_max_abs_native": float(
            np.max(
                np.abs(
                    np.asarray(actual["flattened_state"])
                    - np.asarray(reference["flattened_state"])
                )
            )
        ),
    }


def target_tracking_errors(actual: dict[str, Any], target: dict[str, Any]) -> dict[str, float]:
    return {
        "position_l2_m": l2_error(
            actual["eef_position_m"], target["position_m"]
        ),
        "orientation_angle_rad": orientation_error_rad(
            actual["eef_orientation_matrix"], target["orientation_matrix"]
        ),
        "linear_velocity_to_zero_l2_m_per_s": float(
            np.linalg.norm(actual["eef_linear_velocity_m_per_s"])
        ),
        "angular_velocity_to_zero_l2_rad_per_s": float(
            np.linalg.norm(actual["eef_angular_velocity_rad_per_s"])
        ),
    }


def first_translation_saturation(transitions: list[dict[str, Any]]) -> dict[str, Any]:
    """Find the first saved translation command that reaches the input limit."""
    for transition in transitions:
        translation_action = np.asarray(transition["action"][:3], dtype=np.float64)
        if np.max(np.abs(translation_action)) >= 1.0 - 1e-12:
            return {
                "action_index": transition["action_index"],
                "translation_action": as_list(translation_action),
                "translation_action_abs_max": float(np.max(np.abs(translation_action))),
            }
    return {
        "action_index": None,
        "translation_action": None,
        "translation_action_abs_max": None,
    }


def attach_first_explanatory_difference(
    replay: dict[str, Any], diagnostic_steps: int
) -> None:
    """Record the first execution mismatch and test delta-target error carry-over."""
    transitions = replay["transitions"][:diagnostic_steps]
    if not transitions:
        replay["first_explanatory_difference"] = {
            "candidate_supported": False,
            "reason": "no diagnostic transitions",
        }
        return

    first = transitions[0]
    first_target_errors = first["actual_vs_saved_controller_target_errors"]
    first_state_errors = first["actual_vs_reference_errors"]
    first_tracking = first["controller_target_tracking_errors"]

    propagation_residuals = []
    for previous, current in zip(transitions, transitions[1:]):
        target_error = current["actual_vs_saved_controller_target_errors"][
            "position_l2_m"
        ]
        previous_state_error = previous["actual_vs_reference_errors"][
            "eef_position_l2_m"
        ]
        propagation_residuals.append(abs(target_error - previous_state_error))

    saturation = first_translation_saturation(transitions)
    max_propagation_residual_m = max(propagation_residuals, default=0.0)
    target_matches_saved = (
        first_target_errors["position_l2_m"] <= 1e-6
        and first_target_errors["orientation_angle_rad"] <= 1e-6
    )
    propagation_supported = (
        replay["controller_use_delta"]
        and len(propagation_residuals) > 0
        and max_propagation_residual_m <= 1e-6
    )
    replay["first_explanatory_difference"] = {
        "candidate_supported": bool(
            replay["controller_use_delta"]
            and target_matches_saved
            and first_state_errors["eef_position_l2_m"] > 1e-6
            and propagation_supported
        ),
        "action_index": first["action_index"],
        "state_transition": "state[0] + action[0] -> state[1]",
        "mechanism": (
            "action_0_target_matches_saved_target_but_post_step_EEF_state_diverges; "
            "delta_OSC_uses_the_diverged_current_EEF_as_the_next_target_origin"
        ),
        "action_0_target_position_error_m": first_target_errors["position_l2_m"],
        "action_0_target_orientation_error_rad": first_target_errors[
            "orientation_angle_rad"
        ],
        "action_0_post_step_eef_position_error_m": first_state_errors[
            "eef_position_l2_m"
        ],
        "action_0_post_step_eef_position_error_mm": first_state_errors[
            "eef_position_l2_m"
        ]
        * 1000.0,
        "action_0_post_step_eef_orientation_error_rad": first_state_errors[
            "eef_orientation_angle_rad"
        ],
        "action_0_post_step_eef_linear_velocity_error_m_per_s": first_state_errors[
            "eef_linear_velocity_l2_m_per_s"
        ],
        "action_0_post_step_eef_angular_velocity_error_rad_per_s": first_state_errors[
            "eef_angular_velocity_l2_rad_per_s"
        ],
        "action_0_post_step_bowl_position_error_m": first_state_errors[
            "bowl_position_l2_m"
        ],
        "action_0_post_step_gripper_qpos_error_m": first_state_errors[
            "gripper_qpos_l2_m"
        ],
        "action_0_target_tracking_position_m": first_tracking["position_l2_m"],
        "action_0_target_tracking_position_mm": first_tracking["position_l2_m"]
        * 1000.0,
        "early_translation_action_abs_max": [
            float(np.max(np.abs(np.asarray(item["action"][:3], dtype=np.float64))))
            for item in transitions[:8]
        ],
        "early_target_tracking_position_mm": [
            item["controller_target_tracking_errors"]["position_l2_m"] * 1000.0
            for item in transitions[:8]
        ],
        "first_translation_saturation": saturation,
        "delta_target_error_propagation": {
            "checked_transitions": len(propagation_residuals),
            "max_abs_residual_m": max_propagation_residual_m,
            "max_abs_residual_mm": max_propagation_residual_m * 1000.0,
            "supported": bool(propagation_supported),
        },
        "initial_joint_reference": (
            "set once to restored state[0] arm joints before action[0] and held fixed "
            "during continuous replay"
        ),
    }


def saved_target_alignment(demo: Any, controller_config: dict[str, Any], steps: int) -> dict[str, Any]:
    actions = np.asarray(demo["actions"][:steps], dtype=np.float64)
    eef_pose = np.asarray(demo["datagen_info/eef_pose"][:steps], dtype=np.float64)
    target_pose = np.asarray(demo["datagen_info/target_pose"][:steps], dtype=np.float64)
    input_min = np.broadcast_to(controller_config["input_min"], 6)
    input_max = np.broadcast_to(controller_config["input_max"], 6)
    output_min = np.asarray(controller_config["output_min"], dtype=np.float64)
    output_max = np.asarray(controller_config["output_max"], dtype=np.float64)
    clipped = np.clip(actions[:, :6], input_min, input_max)
    scale = np.abs(output_max - output_min) / np.abs(input_max - input_min)
    input_center = (input_max + input_min) / 2.0
    output_center = (output_max + output_min) / 2.0
    scaled = (clipped - input_center) * scale + output_center
    expected_positions = eef_pose[:, :3, 3] + scaled[:, :3]
    position_errors = np.linalg.norm(expected_positions - target_pose[:, :3, 3], axis=1)
    return {
        "control_delta": bool(controller_config["control_delta"]),
        "actions_abs_present": "actions_abs" in demo,
        "state_action_indexing": "state[t] + action[t] -> state[t+1]",
        "position_target_reconstruction_max_error_m": float(position_errors.max()),
        "position_target_reconstruction_per_step_m": position_errors.tolist(),
    }


def absolute_target_actions(demo: Any) -> tuple[np.ndarray, np.ndarray]:
    import robosuite.utils.transform_utils as transform_utils

    target_poses = np.asarray(demo["datagen_info/target_pose"], dtype=np.float64)
    source_actions = np.asarray(demo["actions"], dtype=np.float64)
    actions = []
    for target_pose, source_action in zip(target_poses, source_actions):
        orientation = transform_utils.quat2axisangle(
            transform_utils.mat2quat(target_pose[:3, :3]).copy()
        )
        actions.append(
            np.concatenate((target_pose[:3, 3], orientation, source_action[-1:]))
        )
    return np.asarray(actions), target_poses


def restored_references(env: Any, states: np.ndarray, steps: int) -> list[dict[str, Any]]:
    references = []
    for state_index in range(steps + 1):
        env.reset_to({"states": states[state_index]})
        references.append(
            {
                "state_index": state_index,
                "state": direct_telemetry(env),
                "controller_cache_after_restore": controller_cache(env),
            }
        )
    return references


def replay_mode(
    env: Any,
    states: np.ndarray,
    actions: np.ndarray,
    model_xml: str | None,
    saved_target_poses: np.ndarray,
    steps: int,
    synchronize: bool,
    use_delta: bool,
) -> dict[str, Any]:
    import robosuite.utils.transform_utils as transform_utils

    reset_payload = {"states": states[0]}
    if model_xml is not None:
        reset_payload["model"] = model_xml
    env.reset_to(reset_payload)
    restored = direct_telemetry(env)
    reset_gripper_cache(env)
    cache_before_sync = controller_cache(env)
    controller = controller_for(env)
    cache_position_error = l2_error(
        cache_before_sync["cached_eef_position_m"], restored["eef_position_m"]
    )
    cache_orientation_error = orientation_error_rad(
        controller.ee_ori_mat, restored["eef_orientation_matrix"]
    )
    if synchronize:
        synchronize_runtime_state(env)
    controller_for(env).use_delta = use_delta
    cache_before_first_action = controller_cache(env)

    transitions = []
    reward = float(env.reward())
    for action_index, action in enumerate(actions):
        time_before = float(getattr(env, "cur_time", np.nan))
        _, reward, _, _ = env.step(action)
        time_after = float(getattr(env, "cur_time", np.nan))
        if action_index < steps:
            actual = direct_telemetry(env)
            controller = controller_for(env)
            target = {
                "position_m": as_list(controller.goal_pos),
                "orientation_xyzw": as_list(
                    transform_utils.mat2quat(controller.goal_ori)
                ),
                "orientation_matrix": np.asarray(controller.goal_ori).tolist(),
                "linear_velocity_m_per_s": [0.0, 0.0, 0.0],
                "angular_velocity_rad_per_s": [0.0, 0.0, 0.0],
            }
            saved_target_pose = saved_target_poses[action_index]
            transitions.append(
                {
                    "action_index": action_index,
                    "source_state_index": action_index,
                    "reference_next_state_index": action_index + 1,
                    "actual_control_interval_s": time_after - time_before,
                    "action": as_list(action),
                    "gripper_command": float(action[-1]),
                    "actual_controller_target": target,
                    "saved_controller_target": {
                        "position_m": as_list(saved_target_pose[:3, 3]),
                        "orientation_xyzw": as_list(
                            transform_utils.mat2quat(saved_target_pose[:3, :3])
                        ),
                        "orientation_matrix": np.asarray(
                            saved_target_pose[:3, :3]
                        ).tolist(),
                    },
                    "actual_vs_saved_controller_target_errors": {
                        "position_l2_m": l2_error(
                            target["position_m"], saved_target_pose[:3, 3]
                        ),
                        "orientation_angle_rad": orientation_error_rad(
                            target["orientation_matrix"], saved_target_pose[:3, :3]
                        ),
                    },
                    "actual_after_step": actual,
                    "controller_target_tracking_errors": target_tracking_errors(
                        actual, target
                    ),
                }
            )

    return {
        "controller_sync_applied": synchronize,
        "controller_use_delta": use_delta,
        "initial_controller_cache_vs_restored_state": {
            "position_l2_m": cache_position_error,
            "orientation_angle_rad": cache_orientation_error,
            "cache": cache_before_sync,
            "restored_state": restored,
        },
        "controller_cache_before_first_action": cache_before_first_action,
        "full_replay_reward": float(reward),
        "full_replay_success": bool(env._check_success()),
        "transitions": transitions,
    }


def attach_restored_references(
    replay: dict[str, Any], references: list[dict[str, Any]]
) -> None:
    for transition in replay["transitions"]:
        reference = references[transition["reference_next_state_index"]]["state"]
        actual = transition["actual_after_step"]
        transition["reference_next_restored_state"] = reference
        transition["actual_vs_reference_errors"] = state_errors(actual, reference)
    replay["first_eef_position_divergence_action_index"] = next(
        (
            item["action_index"]
            for item in replay["transitions"]
            if item["actual_vs_reference_errors"]["eef_position_l2_m"] > 1e-6
        ),
        None,
    )
    replay["first_eef_orientation_divergence_action_index"] = next(
        (
            item["action_index"]
            for item in replay["transitions"]
            if item["actual_vs_reference_errors"]["eef_orientation_angle_rad"]
            > 1e-6
        ),
        None,
    )


def diagnose_case(suite: Any, upstream: Path, layout: int, demo_key: str, steps: int) -> dict[str, Any]:
    task_name = f"bowl_on_plate_{layout}"
    dataset_path = (
        upstream
        / "data"
        / "mimicgen_data"
        / "bowl_on_plate"
        / task_name
        / "demo"
        / "demo.hdf5"
    )
    bddl_path = (
        upstream
        / "third_party"
        / "mimiclabs"
        / "mimiclabs"
        / "mimiclabs"
        / "task_suites"
        / "new_task_suite"
        / f"{task_name}.bddl"
    )
    with h5py.File(dataset_path, "r") as handle:
        env_spec = json.loads(str(handle["data"].attrs["env_args"]))
        env_kwargs = dict(env_spec["env_kwargs"])
        env_kwargs.pop("env_lang", None)
        env_kwargs.update(
            {
                "has_renderer": False,
                "has_offscreen_renderer": False,
                "use_camera_obs": False,
                "bddl_file_name": str(bddl_path),
            }
        )
        env = suite.make(env_name=env_spec["env_name"], **env_kwargs)
        env.reset()
        try:
            demo = handle["data"][demo_key]
            states = np.asarray(demo["states"])
            actions = np.asarray(demo["actions"])
            if steps >= len(states):
                raise ValueError(
                    f"requested {steps} transitions, but {layout}/{demo_key} has "
                    f"only {len(states)} saved states"
                )
            original_xml = demo.attrs["model_file"]
            if isinstance(original_xml, bytes):
                original_xml = original_xml.decode("utf-8")
            migrated_xml = migrate_saved_model_xml(str(original_xml))
            controller_configs = env_spec["env_kwargs"]["controller_configs"]
            if isinstance(controller_configs, dict):
                controller_configs = [controller_configs]
            absolute_actions, saved_target_poses = absolute_target_actions(demo)
            before = replay_mode(
                env,
                states,
                actions,
                migrated_xml,
                saved_target_poses,
                steps,
                synchronize=False,
                use_delta=True,
            )
            after = replay_mode(
                env,
                states,
                actions,
                migrated_xml,
                saved_target_poses,
                steps,
                synchronize=True,
                use_delta=True,
            )
            absolute = replay_mode(
                env,
                states,
                absolute_actions,
                migrated_xml,
                saved_target_poses,
                steps,
                synchronize=True,
                use_delta=False,
            )
            references = restored_references(env, states, steps)
            attach_restored_references(before, references)
            attach_restored_references(after, references)
            attach_restored_references(absolute, references)
            attach_first_explanatory_difference(before, steps)
            attach_first_explanatory_difference(after, steps)
            attach_first_explanatory_difference(absolute, steps)
            control_freq = float(getattr(env, "control_freq", 20.0))
            expected_control_interval_s = 1.0 / control_freq
            actual_intervals = [
                item["actual_control_interval_s"]
                for item in after["transitions"]
            ]
            return {
                "layout": layout,
                "demo_key": demo_key,
                "trajectory_steps": len(actions),
                "diagnostic_transitions": steps,
                "dataset": str(dataset_path.relative_to(ROOT)),
                "controller_config": controller_configs[0],
                "time_alignment": {
                    "control_freq_hz": control_freq,
                    "saved_action_interval_s": expected_control_interval_s,
                    "actual_control_intervals_s": actual_intervals,
                    "actual_interval_max_abs_error_s": max(
                        (
                            abs(interval - expected_control_interval_s)
                            for interval in actual_intervals
                        ),
                        default=float("inf"),
                    ),
                    "supported": bool(
                        actual_intervals
                        and max(
                            abs(interval - expected_control_interval_s)
                            for interval in actual_intervals
                        )
                        <= 1e-12
                    ),
                    "state_action_pairing": "state[t] + action[t] -> state[t+1]",
                },
                "action_semantics_and_alignment": saved_target_alignment(
                    demo, controller_configs[0], steps
                ),
                "per_frame_restoration": references,
                "continuous_before_repair": before,
                "continuous_with_controller_sync": after,
                "continuous_with_saved_absolute_targets": absolute,
            }
        finally:
            env.close()


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = []
    for case in cases:
        before = case["continuous_before_repair"]
        after = case["continuous_with_controller_sync"]
        absolute = case["continuous_with_saved_absolute_targets"]
        evidence = after["first_explanatory_difference"]
        propagation = evidence["delta_target_error_propagation"]
        time_alignment = case["time_alignment"]
        summaries.append(
            {
                "layout": case["layout"],
                "demo_key": case["demo_key"],
                "initial_cached_position_error_m": before[
                    "initial_controller_cache_vs_restored_state"
                ]["position_l2_m"],
                "initial_cached_orientation_error_rad": before[
                    "initial_controller_cache_vs_restored_state"
                ]["orientation_angle_rad"],
                "before_first_position_divergence_action_index": before[
                    "first_eef_position_divergence_action_index"
                ],
                "after_first_position_divergence_action_index": after[
                    "first_eef_position_divergence_action_index"
                ],
                "before_success": before["full_replay_success"],
                "after_success": after["full_replay_success"],
                "absolute_target_success": absolute["full_replay_success"],
                "absolute_target_first_position_divergence_action_index": absolute[
                    "first_eef_position_divergence_action_index"
                ],
                "first_explanatory_difference": evidence,
                "first_action_target_position_error_m": evidence[
                    "action_0_target_position_error_m"
                ],
                "first_action_eef_position_error_mm": evidence[
                    "action_0_post_step_eef_position_error_mm"
                ],
                "first_action_target_tracking_position_mm": evidence[
                    "action_0_target_tracking_position_mm"
                ],
                "first_translation_saturation_action_index": evidence[
                    "first_translation_saturation"
                ]["action_index"],
                "max_first_diagnostic_eef_position_error_mm": max(
                    item["actual_vs_reference_errors"]["eef_position_l2_m"]
                    for item in after["transitions"]
                )
                * 1000.0,
                "max_first_diagnostic_target_tracking_position_mm": max(
                    item["controller_target_tracking_errors"]["position_l2_m"]
                    for item in after["transitions"]
                )
                * 1000.0,
                "delta_target_error_propagation_supported": propagation[
                    "supported"
                ],
                "delta_target_error_propagation_max_residual_mm": propagation[
                    "max_abs_residual_mm"
                ],
                "time_alignment_supported": time_alignment["supported"],
            }
        )
    return {
        "diagnostic_scope": [
            {"layout": layout, "demo_key": demo_key}
            for layout, demo_key in DEFAULT_CASES
        ],
        "action_semantics_supported": all(
            case["action_semantics_and_alignment"]["control_delta"]
            and not case["action_semantics_and_alignment"]["actions_abs_present"]
            and case["action_semantics_and_alignment"][
                "position_target_reconstruction_max_error_m"
            ]
            < 1e-12
            for case in cases
        ),
        "time_alignment_supported": all(
            case["time_alignment"]["supported"] for case in cases
        ),
        "first_explanatory_difference_supported": all(
            item["first_explanatory_difference"]["candidate_supported"]
            for item in summaries
        ),
        "controller_cache_mismatch_present": any(
            item["initial_cached_position_error_m"] > 1e-6
            or item["initial_cached_orientation_error_rad"] > 1e-6
            for item in summaries
        ),
        "controller_sync_repairs_all_three": all(item["after_success"] for item in summaries),
        "saved_absolute_targets_repair_all_three": all(
            item["absolute_target_success"] for item in summaries
        ),
        "cases": summaries,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    runtime = result.get("runtime", {})
    runtime_line = ""
    if runtime:
        runtime_line = (
            f"Runtime: robosuite `{runtime.get('robosuite_version', 'unknown')}`, "
            f"MuJoCo `{runtime.get('mujoco_version', 'unknown')}`.\n\n"
        )
    rows = [
        "| layout | demo | stale pos (m) | stale ori (rad) | delta baseline | delta + sync | saved absolute target |",
        "|---:|---|---:|---:|---|---|---|",
    ]
    for item in summary["cases"]:
        rows.append(
            "| {layout} | {demo_key} | {initial_cached_position_error_m:.9g} | "
            "{initial_cached_orientation_error_rad:.9g} | {before_success} | "
            "{after_success} | {absolute_target_success} |".format(**item)
        )
    evidence_rows = [
        "| layout | action 0 target err (mm) | action 0 EEF err (mm) | "
        "action 0 target tracking (mm) | first translation saturation | "
        "max EEF err in first 20 (mm) | carry-over residual (mm) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["cases"]:
        saturation = item["first_translation_saturation_action_index"]
        saturation_text = "none" if saturation is None else str(saturation)
        evidence_rows.append(
            f"| {item['layout']} | {item['first_action_target_position_error_m'] * 1000:.6g} | "
            f"{item['first_action_eef_position_error_mm']:.6g} | "
            f"{item['first_action_target_tracking_position_mm']:.6g} | "
            f"{saturation_text} | {item['max_first_diagnostic_eef_position_error_mm']:.6g} | "
            f"{item['delta_target_error_propagation_max_residual_mm']:.6g} |"
        )
    time_alignment_rows = [
        "| layout | expected action interval (s) | actual interval max error (s) | supported |",
        "|---:|---:|---:|---|",
    ]
    for case, item in zip(result["cases"], summary["cases"]):
        alignment = case["time_alignment"]
        time_alignment_rows.append(
            f"| {item['layout']} | {alignment['saved_action_interval_s']:.12g} | "
            f"{alignment['actual_interval_max_abs_error_s']:.12g} | "
            f"{alignment['supported']} |"
        )
    carry_over_note = (
        "The strict delta-target carry-over check is supported for this runtime."
        if summary["first_explanatory_difference_supported"]
        else "The first-step execution mismatch is present, but the strict delta-target carry-over check is not asserted for this runtime."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# V1-R Expert Replay Targeted Diagnosis\n\n"
        + runtime_line
        + "Scope: layout 1 `demo_0`, layout 2 `demo_0`, and layout 3 `demo_3`; "
        f"first {result['diagnostic_transitions']} transitions plus full success replay.\n\n"
        + "\n".join(rows)
        + "\n\n## First Explanatory Difference\n\n"
        + "The first mismatch is action execution after exact state restoration, not action decoding: "
        "after controller synchronization, action 0 produces the saved controller target, but the "
        "post-step EEF state diverges. In delta OSC, the next target is based on that actual EEF, "
        "so the target error from action 1 onward carries the previous EEF error.\n\n"
        + "\n".join(evidence_rows)
        + "\n\n"
        + "`initial_joint` is set once from restored `state[0]` and held fixed during continuous replay; "
        "it is not updated per frame.\n\n"
        + "## Time Alignment\n\n"
        + "\n".join(time_alignment_rows)
        + "\n\n"
        + "- Action semantics: saved configuration is delta OSC; `actions_abs` is absent.\n"
        + "- Time alignment: collection stores `state[t]` before executing `action[t]`; "
        "saved targets reconstruct from `eef_pose[t] + scaled action[t]`.\n"
        + "- Controller restoration: `reset_to()` restores MuJoCo state but leaves the "
        "controller cache at the deterministic reset pose until explicitly synchronized; "
        "the sync-only result is reported separately.\n"
        + "- Absolute-target check: `datagen_info/target_pose` is converted to the "
        "world-frame absolute OSC representation used by robosuite, without changing "
        "position/orientation scaling or the saved gripper command.\n"
        + "- Runtime qualification: "
        + carry_over_note
        + "\n"
        + "- Units: position m, orientation angle rad, linear velocity m/s, angular "
        "velocity rad/s, Panda finger joint position m.\n\n"
        + "The JSON companion contains per-frame restored references, continuous replay "
        "telemetry, actual controller targets, and separated error channels.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--steps", type=int, default=20, choices=range(10, 21))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    upstream = args.upstream.resolve()
    add_upstream_paths(upstream)

    import robosuite as suite
    from mimiclabs.mimiclabs.envs.problems import (  # noqa: F401
        MimicLabs_Lab1_Tabletop_Manipulation,
    )

    cases = []
    for layout, demo_key in DEFAULT_CASES:
        case = diagnose_case(suite, upstream, layout, demo_key, args.steps)
        cases.append(case)
        print(
            f"layout={layout} demo={demo_key} "
            f"before={case['continuous_before_repair']['full_replay_success']} "
            f"after_sync={case['continuous_with_controller_sync']['full_replay_success']} "
            "absolute_target="
            f"{case['continuous_with_saved_absolute_targets']['full_replay_success']}",
            flush=True,
        )
    result = {
        "stage": "V1-R.2F-targeted-diagnosis",
        "diagnostic_transitions": args.steps,
        "runtime": {
            "robosuite_version": getattr(suite, "__version__", "unknown"),
            "mujoco_version": __import__("mujoco").__version__,
        },
        "summary": summarize(cases),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.report, result)
    print(json.dumps(result["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
