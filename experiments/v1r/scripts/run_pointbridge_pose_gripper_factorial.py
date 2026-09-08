#!/usr/bin/env python3
"""Run the V1-R.2H Point Bridge pose-target / gripper-alignment factorial.

The experiment replays the same twenty V1-R.2G demonstrations from the same
initial states under four action contracts. It changes only the pose target
source (next measured EEF pose versus saved controller target) and gripper
index (next state versus current state-transition command).
"""

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
sys.path.insert(0, str(ROOT))

from experiments.v1r.scripts.validate_pointbridge_absolute_pose import (  # noqa: E402
    ACT_SUBSAMPLE,
    CONTROL_FREQ_HZ,
    DEFAULT_UPSTREAM,
    PER_LAYOUT,
    REQUIRED_MUJOCO,
    REQUIRED_ROBOSUITE,
    _capture_dataset_metadata,
    _controller_for,
    _dataset_contract,
    _hdf5_demo_for_episode,
    _jsonable,
    _make_official_environment,
    _phase_observation,
    _pointbridge_core_env,
    _runtime_contract,
    _synchronize_restored_runtime,
    add_upstream_paths,
    coordinate_frame_contract,
    failure_stage,
    file_sha256,
    normalize_and_postprocess,
    raw_array_sha256,
    text_sha256,
)


GROUP_SPECS = {
    "A": {
        "pose": "P0_next_measured_eef_pose",
        "gripper": "G0_next_raw_gripper_state",
        "purpose": "current_pointbridge_baseline",
    },
    "B": {
        "pose": "P0_next_measured_eef_pose",
        "gripper": "G1_current_transition_gripper_command",
        "purpose": "gripper_alignment_only",
    },
    "C": {
        "pose": "P1_saved_controller_absolute_target",
        "gripper": "G0_next_raw_gripper_state",
        "purpose": "pose_target_source_only",
    },
    "D": {
        "pose": "P1_saved_controller_absolute_target",
        "gripper": "G1_current_transition_gripper_command",
        "purpose": "saved_controller_target_and_transition_gripper",
    },
}
STABLE_GRASP_STEPS = 5
CLOSED_APERTURE_THRESHOLD_M = 0.02
SWITCH_WINDOW_RADIUS = 5


def first_closing_switch(commands: Any) -> int | None:
    """Return the first index that changes from open/nonpositive to closing."""

    values = np.asarray(commands, dtype=np.float64).reshape(-1)
    for index in range(1, len(values)):
        if values[index - 1] <= 0.0 < values[index]:
            return index
    return None


def first_stable_true_run(values: Any, minimum_steps: int = STABLE_GRASP_STEPS) -> int | None:
    """Return the start index of the first sufficiently long True run."""

    if minimum_steps <= 0:
        raise ValueError("minimum_steps must be positive")
    run_start: int | None = None
    for index, value in enumerate(np.asarray(values, dtype=bool).reshape(-1)):
        if value:
            if run_start is None:
                run_start = index
            if index - run_start + 1 >= minimum_steps:
                return run_start
        else:
            run_start = None
    return None


def longest_true_run(values: Any) -> int:
    """Return the longest consecutive True run length."""

    longest = 0
    current = 0
    for value in np.asarray(values, dtype=bool).reshape(-1):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def saved_targets_to_pointbridge_pose(
    saved_target_poses: Any,
    robot_base: Any,
    gripper_transform: Any,
    matrix_to_rotation_6d: Any,
) -> np.ndarray:
    """Convert saved world-frame controller targets to Point Bridge pose labels."""

    targets = np.asarray(saved_target_poses, dtype=np.float64)
    base = np.asarray(robot_base, dtype=np.float64)
    gripper = np.asarray(gripper_transform, dtype=np.float64)
    if targets.ndim != 3 or targets.shape[1:] != (4, 4):
        raise ValueError(f"saved targets must have shape (T, 4, 4), got {targets.shape}")
    if base.shape != (4, 4) or gripper.shape != (4, 4):
        raise ValueError("robot_base and gripper_transform must both be 4x4")
    inverse_base = np.linalg.inv(base)
    actions = []
    for target in targets:
        pointbridge_target = inverse_base @ target @ gripper
        actions.append(
            np.concatenate(
                [
                    pointbridge_target[:3, 3],
                    np.asarray(matrix_to_rotation_6d(pointbridge_target[:3, :3])),
                ]
            )
        )
    return np.asarray(actions, dtype=np.float64)


def pointbridge_pose_to_world_targets(
    pointbridge_pose: Any,
    robot_base: Any,
    gripper_transform: Any,
    rotation_6d_to_matrix: Any,
) -> np.ndarray:
    """Map Point Bridge pose labels through the wrapper's world-target transform."""

    poses = np.asarray(pointbridge_pose, dtype=np.float64)
    base = np.asarray(robot_base, dtype=np.float64)
    gripper_inverse = np.linalg.inv(np.asarray(gripper_transform, dtype=np.float64))
    targets = []
    for pose in poses:
        pointbridge_target = np.eye(4)
        pointbridge_target[:3, 3] = pose[:3]
        pointbridge_target[:3, :3] = rotation_6d_to_matrix(pose[3:9])
        targets.append(base @ pointbridge_target @ gripper_inverse)
    return np.asarray(targets)


def build_raw_factorial_actions(
    pointbridge_actions: Any,
    raw_gripper_states: Any,
    saved_target_poses: Any,
    robot_base: Any,
    gripper_transform: Any,
    matrix_to_rotation_6d: Any,
) -> dict[str, np.ndarray]:
    """Construct A/B/C/D while changing only pose source and gripper index."""

    baseline = np.asarray(pointbridge_actions, dtype=np.float64)
    raw_gripper = np.asarray(raw_gripper_states, dtype=np.float64).reshape(-1)
    if baseline.ndim != 2 or baseline.shape[1] != 10:
        raise ValueError(f"Point Bridge actions must have shape (T, 10), got {baseline.shape}")
    if len(raw_gripper) != len(baseline):
        raise ValueError("raw gripper state length must match Point Bridge actions")
    p0 = baseline[:, :9]
    g0 = baseline[:, -1]
    p1 = saved_targets_to_pointbridge_pose(
        saved_target_poses,
        robot_base,
        gripper_transform,
        matrix_to_rotation_6d,
    )
    if len(p1) != len(baseline):
        raise ValueError("saved target length must match Point Bridge actions")
    g1 = raw_gripper
    return {
        "A": np.concatenate([p0, g0[:, None]], axis=1),
        "B": np.concatenate([p0, g1[:, None]], axis=1),
        "C": np.concatenate([p1, g0[:, None]], axis=1),
        "D": np.concatenate([p1, g1[:, None]], axis=1),
    }


def extract_switch_window(
    rows: list[dict[str, Any]],
    switch_index: int | None,
    radius: int = SWITCH_WINDOW_RADIUS,
) -> list[dict[str, Any]]:
    if switch_index is None:
        return []
    return [
        dict(row, relative_to_applied_close=row["action_index"] - switch_index)
        for row in rows
        if abs(row["action_index"] - switch_index) <= radius
    ]


def classify_factorial_outcome(
    group_passed: dict[str, int], baseline_replication_passed: bool
) -> dict[str, Any]:
    """Apply the predeclared four-outcome interpretation with a strict 20/20 gate."""

    if not baseline_replication_passed:
        return {
            "result": "invalid_baseline_replication",
            "formal_gate": "failed",
            "winning_group": None,
            "conclusion": "group_A_did_not_reproduce_the_authoritative_v1r_2g_baseline",
            "next_stage": "repair_factorial_baseline_replication_before_interpretation",
        }
    passing = [group for group, passed in group_passed.items() if passed == 20]
    if group_passed.get("B") == 20 and group_passed.get("C") != 20:
        return {
            "result": "result_1_gripper_alignment_repairs_contract",
            "formal_gate": "passed",
            "winning_group": "B",
            "conclusion": "P0_is_executable_when_gripper_uses_current_transition_alignment_G1",
            "next_stage": "rebuild_pkl_with_p0_g1_recompute_action_stats_and_replay_20_without_training",
        }
    if group_passed.get("C") == 20 and group_passed.get("B") != 20:
        return {
            "result": "result_2_saved_pose_target_repairs_contract",
            "formal_gate": "passed",
            "winning_group": "C",
            "conclusion": "saved_controller_targets_P1_outperform_next_measured_eef_targets_P0",
            "next_stage": "rebuild_pkl_with_p1_g0_recompute_action_stats_and_replay_20_without_training",
        }
    if (
        group_passed.get("D") == 20
        and group_passed.get("B") != 20
        and group_passed.get("C") != 20
    ):
        return {
            "result": "result_3_both_pose_and_gripper_factors_required",
            "formal_gate": "passed",
            "winning_group": "D",
            "conclusion": "saved_controller_targets_P1_and_transition_gripper_G1_are_jointly_required",
            "next_stage": "rebuild_pkl_with_p1_g1_recompute_action_stats_and_replay_20_without_training",
        }
    if not passing:
        return {
            "result": "result_4_no_group_reaches_20_of_20",
            "formal_gate": "failed",
            "winning_group": None,
            "conclusion": "current_saved_state_sequences_do_not_admit_a_simple_pose_or_gripper_index_fix",
            "next_stage": "regenerate_sequential_success_demos_in_formal_runtime_and_stop_using_current_pkl_for_training",
        }
    return {
        "result": "mixed_multiple_groups_reach_20_of_20",
        "formal_gate": "passed",
        "winning_group": None,
        "passing_groups": passing,
        "conclusion": "multiple_action_contracts_pass_and_require_a_minimal_contract_selection_audit",
        "next_stage": "select_minimal_passing_contract_then_rebuild_pkl_and_replay_20_without_training",
    }


def _gripper_qpos(robosuite_env: Any) -> np.ndarray:
    robot = robosuite_env.robots[0]
    indexes = np.asarray(robot._ref_gripper_joint_pos_indexes, dtype=int)
    return np.asarray(robosuite_env.sim.data.qpos[indexes], dtype=np.float64).copy()


def _gripper_aperture(qpos: Any) -> float:
    values = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if len(values) == 2:
        return float(abs(values[0] - values[1]))
    return float(np.sum(np.abs(values)))


def _any_gripper_bowl_contact(robosuite_env: Any) -> bool:
    return bool(
        robosuite_env.check_contact(
            robosuite_env.robots[0].gripper,
            robosuite_env.objects_dict["bowl"],
        )
    )


def _orientation_error(scipy_rotation: Any, actual: Any, expected: Any) -> float:
    return float(
        scipy_rotation.from_matrix(
            np.asarray(actual, dtype=np.float64)
            @ np.asarray(expected, dtype=np.float64).T
        ).magnitude()
    )


def _replay_group(
    env: Any,
    episode: dict[str, Any],
    hdf5_demo: Any,
    actions: np.ndarray,
    raw_actions: np.ndarray,
    group: str,
    layout: int,
    pkl_episode_index: int,
    demo_key: str,
    source_pkl: Path,
    migrated_xml: str,
    migrate_source_xml_sha256: str,
    scipy_rotation: Any,
    rotation_6d_to_matrix: Any,
) -> dict[str, Any]:
    started = time.monotonic()
    observations = episode["observation"]
    states = np.asarray(observations["states"])
    raw_gripper = np.asarray(observations["gripper_states"], dtype=np.float64).reshape(-1)
    saved_target_poses = np.asarray(hdf5_demo["datagen_info/target_pose"], dtype=np.float64)
    original_close_index = first_closing_switch(raw_gripper)
    applied_close_index = first_closing_switch(actions[:, -1])
    record: dict[str, Any] = {
        "layout": layout,
        "pkl_episode_index": pkl_episode_index,
        "hdf5_demo_key": demo_key,
        "group": group,
        "pose_factor": GROUP_SPECS[group]["pose"],
        "gripper_factor": GROUP_SPECS[group]["gripper"],
        "purpose": GROUP_SPECS[group]["purpose"],
        "source_pkl": str(source_pkl.relative_to(ROOT)),
        "source_pkl_sha256": file_sha256(source_pkl),
        "source_hdf5_model_xml_sha256": migrate_source_xml_sha256,
        "runtime_model_xml_sha256": text_sha256(migrated_xml),
        "planned_action_count": int(len(actions)),
        "raw_action_shape": list(raw_actions.shape),
        "deployment_action_shape": list(actions.shape),
        "raw_action_dtype": str(raw_actions.dtype),
        "deployment_action_dtype": str(actions.dtype),
        "normalization_roundtrip_max_abs_error": float(
            np.max(np.abs(actions - raw_actions))
        ),
        "gripper_sign_preserved": bool(
            np.array_equal(np.sign(actions[:, -1]), np.sign(raw_actions[:, -1]))
        ),
        "original_raw_close_action_index": original_close_index,
        "pointbridge_applied_close_action_index": applied_close_index,
        "close_index_offset_steps": (
            None
            if original_close_index is None or applied_close_index is None
            else applied_close_index - original_close_index
        ),
        "restored_initial_state_match": False,
        "steps_executed": 0,
        "first_success_action_index": None,
        "success": False,
        "failure_stage": "simulator",
        "exception": None,
        "switch_window": [],
    }
    try:
        env.reset()
        core = _pointbridge_core_env(env)
        robosuite_env = core._env
        robosuite_env.reset_to({"states": states[0], "model": migrated_xml})
        _synchronize_restored_runtime(core, float(raw_gripper[0]))
        controller = _controller_for(robosuite_env)
        if controller.use_delta:
            raise RuntimeError("factorial requires absolute OSC_POSE after reset_to")
        actual_state = np.asarray(robosuite_env.sim.get_state().flatten())
        expected_state = np.asarray(states[0])
        state_match = bool(
            actual_state.shape == expected_state.shape
            and np.array_equal(actual_state.astype(expected_state.dtype), expected_state)
        )
        state_error = float(np.max(np.abs(actual_state - expected_state)))
        record.update(
            {
                "expected_initial_state_sha256": raw_array_sha256(expected_state),
                "actual_initial_state_sha256": raw_array_sha256(
                    actual_state.astype(expected_state.dtype)
                ),
                "initial_state_max_abs_error": state_error,
                "restored_initial_state_match": state_match,
                "robot_base_sha256": raw_array_sha256(np.asarray(core.robot_base)),
            }
        )
        if not state_match:
            raise RuntimeError(f"restored initial state mismatch: max_abs={state_error}")

        phase = _phase_observation(robosuite_env)
        min_eef_bowl = phase["eef_bowl_distance_m"]
        min_bowl_plate = phase["bowl_plate_distance_m"]
        grasped_any = bool(phase["grasped"])
        dropped_after_grasp = False
        initial_contact = _any_gripper_bowl_contact(robosuite_env)
        initial_grasp = bool(phase["grasped"])
        rows: list[dict[str, Any]] = []
        for action_index, action in enumerate(actions):
            time_step = env.step(action)
            controller.update(force=True)
            actual_native = np.asarray(core._current_pose, dtype=np.float64)
            target_native_rotation = rotation_6d_to_matrix(action[3:9])
            actual_native_rotation = rotation_6d_to_matrix(actual_native[3:9])
            position_error = float(np.linalg.norm(action[:3] - actual_native[:3]))
            rotation_error = _orientation_error(
                scipy_rotation, target_native_rotation, actual_native_rotation
            )
            phase = _phase_observation(robosuite_env)
            contact = _any_gripper_bowl_contact(robosuite_env)
            qpos = _gripper_qpos(robosuite_env)
            aperture = _gripper_aperture(qpos)
            success = bool(time_step.observation.get("goal_achieved", False))
            min_eef_bowl = min(min_eef_bowl, phase["eef_bowl_distance_m"])
            min_bowl_plate = min(min_bowl_plate, phase["bowl_plate_distance_m"])
            if phase["grasped"]:
                grasped_any = True
            elif grasped_any:
                dropped_after_grasp = True
            saved_target = saved_target_poses[action_index]
            closed_without_bowl = bool(
                action[-1] > 0.0
                and aperture <= CLOSED_APERTURE_THRESHOLD_M
                and not contact
                and not phase["grasped"]
            )
            rows.append(
                {
                    "action_index": action_index,
                    "commanded_gripper": float(action[-1]),
                    "eef_to_bowl_distance_m": float(phase["eef_bowl_distance_m"]),
                    "actual_gripper_joint_positions_m": qpos,
                    "actual_gripper_aperture_m": aperture,
                    "contact": contact,
                    "grasped": bool(phase["grasped"]),
                    "closed_without_bowl": closed_without_bowl,
                    "task_success": success,
                    "tracking_error": {
                        "position_m": position_error,
                        "rotation_rad": rotation_error,
                    },
                    "actual_controller_vs_saved_target": {
                        "position_m": float(
                            np.linalg.norm(
                                np.asarray(controller.goal_pos) - saved_target[:3, 3]
                            )
                        ),
                        "rotation_rad": _orientation_error(
                            scipy_rotation,
                            np.asarray(controller.goal_ori),
                            saved_target[:3, :3],
                        ),
                    },
                }
            )
            if success:
                record["first_success_action_index"] = action_index
                break

        contact_values = [row["contact"] for row in rows]
        grasp_values = [row["grasped"] for row in rows]
        first_contact = next(
            (row["action_index"] for row in rows if row["contact"]),
            -1 if initial_contact else None,
        )
        first_grasp = next(
            (row["action_index"] for row in rows if row["grasped"]),
            -1 if initial_grasp else None,
        )
        stable_offset = first_stable_true_run(grasp_values)
        first_stable = None if stable_offset is None else rows[stable_offset]["action_index"]
        switch_row = next(
            (
                row
                for row in rows
                if row["action_index"] == applied_close_index
            ),
            None,
        )
        record.update(
            {
                "steps_executed": len(rows),
                "terminated_on_first_success": True,
                "success": bool(rows and rows[-1]["task_success"]),
                "min_eef_bowl_distance_m": float(min_eef_bowl),
                "min_bowl_plate_distance_m": float(min_bowl_plate),
                "final_eef_bowl_distance_m": float(phase["eef_bowl_distance_m"]),
                "final_bowl_plate_distance_m": float(phase["bowl_plate_distance_m"]),
                "initial_contact": initial_contact,
                "first_contact_action_index": first_contact,
                "contact_steps": int(sum(contact_values)),
                "first_grasp_action_index": first_grasp,
                "first_stable_grasp_action_index": first_stable,
                "stable_grasp_definition_steps": STABLE_GRASP_STEPS,
                "grasped_steps": int(sum(grasp_values)),
                "longest_grasp_run_steps": longest_true_run(grasp_values),
                "grasped_any": grasped_any,
                "dropped_after_grasp": dropped_after_grasp,
                "close_command_eef_bowl_distance_m": (
                    None
                    if switch_row is None
                    else switch_row["eef_to_bowl_distance_m"]
                ),
                "actual_gripper_joint_positions_at_close_m": (
                    None
                    if switch_row is None
                    else switch_row["actual_gripper_joint_positions_m"]
                ),
                "actual_gripper_aperture_at_close_m": (
                    None
                    if switch_row is None
                    else switch_row["actual_gripper_aperture_m"]
                ),
                "contact_at_close": None if switch_row is None else switch_row["contact"],
                "grasped_at_close": None if switch_row is None else switch_row["grasped"],
                "closed_without_bowl_observed": any(
                    row["closed_without_bowl"] for row in rows
                ),
                "first_closed_without_bowl_action_index": next(
                    (
                        row["action_index"]
                        for row in rows
                        if row["closed_without_bowl"]
                    ),
                    None,
                ),
                "switch_window": extract_switch_window(rows, applied_close_index),
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
        and record["gripper_sign_preserved"]
        and not record["exception"]
    )
    record["wall_clock_seconds"] = time.monotonic() - started
    return _jsonable(record)


def _summarize_groups(records: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for group in GROUP_SPECS:
        selected = [record for record in records if record["group"] == group]
        per_layout = {}
        for layout in range(1, 5):
            layout_records = [record for record in selected if record["layout"] == layout]
            per_layout[str(layout)] = {
                "checked": len(layout_records),
                "passed": sum(bool(record["passed"]) for record in layout_records),
            }
        summaries[group] = {
            **GROUP_SPECS[group],
            "checked": len(selected),
            "passed": sum(bool(record["passed"]) for record in selected),
            "per_layout": per_layout,
            "restored_initial_state_matches": sum(
                bool(record["restored_initial_state_match"]) for record in selected
            ),
            "simulator_exceptions": sum(bool(record["exception"]) for record in selected),
            "first_contact_observed": sum(
                record.get("first_contact_action_index") is not None for record in selected
            ),
            "stable_grasp_observed": sum(
                record.get("first_stable_grasp_action_index") is not None
                for record in selected
            ),
            "closed_without_bowl_observed": sum(
                bool(record.get("closed_without_bowl_observed")) for record in selected
            ),
            "gripper_sign_preserved": all(
                bool(record["gripper_sign_preserved"]) for record in selected
            ),
            "planned_action_lengths_sha256": raw_array_sha256(
                np.asarray([record["planned_action_count"] for record in selected])
            ),
        }
    return summaries


def _baseline_replication(
    records: list[dict[str, Any]], prior_result: dict[str, Any]
) -> dict[str, Any]:
    current = {
        (record["layout"], record["pkl_episode_index"], record["hdf5_demo_key"]): bool(
            record["success"]
        )
        for record in records
        if record["group"] == "A"
    }
    prior = {
        (record["layout"], record["pkl_episode_index"], record["hdf5_demo_key"]): bool(
            record["success"]
        )
        for record in prior_result["records"]
    }
    selected_ids_match = set(current) == set(prior)
    outcome_matches = bool(
        selected_ids_match and all(current[key] == prior[key] for key in current)
    )
    current_passed = sum(current.values())
    prior_passed = int(prior_result["passed"])
    return {
        "authoritative_report": "experiments/v1r/reports/pointbridge_absolute_pose_contract.json",
        "authoritative_report_sha256": file_sha256(
            ROOT / "experiments" / "v1r" / "reports" / "pointbridge_absolute_pose_contract.json"
        ),
        "selected_trajectory_ids_match": selected_ids_match,
        "per_trajectory_success_matches": outcome_matches,
        "current_checked": len(current),
        "current_passed": current_passed,
        "prior_checked": int(prior_result["checked"]),
        "prior_passed": prior_passed,
        "passed": bool(
            selected_ids_match
            and outcome_matches
            and len(current) == int(prior_result["checked"]) == 20
            and current_passed == prior_passed == 13
        ),
    }


def _paired_effects(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_group = {
        group: {
            (record["layout"], record["pkl_episode_index"], record["hdf5_demo_key"]): bool(
                record["success"]
            )
            for record in records
            if record["group"] == group
        }
        for group in GROUP_SPECS
    }
    baseline = by_group["A"]
    effects = {}
    baseline_failures = {key for key, success in baseline.items() if not success}
    for group in ("B", "C", "D"):
        outcomes = by_group[group]
        repaired = sorted(key for key in baseline_failures if outcomes.get(key, False))
        regressed = sorted(
            key for key, success in baseline.items() if success and not outcomes.get(key, False)
        )
        effects[group] = {
            "baseline_failures_repaired": len(repaired),
            "baseline_successes_regressed": len(regressed),
            "repaired_trajectory_ids": [list(key) for key in repaired],
            "regressed_trajectory_ids": [list(key) for key in regressed],
            "repairs_at_least_6_of_7_baseline_failures": len(repaired) >= 6,
        }
    return {
        "baseline_failure_count": len(baseline_failures),
        "comparisons": effects,
    }


CSV_FIELDS = [
    "layout",
    "pkl_episode_index",
    "hdf5_demo_key",
    "group",
    "pose_factor",
    "gripper_factor",
    "planned_action_count",
    "steps_executed",
    "original_raw_close_action_index",
    "pointbridge_applied_close_action_index",
    "close_index_offset_steps",
    "close_command_eef_bowl_distance_m",
    "actual_gripper_joint_positions_at_close_m",
    "actual_gripper_aperture_at_close_m",
    "contact_at_close",
    "grasped_at_close",
    "first_contact_action_index",
    "first_stable_grasp_action_index",
    "longest_grasp_run_steps",
    "closed_without_bowl_observed",
    "success",
    "failure_stage",
    "exception",
] + [
    (
        f"position_error_close_minus_{abs(offset)}_m"
        if offset < 0
        else f"position_error_close_plus_{offset}_m"
        if offset > 0
        else "position_error_at_close_m"
    )
    for offset in range(-SWITCH_WINDOW_RADIUS, SWITCH_WINDOW_RADIUS + 1)
]


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in CSV_FIELDS}
            window_errors = {
                item["relative_to_applied_close"]: item["tracking_error"]["position_m"]
                for item in record.get("switch_window", [])
            }
            for offset in range(-SWITCH_WINDOW_RADIUS, SWITCH_WINDOW_RADIUS + 1):
                field = (
                    f"position_error_close_minus_{abs(offset)}_m"
                    if offset < 0
                    else f"position_error_close_plus_{offset}_m"
                    if offset > 0
                    else "position_error_at_close_m"
                )
                row[field] = window_errors.get(offset)
            if isinstance(row["actual_gripper_joint_positions_at_close_m"], list):
                row["actual_gripper_joint_positions_at_close_m"] = json.dumps(
                    row["actual_gripper_joint_positions_at_close_m"],
                    separators=(",", ":"),
                )
            writer.writerow(row)


def _markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "## Material Passport",
        "",
        "- Schema: ARS Material Passport 9",
        "- Material type: controlled simulation factorial report",
        "- Source: the same 20 local Point Bridge demonstrations used by V1-R.2G",
        "- Raw data handling: PKL/HDF5/checkpoints remain local and gitignored",
        "- Verification status: VERIFIED",
        "",
        "# V1-R.2H Pose-Target / Gripper-Alignment Factorial",
        "",
        f"Status: `{result['status']}`. Outcome: `{result['outcome']['result']}`.",
        "",
        "## Fixed Contract",
        "",
        "- Runtime: robosuite `1.4.1`, MuJoCo `3.3.5`, `OSC_POSE`, `control_delta=False`, 20 Hz.",
        "- Selection: the identical 20 V1-R.2G demonstrations and exact saved initial states.",
        "- Only pose source P0/P1 and gripper index G0/G1 vary; XML, robot base, rotation helpers, sign convention, source trajectory length, and success predicate remain fixed.",
        f"- Stable grasp means `_check_grasp=True` for at least {STABLE_GRASP_STEPS} consecutive control steps.",
        f"- Closed-without-bowl means commanded close, actual finger aperture <= {CLOSED_APERTURE_THRESHOLD_M:.3f} m, and neither bowl contact nor grasp is present.",
        "- The formal direct-expert action-contract threshold remains 20/20.",
        "",
        "## Results",
        "",
        "| Group | Pose | Gripper | L1 | L2 | L3 | L4 | Total | Formal gate |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for group in GROUP_SPECS:
        summary = result["groups"][group]
        layout_values = [
            f"{summary['per_layout'][str(layout)]['passed']}/5" for layout in range(1, 5)
        ]
        lines.append(
            f"| {group} | {summary['pose']} | {summary['gripper']} | "
            + " | ".join(layout_values)
            + f" | {summary['passed']}/20 | {'pass' if summary['passed'] == 20 else 'fail'} |"
        )
    lines.extend(
        [
            "",
            f"Baseline A replication: `{str(result['baseline_replication']['passed']).lower()}` "
            f"({result['baseline_replication']['current_passed']}/20 versus authoritative "
            f"{result['baseline_replication']['prior_passed']}/20).",
            "",
            "## Paired Baseline-Failure Effects",
            "",
            "| Group | A failures repaired | A successes regressed | Repairs >=6/7 |",
            "|---|---:|---:|---|",
        ]
    )
    for group in ("B", "C", "D"):
        effect = result["paired_effects"]["comparisons"][group]
        lines.append(
            f"| {group} | {effect['baseline_failures_repaired']}/7 | "
            f"{effect['baseline_successes_regressed']} | "
            f"{str(effect['repairs_at_least_6_of_7_baseline_failures']).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Result: `{result['outcome']['result']}`",
            f"- Conclusion: `{result['outcome']['conclusion']}`",
            f"- Formal factorial gate: `{result['outcome']['formal_gate']}`",
            f"- Next stage: `{result['outcome']['next_stage']}`",
            "- B0/B1 training, confirm rollouts, V2, and V3 remain unauthorized.",
            "",
            "## Per-Trajectory Grasp Table",
            "",
            "| Layout | Demo | Group | Raw close | Applied close | Offset | Distance at close (m) | First contact | Stable grasp | Grasp run | Closed empty | Success |",
            "|---:|---|---|---:|---:|---:|---:|---|---:|---:|---|---|",
        ]
    )
    for record in result["records"]:
        distance = record.get("close_command_eef_bowl_distance_m")
        distance_text = "NA" if distance is None else f"{distance:.6f}"
        lines.append(
            f"| {record['layout']} | {record['hdf5_demo_key']} | {record['group']} | "
            f"{record.get('original_raw_close_action_index')} | "
            f"{record.get('pointbridge_applied_close_action_index')} | "
            f"{record.get('close_index_offset_steps')} | {distance_text} | "
            f"{record.get('first_contact_action_index')} | "
            f"{record.get('first_stable_grasp_action_index')} | "
            f"{record.get('longest_grasp_run_steps', 0)} | "
            f"{str(bool(record.get('closed_without_bowl_observed'))).lower()} | "
            f"{str(bool(record.get('success'))).lower()} |"
        )
    lines.extend(
        [
            "",
            "The companion JSON retains the close-switch +/-5-step EEF error, contact, grasp, actual finger-joint, aperture, and closed-without-bowl evidence for every group and trajectory.",
            "",
        ]
    )
    return "\n".join(lines)


def _gate_yaml(result: dict[str, Any]) -> str:
    lines = [
        "stage: V1-R.2H",
        "name: pose-target and gripper-alignment 2x2 factorial",
        f"status: {result['status']}",
        f"outcome: {result['outcome']['result']}",
        f"factorial_gate: {result['outcome']['formal_gate']}",
        "formal_threshold: 20/20",
        f"baseline_replication: {'passed' if result['baseline_replication']['passed'] else 'failed'}",
        "b0_b1_training_authorized: false",
        "confirm_rollouts_authorized: false",
        "v2_formal_experiment_authorized: false",
        "v3_formal_experiment_authorized: false",
        f"next_stage: {result['outcome']['next_stage']}",
        "groups:",
    ]
    for group in GROUP_SPECS:
        summary = result["groups"][group]
        lines.extend(
            [
                f"  {group}:",
                f"    pose: {summary['pose']}",
                f"    gripper: {summary['gripper']}",
                f"    checked: {summary['checked']}",
                f"    passed: {summary['passed']}",
                "    per_layout:",
            ]
        )
        for layout in range(1, 5):
            row = summary["per_layout"][str(layout)]
            lines.append(
                f"      {layout}: {{checked: {row['checked']}, passed: {row['passed']}}}"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--per-layout", type=int, default=PER_LAYOUT, choices=(PER_LAYOUT,))
    parser.add_argument(
        "--prior-report",
        type=Path,
        default=ROOT
        / "experiments"
        / "v1r"
        / "reports"
        / "pointbridge_absolute_pose_contract.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument(
        "--runtime-output-dir",
        type=Path,
        default=ROOT / "outputs" / "v1r" / "pointbridge_pose_gripper_factorial_runtime",
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
    from point_bridge.robot_utils.mimiclabs.utils import T_gripper
    import point_bridge.suite.mimiclabs as pb_suite

    sys.path.insert(0, str(ROOT / "src"))
    from vico_point.envs.mimiclabs_compat import migrate_saved_model_xml

    robosuite_version = str(getattr(robosuite, "__version__", "unknown"))
    mujoco_version = str(getattr(mujoco, "__version__", "unknown"))
    if robosuite_version != REQUIRED_ROBOSUITE or mujoco_version != REQUIRED_MUJOCO:
        raise RuntimeError(
            f"formal V1-R.2H requires robosuite {REQUIRED_ROBOSUITE} and "
            f"MuJoCo {REQUIRED_MUJOCO}; got {robosuite_version} and {mujoco_version}"
        )
    prior_result = json.loads(args.prior_report.resolve().read_text(encoding="utf-8"))
    if prior_result.get("stage") != "V1-R.2G" or prior_result.get("passed") != 13:
        raise RuntimeError("prior report is not the authoritative 13/20 V1-R.2G result")

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
    p1_roundtrip_errors: dict[str, Any] = {}
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
                env,
                robosuite_version=robosuite_version,
                mujoco_version=mujoco_version,
            )
            if not current_runtime["passed"]:
                raise RuntimeError(f"official runtime contract failed: {current_runtime}")
            if runtime_contract is None:
                runtime_contract = current_runtime
            elif current_runtime != runtime_contract:
                raise RuntimeError("runtime contract changed between layouts")
            metadata = source_metadata[source_pkl.resolve()]
            core = _pointbridge_core_env(env)
            coordinate_contract = coordinate_frame_contract(
                metadata["robot_base"], core.robot_base
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
                raise RuntimeError(f"layout {layout} robot-base contract failed")

            with h5py.File(hdf5_path, "r") as hdf5:
                for pkl_episode_index, episode in enumerate(
                    dataset._episodes[path_index][: args.per_layout]
                ):
                    states = np.asarray(episode["observation"]["states"])
                    demo_key, hdf5_demo = _hdf5_demo_for_episode(hdf5["data"], states)
                    source_actions = np.asarray(hdf5_demo["actions"], dtype=np.float64)
                    raw_gripper = np.asarray(
                        episode["observation"]["gripper_states"], dtype=np.float64
                    ).reshape(-1)
                    if not np.array_equal(source_actions[:, -1], raw_gripper):
                        raise RuntimeError(
                            f"layout {layout} {demo_key} PKL/HDF5 raw gripper mismatch"
                        )
                    saved_targets = np.asarray(
                        hdf5_demo["datagen_info/target_pose"], dtype=np.float64
                    )
                    raw_groups = build_raw_factorial_actions(
                        episode["action"],
                        raw_gripper,
                        saved_targets,
                        core.robot_base,
                        T_gripper,
                        matrix_to_rotation_6d,
                    )
                    p1_world = pointbridge_pose_to_world_targets(
                        raw_groups["C"][:, :9],
                        core.robot_base,
                        T_gripper,
                        rotation_6d_to_matrix,
                    )
                    position_error = float(
                        np.max(
                            np.linalg.norm(
                                p1_world[:, :3, 3] - saved_targets[:, :3, 3], axis=1
                            )
                        )
                    )
                    rotation_error = float(
                        max(
                            _orientation_error(R, actual[:3, :3], expected[:3, :3])
                            for actual, expected in zip(p1_world, saved_targets)
                        )
                    )
                    p1_roundtrip_errors[
                        f"{layout}:{pkl_episode_index}:{demo_key}"
                    ] = {
                        "position_max_error_m": position_error,
                        "rotation_max_error_rad": rotation_error,
                    }
                    if position_error > 1e-12 or rotation_error > 1e-7:
                        raise RuntimeError(
                            f"layout {layout} {demo_key} P1 transform roundtrip failed: "
                            f"position={position_error}, rotation={rotation_error}"
                        )

                    original_xml = hdf5_demo.attrs["model_file"]
                    if isinstance(original_xml, bytes):
                        original_xml = original_xml.decode("utf-8")
                    migrated_xml = migrate_saved_model_xml(str(original_xml))
                    for group in GROUP_SPECS:
                        normalized, deployment_actions = normalize_and_postprocess(
                            dataset, raw_groups[group]
                        )
                        if normalized.dtype != np.float32 or deployment_actions.dtype != np.float64:
                            raise RuntimeError("training/deployment dtype contract changed")
                        record = _replay_group(
                            env=env,
                            episode=episode,
                            hdf5_demo=hdf5_demo,
                            actions=deployment_actions,
                            raw_actions=raw_groups[group],
                            group=group,
                            layout=layout,
                            pkl_episode_index=pkl_episode_index,
                            demo_key=demo_key,
                            source_pkl=source_pkl,
                            migrated_xml=migrated_xml,
                            migrate_source_xml_sha256=text_sha256(str(original_xml)),
                            scipy_rotation=R,
                            rotation_6d_to_matrix=rotation_6d_to_matrix,
                        )
                        records.append(record)
                        print(
                            f"layout={layout} episode={pkl_episode_index} demo={demo_key} "
                            f"group={group} success={record['success']} "
                            f"stage={record['failure_stage']}",
                            flush=True,
                        )
        finally:
            env.close()

    if runtime_contract is None:
        raise RuntimeError("no Point Bridge runtime was created")
    groups = _summarize_groups(records)
    baseline_replication = _baseline_replication(records, prior_result)
    paired_effects = _paired_effects(records)
    outcome = classify_factorial_outcome(
        {group: int(summary["passed"]) for group, summary in groups.items()},
        baseline_replication["passed"],
    )
    invariant_contract = {
        "same_20_v1r_2g_trajectories": baseline_replication[
            "selected_trajectory_ids_match"
        ],
        "restored_initial_state_matches": sum(
            bool(record["restored_initial_state_match"]) for record in records
        ),
        "expected_restored_initial_state_matches": 80,
        "controller_configuration_fixed": runtime_contract["passed"],
        "control_frequency_hz": CONTROL_FREQ_HZ,
        "environment_xml_fixed_within_each_trajectory": all(
            len(
                {
                    record["runtime_model_xml_sha256"]
                    for record in records
                    if record["layout"] == layout
                    and record["pkl_episode_index"] == episode
                }
            )
            == 1
            for layout in range(1, 5)
            for episode in range(PER_LAYOUT)
        ),
        "robot_base_contracts_passed": all(
            contract["passed"] for contract in coordinate_contracts.values()
        ),
        "rotation_conversion_fixed": "point_bridge.robot_utils.common.utils matrix_to_rotation_6d / rotation_6d_to_matrix",
        "gripper_sign_preserved": all(
            bool(record["gripper_sign_preserved"]) for record in records
        ),
        "source_trajectory_lengths_fixed_across_groups": all(
            len(
                {
                    record["planned_action_count"]
                    for record in records
                    if record["layout"] == layout
                    and record["pkl_episode_index"] == episode
                }
            )
            == 1
            for layout in range(1, 5)
            for episode in range(PER_LAYOUT)
        ),
        "success_predicate_fixed": "Point Bridge goal_achieved from the official environment; terminate on first success as in V1-R.2G",
        "policy_inference_used": False,
        "language_encoder_inference_used": False,
        "confirm_rollouts_run": False,
        "new_trajectories_added": False,
        "p1_world_target_roundtrip_max_position_error_m": max(
            item["position_max_error_m"] for item in p1_roundtrip_errors.values()
        ),
        "p1_world_target_roundtrip_max_rotation_error_rad": max(
            item["rotation_max_error_rad"] for item in p1_roundtrip_errors.values()
        ),
    }
    invariants_passed = bool(
        invariant_contract["same_20_v1r_2g_trajectories"]
        and invariant_contract["restored_initial_state_matches"] == 80
        and invariant_contract["controller_configuration_fixed"]
        and invariant_contract["environment_xml_fixed_within_each_trajectory"]
        and invariant_contract["robot_base_contracts_passed"]
        and invariant_contract["gripper_sign_preserved"]
        and invariant_contract["source_trajectory_lengths_fixed_across_groups"]
        and not any(record["exception"] for record in records)
    )
    status = outcome["formal_gate"] if invariants_passed else "invalid"
    result = _jsonable(
        {
            "stage": "V1-R.2H",
            "audit": "pose_target_and_gripper_alignment_factorial",
            "status": status,
            "formal_threshold": {"checked": 20, "required_passed": 20},
            "runtime": runtime_contract,
            "dataset_contract": dataset_contract,
            "design": {
                "factors": {
                    "pose": {
                        "P0": "x[t+1] measured EEF pose from native BCDataset",
                        "P1": "x[t] saved controller target from datagen_info/target_pose",
                    },
                    "gripper": {
                        "G0": "g[t+1] raw gripper state from native BCDataset",
                        "G1": "g[t] raw state-transition gripper command",
                    },
                },
                "groups": GROUP_SPECS,
                "stable_grasp_steps": STABLE_GRASP_STEPS,
                "closed_aperture_threshold_m": CLOSED_APERTURE_THRESHOLD_M,
                "switch_window_radius_steps": SWITCH_WINDOW_RADIUS,
            },
            "invariants": invariant_contract,
            "invariants_passed": invariants_passed,
            "coordinate_frame_contracts": coordinate_contracts,
            "p1_roundtrip_by_trajectory": p1_roundtrip_errors,
            "baseline_replication": baseline_replication,
            "groups": groups,
            "paired_effects": paired_effects,
            "outcome": outcome,
            "authorization": {
                "b0_b1_training_authorized": False,
                "confirm_rollouts_authorized": False,
                "v2_formal_experiment_authorized": False,
                "v3_formal_experiment_authorized": False,
            },
            "records": records,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.gate_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.csv_output.resolve(), records)
    args.report.write_text(_markdown_report(result), encoding="utf-8")
    args.gate_output.write_text(_gate_yaml(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "groups": {
                    group: summary["passed"] for group, summary in result["groups"].items()
                },
                "baseline_replication": result["baseline_replication"]["passed"],
                "outcome": result["outcome"],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
