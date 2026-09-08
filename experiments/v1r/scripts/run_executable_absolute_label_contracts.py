#!/usr/bin/env python3
"""Run the V1-R.2J executable absolute-action label-contract gate.

The experiment is restricted to the 20 V1-R.2I sequential successes.  It
compares four absolute OSC contracts without policy inference or training:

* S0-old: the historical next-observation plus terminal-repeat labels;
* S0-transition: each measured post-step pose with the same-step gripper;
* S1-PB: the saved controller target through Point Bridge frame / rotation-6D;
* S1-world: the saved world-frame controller target without that roundtrip.

The no-tail gate and the predeclared post-success-tail diagnostic are reported
separately.  Every replay starts from the full saved initial simulator state;
no state is restored between actions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPSTREAM = ROOT / "third_party" / "pointbridge"
DEFAULT_SOURCE_MANIFEST = (
    ROOT / "outputs" / "v1r" / "sequential_success_demos_2i" / "manifest.json"
)
DEFAULT_SOURCE_VERIFICATION = (
    ROOT / "outputs" / "v1r" / "sequential_success_demos_2i" / "verification.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "v1r" / "executable_absolute_contracts_2j"
REQUIRED_ROBOSUITE = "1.4.1"
REQUIRED_MUJOCO = "3.3.5"
CONTROL_FREQ_HZ = 20
REAL_TAIL_STEPS = 10
HOLD_TAIL_STEPS = 5
STATE_DIVERGENCE_ATOL = 1e-9
CONTRACTS = ("s0_old", "s0_transition", "s1_pb", "s1_world")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_sequential_success_demos import (  # noqa: E402
    add_upstream_paths,
    controller_for,
    controller_target_pointbridge,
    controller_target_world,
    eef_pose_pointbridge,
    file_sha256,
    make_env,
    phase_flags,
    raw_array_sha256,
    robot_base_transform,
    synchronize_runtime_state,
)
from verify_sequential_success_demos import (  # noqa: E402
    build_s0_labels,
    build_s1_labels,
    pointbridge_pose_to_world_actions,
    world_targets_to_actions,
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


def matrix_to_rotation_6d(matrix: Any) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    return value[..., :2, :].reshape(value.shape[:-2] + (6,))


def build_s0_transition_labels(
    post_step_eef_states: Any,
    issued_actions: Any,
) -> np.ndarray:
    """Build [x[t+1], g[t]] Point Bridge labels for every real transition."""

    from scipy.spatial.transform import Rotation as rotation

    eef = np.asarray(post_step_eef_states, dtype=np.float64)
    commands = np.asarray(issued_actions, dtype=np.float64)
    if eef.ndim != 2 or eef.shape[1] != 7:
        raise ValueError(f"expected post-step EEF states shaped (T, 7), got {eef.shape}")
    if commands.ndim != 2 or commands.shape[1] != 7 or len(commands) != len(eef):
        raise ValueError("post-step EEF and issued-action shapes are incompatible")
    rotations = rotation.from_quat(eef[:, 3:]).as_matrix()
    poses = np.concatenate(
        [eef[:, :3], matrix_to_rotation_6d(rotations)], axis=1
    )
    return np.concatenate([poses, commands[:, -1:]], axis=1)


def select_tail_plan(
    source_action_count: int,
    captured_action_count: int,
    real_tail_steps: int = REAL_TAIL_STEPS,
    hold_tail_steps: int = HOLD_TAIL_STEPS,
) -> dict[str, Any]:
    """Choose the predeclared tail rule without inspecting replay outcomes."""

    remaining = source_action_count - captured_action_count
    if remaining < 0:
        raise ValueError("captured action count exceeds source action count")
    if remaining >= real_tail_steps:
        return {
            "strategy": "remaining_hdf5_delta_actions",
            "steps": real_tail_steps,
            "source_start_action_index": captured_action_count,
            "source_stop_action_index_exclusive": captured_action_count + real_tail_steps,
            "remaining_source_actions": remaining,
        }
    return {
        "strategy": "final_absolute_target_hold",
        "steps": hold_tail_steps,
        "source_start_action_index": None,
        "source_stop_action_index_exclusive": None,
        "remaining_source_actions": remaining,
    }


def append_final_target_hold(actions: Any, steps: int = HOLD_TAIL_STEPS) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or not len(values):
        raise ValueError(f"expected non-empty world actions shaped (T, 7), got {values.shape}")
    if steps <= 0:
        raise ValueError("hold steps must be positive")
    return np.concatenate([values, np.repeat(values[-1:], steps, axis=0)], axis=0)


def orientation_error_rad(actual: Any, expected: Any) -> float:
    from scipy.spatial.transform import Rotation as rotation

    actual_matrix = np.asarray(actual, dtype=np.float64)
    expected_matrix = np.asarray(expected, dtype=np.float64)
    return float(rotation.from_matrix(actual_matrix @ expected_matrix.T).magnitude())


def world_actions_to_targets(actions: Any) -> np.ndarray:
    from scipy.spatial.transform import Rotation as rotation

    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError(f"expected world actions shaped (T, 7), got {values.shape}")
    targets = np.repeat(np.eye(4, dtype=np.float64)[None], len(values), axis=0)
    targets[:, :3, 3] = values[:, :3]
    targets[:, :3, :3] = rotation.from_rotvec(values[:, 3:6]).as_matrix()
    return targets


def pointbridge_eef_to_world_targets(
    eef_states: Any,
    robot_base: Any,
    t_gripper: Any,
) -> np.ndarray:
    """Convert saved [xyz, xyzw] EEF measurements directly to world targets."""

    from scipy.spatial.transform import Rotation as rotation

    eef = np.asarray(eef_states, dtype=np.float64)
    if eef.ndim != 2 or eef.shape[1] != 7:
        raise ValueError(f"expected EEF states shaped (T, 7), got {eef.shape}")
    pointbridge_targets = np.repeat(
        np.eye(4, dtype=np.float64)[None], len(eef), axis=0
    )
    pointbridge_targets[:, :3, 3] = eef[:, :3]
    pointbridge_targets[:, :3, :3] = rotation.from_quat(eef[:, 3:]).as_matrix()
    base = np.asarray(robot_base, dtype=np.float64)
    gripper_inverse = np.linalg.inv(np.asarray(t_gripper, dtype=np.float64))
    return np.asarray(
        [base @ target @ gripper_inverse for target in pointbridge_targets],
        dtype=np.float64,
    )


def first_divergence_index(
    actual_states: Any,
    reference_states: Any,
    atol: float = STATE_DIVERGENCE_ATOL,
) -> int | None:
    actual = np.asarray(actual_states, dtype=np.float64)
    reference = np.asarray(reference_states, dtype=np.float64)
    if actual.shape != reference.shape:
        raise ValueError(f"state shapes differ: {actual.shape} != {reference.shape}")
    if actual.ndim != 2:
        raise ValueError("state sequences must be two-dimensional")
    errors = np.max(np.abs(actual - reference), axis=1)
    indices = np.flatnonzero(errors > atol)
    return int(indices[0]) if len(indices) else None


def classify_failure(telemetry: list[dict[str, Any]]) -> str:
    if any(row["task_success"] for row in telemetry):
        return "success"
    if not any(row["contact"] for row in telemetry):
        return "no_contact"
    grasp_indices = [row["action_index"] for row in telemetry if row["grasped"]]
    if not grasp_indices:
        return "no_grasp"
    if any(
        not row["grasped"] and row["action_index"] > grasp_indices[0]
        for row in telemetry
    ):
        return "post_grasp_drop"
    return "post_grasp_no_task_success"


def classify_contract_decision(
    base_passed: dict[str, int],
    tail_passed: dict[str, int],
    checked: int,
    actual_delta_passed: int,
) -> dict[str, Any]:
    """Apply the predeclared A/B/C/D decision order."""

    base_gates = {name: base_passed.get(name, 0) == checked for name in CONTRACTS}
    tail_gates = {name: tail_passed.get(name, 0) == checked for name in CONTRACTS}
    selected: str | None = None
    tail_required = False
    if actual_delta_passed != checked:
        case = "invalid_sequential_reference"
        next_action = "repair_sequential_reference_before_label_selection"
    elif base_gates["s0_transition"]:
        case = "B"
        selected = "s0_transition"
        next_action = "freeze_s0_transition_and_rebuild_training_data"
    elif base_gates["s1_pb"]:
        case = "absolute_contract_passed"
        selected = "s1_pb"
        next_action = "freeze_s1_pb_and_rebuild_training_data"
    elif base_gates["s1_world"]:
        case = "A"
        next_action = "repair_pointbridge_frame_or_rotation_roundtrip_then_require_20_of_20"
    else:
        tail_candidates = [
            name for name in ("s0_transition", "s1_pb", "s1_world") if tail_gates[name]
        ]
        if tail_candidates:
            case = "C"
            tail_required = True
            if "s0_transition" in tail_candidates:
                selected = "s0_transition_with_uniform_tail"
                next_action = "freeze_transition_contract_with_uniform_tail_and_rebuild_training_data"
            elif "s1_pb" in tail_candidates:
                selected = "s1_pb_with_uniform_tail"
                next_action = "freeze_s1_pb_with_uniform_tail_and_rebuild_training_data"
            else:
                next_action = "repair_pointbridge_roundtrip_with_uniform_tail_then_require_20_of_20"
        else:
            case = "D"
            next_action = "register_delta_pose_contract_and_verify_normalization_execution_roundtrip"
    return {
        "case": case,
        "base_gates": base_gates,
        "post_success_tail_diagnostic_gates": tail_gates,
        "selected_label_contract": selected,
        "uniform_tail_required": tail_required,
        "next_action": next_action,
        "b0_b1_training_authorized": False,
        "seed0_training_authorized": selected is not None,
        "confirm_rollouts_authorized": False,
        "v2_formal_experiment_authorized": False,
        "v3_formal_experiment_authorized": False,
    }


def validate_source_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = [record for record in manifest["records"] if record.get("accepted")]
    grouped = {layout: [] for layout in (1, 2, 3, 4)}
    for record in records:
        layout = int(record["layout"])
        if layout not in grouped:
            raise ValueError(f"unexpected layout {layout}")
        grouped[layout].append(record)
    counts = {layout: len(items) for layout, items in grouped.items()}
    if any(count != 5 for count in counts.values()) or len(records) != 20:
        raise ValueError(f"V1-R.2J requires exactly five accepted records per layout: {counts}")
    return [record for layout in grouped for record in grouped[layout]]


def capture_delta_reference(
    env: Any,
    initial_state: Any,
    model_xml: str,
    actions: Any,
    robot_base: np.ndarray,
    t_gripper: np.ndarray,
) -> dict[str, Any]:
    """Execute the selected HDF5 delta prefix and record its real transitions."""

    commands = np.asarray(actions, dtype=np.float64)
    env.reset_to({"states": np.asarray(initial_state), "model": model_xml})
    synchronize_runtime_state(env)
    actual_initial = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
    states_before = []
    states_after = []
    native_eef_states = []
    post_step_eef_states = []
    targets_world = []
    targets_pointbridge = []
    contact_flags = []
    grasp_flags = []
    success_flags = []
    for command in commands:
        states_before.append(np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy())
        native_eef_states.append(eef_pose_pointbridge(env, robot_base, t_gripper))
        env.step(command)
        states_after.append(np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy())
        post_step_eef_states.append(eef_pose_pointbridge(env, robot_base, t_gripper))
        target_world = controller_target_world(env)
        targets_world.append(target_world)
        targets_pointbridge.append(
            controller_target_pointbridge(target_world, robot_base, t_gripper)
        )
        contact, grasped, success = phase_flags(env)
        contact_flags.append(contact)
        grasp_flags.append(grasped)
        success_flags.append(success)
    return {
        "initial_state": actual_initial,
        "states_before": np.asarray(states_before),
        "states_after": np.asarray(states_after),
        "issued_actions": commands.copy(),
        "native_eef_states": np.asarray(native_eef_states),
        "native_gripper_states": commands[:, -1].copy(),
        "post_step_eef_states": np.asarray(post_step_eef_states),
        "controller_targets_world": np.asarray(targets_world),
        "controller_targets_pointbridge": np.asarray(targets_pointbridge),
        "contact_flags": np.asarray(contact_flags, dtype=np.bool_),
        "grasp_flags": np.asarray(grasp_flags, dtype=np.bool_),
        "success_flags": np.asarray(success_flags, dtype=np.bool_),
    }


def build_contract_sequences(
    arrays: dict[str, np.ndarray],
    robot_base: np.ndarray,
    t_gripper: np.ndarray,
    rotation_6d_to_matrix_fn: Any,
) -> dict[str, dict[str, np.ndarray]]:
    issued = arrays["issued_actions"]
    s0_old_labels = build_s0_labels(
        arrays["native_eef_states"], arrays["native_gripper_states"]
    )
    s0_transition_labels = build_s0_transition_labels(
        arrays["post_step_eef_states"], issued
    )
    s1_labels = build_s1_labels(arrays["controller_targets_pointbridge"], issued)
    world_actions = {
        "s0_old": pointbridge_pose_to_world_actions(
            s0_old_labels, robot_base, t_gripper, rotation_6d_to_matrix_fn
        ),
        "s0_transition": pointbridge_pose_to_world_actions(
            s0_transition_labels, robot_base, t_gripper, rotation_6d_to_matrix_fn
        ),
        "s1_pb": pointbridge_pose_to_world_actions(
            s1_labels, robot_base, t_gripper, rotation_6d_to_matrix_fn
        ),
        "s1_world": world_targets_to_actions(
            arrays["controller_targets_world"], issued[:, -1]
        ),
    }
    s0_old_eef = np.concatenate(
        [arrays["native_eef_states"][1:], arrays["native_eef_states"][-1:]],
        axis=0,
    )
    captured_targets = {
        "s0_old": pointbridge_eef_to_world_targets(
            s0_old_eef, robot_base, t_gripper
        ),
        "s0_transition": pointbridge_eef_to_world_targets(
            arrays["post_step_eef_states"], robot_base, t_gripper
        ),
        "s1_pb": arrays["controller_targets_world"].copy(),
        "s1_world": arrays["controller_targets_world"].copy(),
    }
    return {
        name: {
            "actions": actions,
            "input_targets_world": world_actions_to_targets(actions),
            "captured_targets_world": captured_targets[name],
        }
        for name, actions in world_actions.items()
    }


def extend_with_hold(sequence: dict[str, np.ndarray], steps: int) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([values, np.repeat(values[-1:], steps, axis=0)], axis=0)
        for key, values in sequence.items()
    }


def replay_absolute_actions(
    env: Any,
    initial_state: np.ndarray,
    model_xml: str,
    actions: np.ndarray,
    captured_targets_world: np.ndarray,
    reference_states_after: np.ndarray,
    base_action_count: int,
) -> dict[str, Any]:
    """Replay one contract and retain per-step controller and state telemetry."""

    env.reset_to({"states": initial_state, "model": model_xml})
    synchronize_runtime_state(env)
    actual_initial = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
    controller = controller_for(env)
    telemetry: list[dict[str, Any]] = []
    actual_states = []
    reference_count = len(reference_states_after)
    for action_index, action in enumerate(actions):
        env.step(action)
        actual_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy()
        actual_states.append(actual_state)
        input_target = world_actions_to_targets(np.asarray(action)[None])[0]
        captured_target = captured_targets_world[action_index]
        reference_available = action_index < reference_count
        if reference_available:
            state_difference = np.abs(actual_state - reference_states_after[action_index])
            state_max_abs_error = float(np.max(state_difference))
            state_l2_error = float(np.linalg.norm(state_difference))
            state_diverged = state_max_abs_error > STATE_DIVERGENCE_ATOL
        else:
            state_max_abs_error = None
            state_l2_error = None
            state_diverged = None
        contact, grasped, success = phase_flags(env)
        goal_pos = np.asarray(controller.goal_pos, dtype=np.float64).copy()
        goal_ori = np.asarray(controller.goal_ori, dtype=np.float64).copy()
        telemetry.append(
            {
                "action_index": action_index,
                "tail_step": action_index >= base_action_count,
                "input_absolute_action": np.asarray(action, dtype=np.float64),
                "controller_goal_pos_m": goal_pos,
                "controller_goal_ori_matrix": goal_ori,
                "controller_goal_vs_input_action_target": {
                    "translation_error_m": float(
                        np.linalg.norm(goal_pos - input_target[:3, 3])
                    ),
                    "rotation_error_rad": orientation_error_rad(
                        goal_ori, input_target[:3, :3]
                    ),
                },
                "controller_goal_vs_captured_target": {
                    "translation_error_m": float(
                        np.linalg.norm(goal_pos - captured_target[:3, 3])
                    ),
                    "rotation_error_rad": orientation_error_rad(
                        goal_ori, captured_target[:3, :3]
                    ),
                },
                "continuous_delta_state_reference_available": reference_available,
                "state_max_abs_error": state_max_abs_error,
                "state_l2_error": state_l2_error,
                "state_diverged": state_diverged,
                "contact": contact,
                "close_command": bool(action[-1] > 0.0),
                "grasped": grasped,
                "task_success": success,
            }
        )
    success_indices = [row["action_index"] for row in telemetry if row["task_success"]]
    divergence_indices = [
        row["action_index"] for row in telemetry if row["state_diverged"] is True
    ]
    exact_divergence_indices = [
        index
        for index, actual_state in enumerate(actual_states[:reference_count])
        if not np.array_equal(actual_state, reference_states_after[index])
    ]
    base_success = any(index < base_action_count for index in success_indices)
    return {
        "initial_state_match": bool(np.array_equal(actual_initial, initial_state)),
        "initial_state_max_abs_error": float(np.max(np.abs(actual_initial - initial_state))),
        "steps_executed": len(actions),
        "base_steps": base_action_count,
        "tail_steps": len(actions) - base_action_count,
        "base_success": base_success,
        "success": bool(success_indices),
        "first_success_action_index": success_indices[0] if success_indices else None,
        "first_state_divergence_action_index": (
            divergence_indices[0] if divergence_indices else None
        ),
        "first_exact_state_divergence_action_index": (
            exact_divergence_indices[0] if exact_divergence_indices else None
        ),
        "state_divergence_atol": STATE_DIVERGENCE_ATOL,
        "first_contact_action_index": next(
            (row["action_index"] for row in telemetry if row["contact"]), None
        ),
        "first_close_command_action_index": next(
            (row["action_index"] for row in telemetry if row["close_command"]), None
        ),
        "first_grasp_action_index": next(
            (row["action_index"] for row in telemetry if row["grasped"]), None
        ),
        "contact_observed": any(row["contact"] for row in telemetry),
        "grasp_observed": any(row["grasped"] for row in telemetry),
        "failure_stage": classify_failure(telemetry),
        "max_controller_goal_vs_captured_translation_error_m": max(
            row["controller_goal_vs_captured_target"]["translation_error_m"]
            for row in telemetry
        ),
        "max_controller_goal_vs_captured_rotation_error_rad": max(
            row["controller_goal_vs_captured_target"]["rotation_error_rad"]
            for row in telemetry
        ),
        "telemetry": telemetry,
    }


def save_reference_artifact(path: Path, reference: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **reference)


def compact_replay(replay: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in replay.items() if key != "telemetry"}


def run_record(
    delta_env: Any,
    absolute_env: Any,
    source_record: dict[str, Any],
    source_actions: np.ndarray,
    source_arrays: dict[str, np.ndarray],
    model_xml: str,
    t_gripper: np.ndarray,
    rotation_6d_to_matrix_fn: Any,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    layout = int(source_record["layout"])
    demo_key = str(source_record["demo_key"])
    base_count = len(source_arrays["issued_actions"])
    tail_plan = select_tail_plan(len(source_actions), base_count)
    result: dict[str, Any] = {
        "layout": layout,
        "demo_key": demo_key,
        "source_artifact": source_record["artifact"],
        "source_artifact_sha256": source_record["artifact_sha256"],
        "base_action_count": base_count,
        "source_action_count": len(source_actions),
        "tail_plan": tail_plan,
        "source_action_prefix_match": False,
        "delta_reference": None,
        "contracts": {},
        "exception": None,
    }
    try:
        source_prefix = source_actions[:base_count]
        result["source_action_prefix_match"] = bool(
            source_prefix.shape == source_arrays["issued_actions"].shape
            and np.array_equal(source_prefix, source_arrays["issued_actions"])
        )
        if not result["source_action_prefix_match"]:
            raise RuntimeError("V1-R.2I issued actions do not match the HDF5 prefix")
        initial_state = source_arrays["initial_state"]
        robot_base = source_arrays["robot_base"]
        base_sequences = build_contract_sequences(
            source_arrays, robot_base, t_gripper, rotation_6d_to_matrix_fn
        )

        delta_count = base_count
        if tail_plan["strategy"] == "remaining_hdf5_delta_actions":
            delta_count += int(tail_plan["steps"])
        delta_reference = capture_delta_reference(
            delta_env,
            initial_state,
            model_xml,
            source_actions[:delta_count],
            robot_base,
            t_gripper,
        )
        reference_stem = f"layout_{layout}_demo_{int(demo_key.rsplit('_', 1)[1]):03d}"
        reference_path = output_dir / "references" / f"{reference_stem}.npz"
        save_reference_artifact(reference_path, delta_reference)
        base_state_exact_match = bool(
            np.array_equal(
                delta_reference["states_after"][:base_count],
                source_arrays["states_after"],
            )
        )
        delta_success_indices = np.flatnonzero(delta_reference["success_flags"])
        result["delta_reference"] = {
            "artifact": str(reference_path.relative_to(ROOT)),
            "artifact_sha256": file_sha256(reference_path),
            "initial_state_match": bool(
                np.array_equal(delta_reference["initial_state"], initial_state)
            ),
            "base_state_exact_match": base_state_exact_match,
            "base_state_max_abs_error": float(
                np.max(
                    np.abs(
                        delta_reference["states_after"][:base_count]
                        - source_arrays["states_after"]
                    )
                )
            ),
            "base_success": bool(np.any(delta_reference["success_flags"][:base_count])),
            "success": bool(len(delta_success_indices)),
            "first_success_action_index": (
                int(delta_success_indices[0]) if len(delta_success_indices) else None
            ),
            "executed_actions": delta_count,
            "issued_actions_sha256": raw_array_sha256(
                delta_reference["issued_actions"]
            ),
            "states_after_sha256": raw_array_sha256(
                delta_reference["states_after"]
            ),
        }

        if tail_plan["strategy"] == "remaining_hdf5_delta_actions":
            tail_sequences = build_contract_sequences(
                delta_reference, robot_base, t_gripper, rotation_6d_to_matrix_fn
            )
            tail_reference_states = delta_reference["states_after"]
        else:
            tail_sequences = {
                name: extend_with_hold(sequence, int(tail_plan["steps"]))
                for name, sequence in base_sequences.items()
            }
            tail_reference_states = source_arrays["states_after"]

        for contract in CONTRACTS:
            base = base_sequences[contract]
            tail = tail_sequences[contract]
            base_replay = replay_absolute_actions(
                absolute_env,
                initial_state,
                model_xml,
                base["actions"],
                base["captured_targets_world"],
                source_arrays["states_after"],
                base_count,
            )
            tail_replay = replay_absolute_actions(
                absolute_env,
                initial_state,
                model_xml,
                tail["actions"],
                tail["captured_targets_world"],
                tail_reference_states,
                base_count,
            )
            result["contracts"][contract] = {
                "base": base_replay,
                "post_success_tail_diagnostic": tail_replay,
                "tail_prefix_action_max_abs_difference": float(
                    np.max(np.abs(tail["actions"][:base_count] - base["actions"]))
                ),
                "base_actions_sha256": raw_array_sha256(base["actions"]),
                "tail_actions_sha256": raw_array_sha256(tail["actions"]),
            }
    except Exception as exc:
        result["exception"] = (
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-5000:]}"
        )
    result["wall_clock_seconds"] = time.monotonic() - started
    return jsonable(result)


def aggregate_results(
    records: list[dict[str, Any]],
    prior_verification: dict[str, Any],
) -> dict[str, Any]:
    checked = len(records)
    base_passed = {
        contract: sum(
            bool(record["contracts"][contract]["base"]["success"])
            for record in records
            if not record.get("exception")
        )
        for contract in CONTRACTS
    }
    tail_passed = {
        contract: sum(
            bool(
                record["contracts"][contract]["post_success_tail_diagnostic"][
                    "success"
                ]
            )
            for record in records
            if not record.get("exception")
        )
        for contract in CONTRACTS
    }
    actual_delta_passed = sum(
        bool((record.get("delta_reference") or {}).get("base_success"))
        for record in records
        if not record.get("exception")
    )
    initial_matches = sum(
        bool((record.get("delta_reference") or {}).get("initial_state_match"))
        for record in records
        if not record.get("exception")
    )
    prior_by_key = {
        (int(record["layout"]), str(record["demo_key"])): record
        for record in prior_verification["records"]
    }
    replication_records = []
    for record in records:
        if record.get("exception"):
            continue
        prior = prior_by_key[(int(record["layout"]), str(record["demo_key"]))]
        replication_records.append(
            bool(record["contracts"]["s0_old"]["base"]["success"])
            == bool(prior["s0_passed"])
            and bool(record["contracts"]["s1_pb"]["base"]["success"])
            == bool(prior["s1_passed"])
        )
    per_layout: dict[str, Any] = {}
    for layout in (1, 2, 3, 4):
        items = [record for record in records if int(record["layout"]) == layout]
        per_layout[str(layout)] = {
            "checked": len(items),
            "base_passed": {
                contract: sum(
                    bool(item["contracts"][contract]["base"]["success"])
                    for item in items
                    if not item.get("exception")
                )
                for contract in CONTRACTS
            },
            "post_success_tail_diagnostic_passed": {
                contract: sum(
                    bool(
                        item["contracts"][contract]["post_success_tail_diagnostic"][
                            "success"
                        ]
                    )
                    for item in items
                    if not item.get("exception")
                )
                for contract in CONTRACTS
            },
        }
    decision = classify_contract_decision(
        base_passed, tail_passed, checked, actual_delta_passed
    )
    invariants = {
        "same_20_v1r_2i_demonstrations": checked == 20,
        "source_action_prefix_matches": sum(
            bool(record["source_action_prefix_match"]) for record in records
        ),
        "delta_initial_state_matches": initial_matches,
        "delta_base_successes": actual_delta_passed,
        "base_delta_state_exact_matches": sum(
            bool((record.get("delta_reference") or {}).get("base_state_exact_match"))
            for record in records
            if not record.get("exception")
        ),
        "simulator_exceptions": sum(bool(record.get("exception")) for record in records),
        "s0_old_and_s1_pb_v1r_2i_outcomes_reproduced": bool(
            len(replication_records) == checked and all(replication_records)
        ),
        "policy_inference_used": False,
        "training_run": False,
        "intermediate_state_restore_count": 0,
    }
    invariants["passed"] = bool(
        checked == 20
        and invariants["source_action_prefix_matches"] == 20
        and initial_matches == 20
        and actual_delta_passed == 20
        and invariants["base_delta_state_exact_matches"] == 20
        and invariants["simulator_exceptions"] == 0
        and invariants["s0_old_and_s1_pb_v1r_2i_outcomes_reproduced"]
    )
    return {
        "checked": checked,
        "base_passed": base_passed,
        "post_success_tail_diagnostic_passed": tail_passed,
        "per_layout": per_layout,
        "invariants": invariants,
        "decision": decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument(
        "--source-verification", type=Path, default=DEFAULT_SOURCE_VERIFICATION
    )
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "results.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse per-record checkpoints only when their runner hash matches",
    )
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
    from point_bridge.robot_utils.common.utils import rotation_6d_to_matrix
    from point_bridge.robot_utils.mimiclabs.utils import T_gripper, T_robot_base

    if str(getattr(robosuite, "__version__", "unknown")) != REQUIRED_ROBOSUITE:
        raise RuntimeError(f"expected robosuite {REQUIRED_ROBOSUITE}")
    if str(getattr(mujoco, "__version__", "unknown")) != REQUIRED_MUJOCO:
        raise RuntimeError(f"expected MuJoCo {REQUIRED_MUJOCO}")

    manifest_path = args.manifest.resolve()
    verification_path = args.source_verification.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior_verification = json.loads(verification_path.read_text(encoding="utf-8"))
    source_records = validate_source_records(manifest)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "record_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runner_sha256 = file_sha256(Path(__file__).resolve())
    records: list[dict[str, Any]] = []

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
        with h5py.File(hdf5_path, "r") as handle:
            env_spec = json.loads(str(handle["data"].attrs["env_args"]))
            delta_env = make_env(robosuite, env_spec, bddl_path, control_delta=True)
            absolute_env = make_env(robosuite, env_spec, bddl_path, control_delta=False)
            delta_env.reset()
            absolute_env.reset()
            if not bool(controller_for(delta_env).use_delta):
                raise RuntimeError("delta reference environment is not delta OSC")
            if bool(controller_for(absolute_env).use_delta):
                raise RuntimeError("label replay environment is not absolute OSC")
            try:
                runtime_base = robot_base_transform(
                    delta_env, np.asarray(T_robot_base, dtype=np.float64)
                )
                for source_record in (
                    record for record in source_records if int(record["layout"]) == layout
                ):
                    checkpoint_path = checkpoint_dir / (
                        f"layout_{layout}_demo_"
                        f"{int(source_record['demo_key'].rsplit('_', 1)[1]):03d}.json"
                    )
                    if args.resume and checkpoint_path.is_file():
                        checkpoint = json.loads(
                            checkpoint_path.read_text(encoding="utf-8")
                        )
                        if checkpoint.get("runner_sha256") != runner_sha256:
                            raise RuntimeError(
                                f"checkpoint runner hash mismatch: {checkpoint_path}"
                            )
                        records.append(checkpoint)
                        print(
                            f"layout={layout} demo={source_record['demo_key']} resumed=true",
                            flush=True,
                        )
                        continue
                    artifact_path = ROOT / source_record["artifact"]
                    if file_sha256(artifact_path) != source_record["artifact_sha256"]:
                        raise RuntimeError(f"artifact hash mismatch: {artifact_path}")
                    with np.load(artifact_path, allow_pickle=False) as bundle:
                        source_arrays = {
                            key: np.asarray(bundle[key]).copy() for key in bundle.files
                        }
                    if not np.array_equal(runtime_base, source_arrays["robot_base"]):
                        raise RuntimeError(
                            f"robot base mismatch for layout {layout} {source_record['demo_key']}"
                        )
                    model_path = ROOT / source_record["model_xml"]
                    if file_sha256(model_path) != source_record["model_xml_sha256"]:
                        raise RuntimeError(f"model XML hash mismatch: {model_path}")
                    source_actions = np.asarray(
                        handle["data"][source_record["demo_key"]]["actions"],
                        dtype=np.float64,
                    )
                    record = run_record(
                        delta_env,
                        absolute_env,
                        source_record,
                        source_actions,
                        source_arrays,
                        model_path.read_text(encoding="utf-8"),
                        np.asarray(T_gripper, dtype=np.float64),
                        rotation_6d_to_matrix,
                        output_dir,
                    )
                    record["runner_sha256"] = runner_sha256
                    checkpoint_path.write_text(
                        json.dumps(record, separators=(",", ":")) + "\n",
                        encoding="utf-8",
                    )
                    records.append(record)
                    base_status = {
                        name: record.get("contracts", {}).get(name, {}).get("base", {}).get("success")
                        for name in CONTRACTS
                    }
                    tail_status = {
                        name: record.get("contracts", {}).get(name, {}).get(
                            "post_success_tail_diagnostic", {}
                        ).get("success")
                        for name in CONTRACTS
                    }
                    print(
                        f"layout={layout} demo={source_record['demo_key']} "
                        f"tail={record['tail_plan']['strategy']} "
                        f"base={base_status} tail_result={tail_status}",
                        flush=True,
                    )
            finally:
                delta_env.close()
                absolute_env.close()

    aggregate = aggregate_results(records, prior_verification)
    result = {
        "stage": "V1-R.2J",
        "status": "completed" if aggregate["invariants"]["passed"] else "invalid",
        "formal_threshold": {"checked_per_contract": 20, "required_passed": 20},
        "runtime": {
            "robosuite": REQUIRED_ROBOSUITE,
            "mujoco": REQUIRED_MUJOCO,
            "control_freq_hz": CONTROL_FREQ_HZ,
            "controller": "OSC_POSE",
            "delta_reference_control_delta": True,
            "absolute_replay_control_delta": False,
        },
        "contracts": {
            "s0_old": "native_eef[1:] plus terminal repeat; next-index gripper",
            "s0_transition": "post_step_eef_states[t] plus issued_actions[t, gripper]",
            "s1_pb": "controller target through Point Bridge frame and rotation-6D roundtrip",
            "s1_world": "direct controller_targets_world[t] absolute OSC action",
        },
        "post_success_tail_diagnostic": {
            "real_source_rule": f"append exactly {REAL_TAIL_STEPS} remaining HDF5 delta transitions",
            "fallback_rule": f"when fewer than {REAL_TAIL_STEPS} remain, append {HOLD_TAIL_STEPS} repeats of each contract's final absolute target",
            "outcome_selection_blind": True,
            "reported_separately_from_base": True,
        },
        "state_divergence_atol": STATE_DIVERGENCE_ATOL,
        "source": {
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": file_sha256(manifest_path),
            "verification": str(verification_path.relative_to(ROOT)),
            "verification_sha256": file_sha256(verification_path),
        },
        **aggregate,
        "records": records,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(jsonable(result), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
