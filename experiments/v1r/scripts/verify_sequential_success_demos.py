#!/usr/bin/env python3
"""Verify V1-R.2I sequential demos and compare S0/S1 absolute labels.

The three checks are intentionally reported separately:

1. replay the exact delta commands captured from the formal runtime;
2. replay S0, the original Point Bridge next-measured-pose label path;
3. replay S1, the recorded absolute controller target plus its command gripper.

This script never treats stable grasp as a hard acceptance condition.  The
formal acceptance predicate is task success during a continuous sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPSTREAM = ROOT / "third_party" / "pointbridge"
REQUIRED_ROBOSUITE = "1.4.1"
REQUIRED_MUJOCO = "3.3.5"
CONTROL_FREQ_HZ = 20

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_sequential_success_demos import (  # noqa: E402
    add_upstream_paths,
    controller_for,
    make_env,
    model_xml_text,
    robot_base_transform,
    raw_array_sha256,
    synchronize_runtime_state,
)


def matrix_to_rotation_6d(matrix: Any) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    return value[..., :2, :].reshape(value.shape[:-2] + (6,))


def eef_quaternion_to_pointbridge_pose(
    eef_states: Any, matrix_to_rotation_6d_fn: Any = matrix_to_rotation_6d
) -> np.ndarray:
    """Convert Point Bridge [xyz, xyzw] observations to [xyz, rotation6d]."""

    from scipy.spatial.transform import Rotation as rotation

    values = np.asarray(eef_states, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError(f"expected eef states shaped (T, 7), got {values.shape}")
    matrices = rotation.from_quat(values[:, 3:]).as_matrix()
    return np.concatenate(
        [values[:, :3], matrix_to_rotation_6d_fn(matrices)], axis=1
    )


def pointbridge_pose_to_world_actions(
    pointbridge_actions: Any,
    robot_base: Any,
    t_gripper: Any,
    rotation_6d_to_matrix_fn: Any,
) -> np.ndarray:
    """Convert Point Bridge absolute pose actions to robosuite OSC actions."""

    from scipy.spatial.transform import Rotation as rotation

    actions = np.asarray(pointbridge_actions, dtype=np.float64)
    base = np.asarray(robot_base, dtype=np.float64)
    gripper_inverse = np.linalg.inv(np.asarray(t_gripper, dtype=np.float64))
    world_actions = []
    for action in actions:
        pointbridge_target = np.eye(4, dtype=np.float64)
        pointbridge_target[:3, 3] = action[:3]
        pointbridge_target[:3, :3] = rotation_6d_to_matrix_fn(action[3:9])
        world_target = base @ pointbridge_target @ gripper_inverse
        world_actions.append(
            np.concatenate(
                [
                    world_target[:3, 3],
                    rotation.from_matrix(world_target[:3, :3]).as_rotvec(),
                    [action[-1]],
                ]
            )
        )
    return np.asarray(world_actions, dtype=np.float64)


def world_targets_to_actions(world_targets: Any, gripper_commands: Any) -> np.ndarray:
    """Convert recorded world-frame controller targets to OSC actions."""

    from scipy.spatial.transform import Rotation as rotation

    targets = np.asarray(world_targets, dtype=np.float64)
    gripper = np.asarray(gripper_commands, dtype=np.float64).reshape(-1)
    if targets.ndim != 3 or targets.shape[1:] != (4, 4):
        raise ValueError(f"expected world targets shaped (T, 4, 4), got {targets.shape}")
    if len(targets) != len(gripper):
        raise ValueError("world target and gripper lengths differ")
    return np.asarray(
        [
            np.concatenate(
                [
                    target[:3, 3],
                    rotation.from_matrix(target[:3, :3]).as_rotvec(),
                    [command],
                ]
            )
            for target, command in zip(targets, gripper)
        ],
        dtype=np.float64,
    )


def build_s0_labels(
    native_eef_states: Any,
    native_gripper_states: Any,
    matrix_to_rotation_6d_fn: Any = matrix_to_rotation_6d,
) -> np.ndarray:
    """Apply BCDataset's original next-pose/next-gripper terminal-repeat path."""

    eef = np.asarray(native_eef_states, dtype=np.float64)
    gripper = np.asarray(native_gripper_states, dtype=np.float64).reshape(-1)
    if len(eef) != len(gripper) or len(eef) < 2:
        raise ValueError("S0 needs matching native arrays with at least two observations")
    next_eef = eef[1:]
    next_gripper = gripper[1:]
    next_eef = np.concatenate([next_eef, next_eef[-1:]], axis=0)
    next_gripper = np.concatenate([next_gripper, next_gripper[-1:]], axis=0)
    return np.concatenate(
        [eef_quaternion_to_pointbridge_pose(next_eef, matrix_to_rotation_6d_fn), next_gripper[:, None]],
        axis=1,
    )


def build_s1_labels(controller_targets_pointbridge: Any, issued_actions: Any) -> np.ndarray:
    """Use the recorded absolute controller target with the same-step gripper command."""

    targets = np.asarray(controller_targets_pointbridge, dtype=np.float64)
    commands = np.asarray(issued_actions, dtype=np.float64)
    if targets.ndim != 3 or targets.shape[1:] != (4, 4):
        raise ValueError(f"expected Point Bridge targets shaped (T, 4, 4), got {targets.shape}")
    if commands.ndim != 2 or commands.shape[1] != 7 or len(targets) != len(commands):
        raise ValueError("S1 target and issued-action shapes are incompatible")
    return np.concatenate(
        [
            np.asarray(
                [
                    np.concatenate(
                        [target[:3, 3], matrix_to_rotation_6d(target[:3, :3])]
                    )
                    for target in targets
                ]
            ),
            commands[:, -1:,],
        ],
        axis=1,
    )


def replay_actions(env: Any, initial_state: Any, model_xml: str, actions: Any) -> dict[str, Any]:
    """Replay one action sequence after one initial reset."""

    env.reset_to({"states": np.asarray(initial_state), "model": model_xml})
    synchronize_runtime_state(env)
    actual_initial = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
    success_steps: list[int] = []
    for index, action in enumerate(np.asarray(actions, dtype=np.float64)):
        env.step(action)
        if bool(env._check_success()):
            success_steps.append(index)
    return {
        "initial_state_match": bool(np.array_equal(actual_initial, np.asarray(initial_state))),
        "initial_state_max_abs_error": float(
            np.max(np.abs(actual_initial - np.asarray(initial_state)))
        ),
        "success": bool(success_steps),
        "first_success_action_index": success_steps[0] if success_steps else None,
        "steps_executed": len(actions),
    }


def verify_record(
    delta_env: Any,
    absolute_env: Any,
    record: dict[str, Any],
    root: Path,
    fallback_robot_base: np.ndarray,
    t_gripper: np.ndarray,
    rotation_6d_to_matrix_fn: Any,
) -> dict[str, Any]:
    artifact_path = root / record["artifact"]
    model_path = root / record["model_xml"]
    with np.load(artifact_path, allow_pickle=False) as bundle:
        arrays = {key: np.asarray(bundle[key]).copy() for key in bundle.files}
    model_xml = model_path.read_text(encoding="utf-8")
    initial_state = arrays["initial_state"]
    robot_base = np.asarray(
        arrays.get("robot_base", fallback_robot_base), dtype=np.float64
    )
    issued_actions = arrays["issued_actions"]
    result: dict[str, Any] = {
        "layout": record["layout"],
        "demo_key": record["demo_key"],
        "artifact": record["artifact"],
        "actual_command_replay": None,
        "s0_label_replay": None,
        "s1_label_replay": None,
        "exception": None,
    }
    try:
        result["actual_command_replay"] = replay_actions(
            delta_env, initial_state, model_xml, issued_actions
        )
        s0_labels = build_s0_labels(
            arrays["native_eef_states"],
            arrays["native_gripper_states"],
        )
        s0_world_actions = pointbridge_pose_to_world_actions(
            s0_labels, robot_base, t_gripper, rotation_6d_to_matrix_fn
        )
        result["s0_label_replay"] = replay_actions(
            absolute_env, initial_state, model_xml, s0_world_actions
        )
        s1_labels = build_s1_labels(
            arrays["controller_targets_pointbridge"], issued_actions
        )
        s1_world_actions = pointbridge_pose_to_world_actions(
            s1_labels, robot_base, t_gripper, rotation_6d_to_matrix_fn
        )
        result["s1_label_replay"] = replay_actions(
            absolute_env, initial_state, model_xml, s1_world_actions
        )
        result["array_hashes"] = {
            "issued_actions": raw_array_sha256(issued_actions),
            "states_after": raw_array_sha256(arrays["states_after"]),
            "native_eef_states": raw_array_sha256(arrays["native_eef_states"]),
            "controller_targets_world": raw_array_sha256(
                arrays["controller_targets_world"]
            ),
        }
    except Exception as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-3000:]}"
    result["actual_command_passed"] = bool(
        result["actual_command_replay"]
        and result["actual_command_replay"]["initial_state_match"]
        and result["actual_command_replay"]["success"]
    )
    result["s0_passed"] = bool(
        result["s0_label_replay"]
        and result["s0_label_replay"]["initial_state_match"]
        and result["s0_label_replay"]["success"]
    )
    result["s1_passed"] = bool(
        result["s1_label_replay"]
        and result["s1_label_replay"]["initial_state_match"]
        and result["s1_label_replay"]["success"]
    )
    return result


def classify_verification(
    actual_passed: int, s0_passed: int, s1_passed: int, checked: int
) -> dict[str, Any]:
    actual_gate = actual_passed == checked
    s0_gate = s0_passed == checked
    s1_gate = s1_passed == checked
    if actual_gate and s0_gate:
        decision = "s0_original_pointbridge_label_contract_passed"
    elif actual_gate and s1_gate:
        decision = "s1_recorded_controller_target_contract_only"
    elif actual_gate:
        decision = "actual_commands_pass_but_no_absolute_label_contract_passes"
    else:
        decision = "sequential_demo_replay_failed"
    return {
        "actual_command_replay_gate": "passed" if actual_gate else "failed",
        "s0_label_replay_gate": "passed" if s0_gate else "failed",
        "s1_label_replay_gate": "passed" if s1_gate else "failed",
        "decision": decision,
        "training_authorized": False,
        "v2_formal_experiment_authorized": False,
        "v3_formal_experiment_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "v1r" / "sequential_success_demos_2i" / "verification.json",
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
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    accepted_records = [record for record in manifest["records"] if record.get("accepted")]
    expected_layouts = {1, 2, 3, 4}
    actual_layouts = {int(record["layout"]) for record in accepted_records}
    if actual_layouts != expected_layouts:
        raise ValueError(
            f"verification requires accepted records for layouts 1-4, got {sorted(actual_layouts)}"
        )
    expected_per_layout = int(manifest.get("target_per_layout", 5))
    records_by_layout: dict[int, list[dict[str, Any]]] = {}
    for record in accepted_records:
        records_by_layout.setdefault(int(record["layout"]), []).append(record)
    incomplete = {
        layout: len(records_by_layout.get(layout, []))
        for layout in sorted(expected_layouts)
        if len(records_by_layout.get(layout, [])) != expected_per_layout
    }
    if incomplete:
        raise ValueError(
            f"verification requires exactly {expected_per_layout} records per layout: {incomplete}"
        )
    checked_records = [record for layout in sorted(records_by_layout) for record in records_by_layout[layout]]

    results: list[dict[str, Any]] = []
    for layout in sorted(records_by_layout):
        task_name = f"bowl_on_plate_{layout}"
        hdf5_path = upstream / "data" / "mimicgen_data" / "bowl_on_plate" / task_name / "demo" / "demo.hdf5"
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
        absolute_spec = json.loads(json.dumps(env_spec))
        absolute_spec["env_kwargs"]["controller_configs"]["control_delta"] = False
        absolute_env = make_env(robosuite, absolute_spec, bddl_path, control_delta=False)
        delta_env.reset()
        absolute_env.reset()
        if not bool(controller_for(delta_env).use_delta):
            raise RuntimeError("actual-command replay environment is not delta OSC")
        if bool(controller_for(absolute_env).use_delta):
            raise RuntimeError("S0/S1 replay environment is not absolute OSC")
        try:
            robot_base = robot_base_transform(delta_env, np.asarray(T_robot_base, dtype=np.float64))
            for record in records_by_layout[layout]:
                results.append(
                    verify_record(
                        delta_env,
                        absolute_env,
                        record,
                        ROOT,
                        robot_base,
                        np.asarray(T_gripper, dtype=np.float64),
                        rotation_6d_to_matrix,
                    )
                )
                print(
                    f"layout={layout} demo={record['demo_key']} "
                    f"actual={results[-1]['actual_command_passed']} "
                    f"s0={results[-1]['s0_passed']} s1={results[-1]['s1_passed']}",
                    flush=True,
                )
        finally:
            delta_env.close()
            absolute_env.close()

    checked = len(results)
    actual_passed = sum(bool(result["actual_command_passed"]) for result in results)
    s0_passed = sum(bool(result["s0_passed"]) for result in results)
    s1_passed = sum(bool(result["s1_passed"]) for result in results)
    decision = classify_verification(actual_passed, s0_passed, s1_passed, checked)
    result = {
        "stage": "V1-R.2I.2",
        "status": "passed" if decision["actual_command_replay_gate"] == "passed" else "failed",
        "formal_threshold": {"checked": checked, "required_passed": checked},
        "runtime": {
            "robosuite": REQUIRED_ROBOSUITE,
            "mujoco": REQUIRED_MUJOCO,
            "control_freq_hz": CONTROL_FREQ_HZ,
        },
        "stable_grasp_is_diagnostic_only": True,
        "gates": {
            "actual_command_replay": {"checked": checked, "passed": actual_passed},
            "s0_label_replay": {"checked": checked, "passed": s0_passed},
            "s1_label_replay": {"checked": checked, "passed": s1_passed},
        },
        "decision": decision,
        "records": results,
        "source_manifest": str(manifest_path.relative_to(ROOT)),
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
