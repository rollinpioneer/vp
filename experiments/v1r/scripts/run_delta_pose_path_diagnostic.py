#!/usr/bin/env python3
"""Run the V1-R.2J-F delta-pose execution-path diagnostic.

The diagnostic is intentionally restricted to three V1-R.2J delta failures
and one successful control.  It crosses two independent factors:

* execution environment: the HDF5 ``env_args`` environment used by capture,
  or the Point Bridge ``delta_pose`` entry point;
* action values: the original issued commands, direct float32 commands, or
  the existing shared min-max -> float32 -> inverse path.

Every replay starts from the captured initial state once and then executes the
full captured action prefix.  The lowest robosuite ``step`` receives a trace
entry before each call, allowing the report to distinguish outer actions from
the actual controller input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPSTREAM = ROOT / "third_party" / "pointbridge"
DEFAULT_MANIFEST = ROOT / "outputs" / "v1r" / "sequential_success_demos_2i" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "v1r" / "delta_pose_path_diagnostic_2j_f" / "results_24.json"
REQUIRED_ROBOSUITE = "1.4.1"
REQUIRED_MUJOCO = "3.3.5"
CONTROL_FREQ_HZ = 20
STATE_DIVERGENCE_ATOL = 1e-9
POSE_DIVERGENCE_ATOL = 1e-12
SELECTED = {
    (1, "demo_18"),
    (3, "demo_10"),
    (4, "demo_2"),
    (2, "demo_27"),
}
ACTION_SOURCE_BY_CONDITION = {
    "A_capture_raw": "raw_issued_actions_float64",
    "B_capture_normalized": "shared_minmax_float32_inverse_float64",
    "C_pointbridge_raw": "raw_issued_actions_float64",
    "D_pointbridge_normalized": "shared_minmax_float32_inverse_float64",
    "E_capture_float32": "raw_issued_actions_float32",
    "F_pointbridge_float32": "raw_issued_actions_float32",
    "G_pointbridge_float32_label_float64_controller": "raw_float32_label_promoted_to_float64_controller",
}


def float32_labels_to_float64_controller(actions: Any) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply the proposed model/controller precision boundary without normalization."""

    raw = np.asarray(actions, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 7:
        raise ValueError(f"delta_pose actions must have shape (T, 7), got {raw.shape}")
    labels = raw.astype(np.float32)
    controller_actions = labels.astype(np.float64)
    return labels, controller_actions, float(np.max(np.abs(controller_actions - raw)))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_sequential_success_demos import (  # noqa: E402
    add_upstream_paths,
    controller_for,
    file_sha256,
    make_env,
    raw_array_sha256,
    synchronize_runtime_state,
)
from verify_delta_pose_contract import (  # noqa: E402
    _PrecomputedLanguageEncoder,
    locate_core_environment,
    normalize_delta_actions,
    shared_action_stats,
    synchronized_reset,
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


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def find_robosuite_env(env: Any) -> Any:
    """Find the actual robosuite environment below Point Bridge wrappers."""

    current = env
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        values = vars(current) if hasattr(current, "__dict__") else {}
        if "sim" in values and "robots" in values:
            return current
        next_env = values.get("_env", values.get("env"))
        if next_env is None:
            break
        current = next_env
    raise RuntimeError("could not locate underlying robosuite environment")


def eef_and_object_positions(env: Any) -> dict[str, np.ndarray]:
    sim = env.sim
    return {
        "eef": np.asarray(sim.data.get_body_xpos("gripper0_eef"), dtype=np.float64).copy(),
        "bowl": np.asarray(sim.data.get_body_xpos("bowl_main"), dtype=np.float64).copy(),
        "plate": np.asarray(sim.data.get_body_xpos("plate_main"), dtype=np.float64).copy(),
    }


def phase_flags(env: Any) -> tuple[bool, bool, bool]:
    gripper = env.robots[0].gripper
    bowl = env.objects_dict["bowl"]
    return (
        bool(env.check_contact(gripper, bowl)),
        bool(env._check_grasp(gripper, bowl)),
        bool(env._check_success()),
    )


def controller_snapshot(controller: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key in (
        "use_delta",
        "input_min",
        "input_max",
        "output_min",
        "output_max",
        "kp",
        "damping",
        "impedance_mode",
        "uncouple_pos_ori",
        "interpolation",
        "ramp_ratio",
        "position_limits",
        "orientation_limits",
    ):
        if hasattr(controller, key):
            values[key] = jsonable(getattr(controller, key))
    values["goal_pos"] = jsonable(np.asarray(controller.goal_pos, dtype=np.float64))
    values["goal_ori"] = jsonable(np.asarray(controller.goal_ori, dtype=np.float64))
    return values


def env_descriptor(env: Any, source: str, env_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    leaf = find_robosuite_env(env)
    controller = controller_for(leaf)
    descriptor = {
        "source": source,
        "robosuite_class": type(leaf).__name__,
        "controller_class": type(controller).__name__,
        "controller": controller_snapshot(controller),
        "control_freq_hz": CONTROL_FREQ_HZ,
    }
    if env_spec is not None:
        descriptor["hdf5_env_args_sha256"] = canonical_sha256(env_spec)
        descriptor["hdf5_controller_config"] = jsonable(
            env_spec["env_kwargs"]["controller_configs"]
        )
    descriptor["descriptor_sha256"] = canonical_sha256(descriptor)
    return descriptor


@contextmanager
def trace_lowest_step(env: Any) -> Iterator[dict[str, list[dict[str, Any]]]]:
    """Trace bottom actions and the controller's actual ``set_goal`` calls."""

    leaf = find_robosuite_env(env)
    controller = controller_for(leaf)
    original_step = leaf.step
    original_set_goal = controller.set_goal
    trace: dict[str, list[dict[str, Any]]] = {"bottom": [], "goals": []}

    def traced_step(self: Any, action: Any, *args: Any, **kwargs: Any) -> Any:
        array = np.asarray(action).copy()
        trace["bottom"].append(
            {
                "call_index": len(trace["bottom"]),
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "action": array,
            }
        )
        return original_step(action, *args, **kwargs)

    def traced_set_goal(self: Any, action: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_set_goal(action, *args, **kwargs)
        trace["goals"].append(
            {
                "call_index": len(trace["goals"]),
                "action": np.asarray(action).copy(),
                "goal_pos": np.asarray(self.goal_pos, dtype=np.float64).copy(),
                "goal_ori": np.asarray(self.goal_ori, dtype=np.float64).copy(),
            }
        )
        return result

    leaf.step = types.MethodType(traced_step, leaf)
    controller.set_goal = types.MethodType(traced_set_goal, controller)
    try:
        yield trace
    finally:
        delattr(leaf, "step")
        delattr(controller, "set_goal")


def prepare_capture_reset(env: Any, initial_state: np.ndarray, model_xml: str) -> Any:
    env.reset_to({"states": initial_state, "model": model_xml})
    synchronize_runtime_state(env)
    return env


def load_record(record: dict[str, Any]) -> tuple[dict[str, np.ndarray], str]:
    artifact = ROOT / record["artifact"]
    model_path = ROOT / record["model_xml"]
    if file_sha256(artifact) != record["artifact_sha256"]:
        raise RuntimeError(f"artifact hash mismatch: {artifact}")
    with np.load(artifact, allow_pickle=False) as bundle:
        arrays = {key: np.asarray(bundle[key]).copy() for key in bundle.files}
    return arrays, model_path.read_text(encoding="utf-8")


def state_and_controller_error(reference: np.ndarray, actual: np.ndarray) -> float:
    if reference.shape != actual.shape:
        return float("inf")
    return float(np.max(np.abs(actual - reference)))


def orientation_angle(first: np.ndarray, second: np.ndarray) -> float:
    from scipy.spatial.transform import Rotation

    relative = np.asarray(first) @ np.asarray(second).T
    return float(Rotation.from_matrix(relative).magnitude())


def failure_stage(success: bool, contacts: list[bool], grasps: list[bool]) -> str:
    if success:
        return "success"
    if any(grasps):
        return "grasp_without_task_success"
    if any(contacts):
        return "contact_without_grasp"
    return "no_contact"


def replay_condition(
    env: Any,
    environment_source: str,
    condition: str,
    record: dict[str, Any],
    arrays: dict[str, np.ndarray],
    model_xml: str,
    actions: np.ndarray,
    input_actions: np.ndarray,
    normalization_error: float | None,
) -> dict[str, Any]:
    started = time.monotonic()
    initial_state = np.asarray(arrays["initial_state"], dtype=np.float64)
    reference_states = np.asarray(arrays["states_after"], dtype=np.float64)
    leaf = find_robosuite_env(env)
    controller = controller_for(leaf)
    if environment_source == "capture_env_args":
        prepare_capture_reset(env, initial_state, model_xml)
    else:
        synchronized_reset(env, initial_state, model_xml)
    actual_initial = np.asarray(leaf.sim.get_state().flatten(), dtype=np.float64)
    telemetry: list[dict[str, Any]] = []
    contacts: list[bool] = []
    grasps: list[bool] = []
    success_indices: list[int] = []
    state_errors: list[float] = []
    outer_dtypes: list[str] = []
    with trace_lowest_step(env) as traces:
        bottom_trace = traces["bottom"]
        goal_trace = traces["goals"]
        for action_index, action in enumerate(np.asarray(actions)):
            outer = np.asarray(action).copy()
            outer_dtypes.append(str(outer.dtype))
            time_step = env.step(outer)
            actual_state = np.asarray(leaf.sim.get_state().flatten(), dtype=np.float64)
            contact, grasped, success = phase_flags(leaf)
            positions = eef_and_object_positions(leaf)
            goal_entry = goal_trace[-1] if goal_trace else None
            goal_pos = (
                np.asarray(goal_entry["goal_pos"], dtype=np.float64).copy()
                if goal_entry is not None
                else np.asarray(controller.goal_pos, dtype=np.float64).copy()
            )
            goal_ori = (
                np.asarray(goal_entry["goal_ori"], dtype=np.float64).copy()
                if goal_entry is not None
                else np.asarray(controller.goal_ori, dtype=np.float64).copy()
            )
            state_error = state_and_controller_error(reference_states[action_index], actual_state)
            state_errors.append(state_error)
            contacts.append(contact)
            grasps.append(grasped)
            if success:
                success_indices.append(action_index)
            bottom_entry = bottom_trace[-1] if bottom_trace else None
            bottom_action = (
                np.asarray(bottom_entry["action"], dtype=np.float64).copy()
                if bottom_entry is not None
                else np.empty((0,), dtype=np.float64)
            )
            telemetry.append(
                {
                    "action_index": action_index,
                    "outer_action": outer,
                    "outer_action_dtype": str(outer.dtype),
                    "bottom_action": bottom_action,
                    "bottom_action_dtype": bottom_entry["dtype"] if bottom_entry else None,
                    "bottom_call_count": len(bottom_trace),
                    "bottom_call_index": bottom_entry["call_index"] if bottom_entry else None,
                    "controller_set_goal_call_count": len(goal_trace),
                    "controller_set_goal_call_index": goal_entry["call_index"] if goal_entry else None,
                    "bottom_action_max_abs_error_vs_outer": (
                        float(np.max(np.abs(bottom_action - outer)))
                        if bottom_action.shape == outer.shape
                        else float("inf")
                    ),
                    "controller_goal_pos": goal_pos,
                    "controller_goal_ori": goal_ori,
                    "eef_position": positions["eef"],
                    "bowl_position": positions["bowl"],
                    "plate_position": positions["plate"],
                    "state_max_abs_error_vs_capture_reference": state_error,
                    "state_diverged": bool(state_error > STATE_DIVERGENCE_ATOL),
                    "contact": contact,
                    "grasped": grasped,
                    "task_success": success,
                    "time_step_last": bool(getattr(time_step, "last", lambda: False)()),
                }
            )
    bottom_actions = [np.asarray(item["action"], dtype=np.float64) for item in bottom_trace]
    bottom_array = np.asarray(bottom_actions, dtype=np.float64)
    expected = np.asarray(actions, dtype=np.float64)
    bottom_vs_expected = (
        float(np.max(np.abs(bottom_array - expected)))
        if bottom_array.shape == expected.shape
        else float("inf")
    )
    return jsonable(
        {
            "layout": int(record["layout"]),
            "demo_key": record["demo_key"],
            "environment_source": environment_source,
            "condition": condition,
            "input_action_source": ACTION_SOURCE_BY_CONDITION[condition],
            "input_action_dtype": str(input_actions.dtype),
            "outer_action_count": len(actions),
            "bottom_step_call_count": len(bottom_trace),
            "bottom_step_count_matches_outer": len(bottom_trace) == len(actions),
            "bottom_action_shape": list(bottom_array.shape),
            "bottom_action_dtype_counts": {
                dtype: sum(item["dtype"] == dtype for item in bottom_trace)
                for dtype in sorted({item["dtype"] for item in bottom_trace})
            },
            "bottom_action_max_abs_error_vs_outer_max": bottom_vs_expected,
            "bottom_actions_sha256": raw_array_sha256(bottom_array),
            "outer_actions_sha256": raw_array_sha256(expected),
            "normalization_roundtrip_max_abs_error": normalization_error,
            "initial_state_match": bool(np.array_equal(actual_initial, initial_state)),
            "initial_state_max_abs_error": float(np.max(np.abs(actual_initial - initial_state))),
            "success": bool(success_indices),
            "first_success_action_index": success_indices[0] if success_indices else None,
            "first_contact_action_index": next((i for i, value in enumerate(contacts) if value), None),
            "first_grasp_action_index": next((i for i, value in enumerate(grasps) if value), None),
            "first_state_divergence_action_index": next(
                (i for i, value in enumerate(state_errors) if value > STATE_DIVERGENCE_ATOL),
                None,
            ),
            "max_state_error_vs_capture_reference": max(state_errors) if state_errors else 0.0,
            "contact_observed": any(contacts),
            "grasp_observed": any(grasps),
            "failure_stage": failure_stage(bool(success_indices), contacts, grasps),
            "controller_use_delta": bool(controller.use_delta),
            "controller_snapshot_after_replay": controller_snapshot(controller),
            "outer_action_dtypes": sorted(set(outer_dtypes)),
            "telemetry": telemetry,
            "wall_clock_seconds": time.monotonic() - started,
        }
    )


def first_exceeding(values: np.ndarray, atol: float) -> int | None:
    indices = np.flatnonzero(np.asarray(values) > atol)
    return int(indices[0]) if len(indices) else None


def compare_to_a(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reference = results["A_capture_raw"]
    comparison: dict[str, Any] = {}
    ref_rows = reference["telemetry"]
    for condition, result in results.items():
        rows = result["telemetry"]
        length = min(len(ref_rows), len(rows))
        goal_pos_errors = []
        goal_ori_errors = []
        eef_errors = []
        bowl_errors = []
        state_errors = []
        action_errors = []
        for index in range(length):
            ref = ref_rows[index]
            row = rows[index]
            action_errors.append(
                float(np.max(np.abs(np.asarray(row["bottom_action"]) - np.asarray(ref["bottom_action"]))))
            )
            goal_pos_errors.append(
                float(np.max(np.abs(np.asarray(row["controller_goal_pos"]) - np.asarray(ref["controller_goal_pos"]))))
            )
            goal_ori_errors.append(
                orientation_angle(np.asarray(row["controller_goal_ori"]), np.asarray(ref["controller_goal_ori"]))
            )
            eef_errors.append(
                float(np.max(np.abs(np.asarray(row["eef_position"]) - np.asarray(ref["eef_position"]))))
            )
            bowl_errors.append(
                float(np.max(np.abs(np.asarray(row["bowl_position"]) - np.asarray(ref["bowl_position"]))))
            )
            state_errors.append(float(row["state_max_abs_error_vs_capture_reference"]))
        comparison[condition] = {
            "same_as_reference": condition == "A_capture_raw",
            "aligned_steps_compared": length,
            "first_bottom_action_difference": first_exceeding(action_errors, POSE_DIVERGENCE_ATOL),
            "first_controller_goal_position_difference": first_exceeding(goal_pos_errors, POSE_DIVERGENCE_ATOL),
            "first_controller_goal_orientation_difference": first_exceeding(goal_ori_errors, POSE_DIVERGENCE_ATOL),
            "first_eef_position_difference": first_exceeding(eef_errors, POSE_DIVERGENCE_ATOL),
            "first_bowl_position_difference": first_exceeding(bowl_errors, POSE_DIVERGENCE_ATOL),
            "first_full_state_difference": first_exceeding(state_errors, STATE_DIVERGENCE_ATOL),
            "max_bottom_action_difference": max(action_errors) if action_errors else 0.0,
            "max_controller_goal_position_difference": max(goal_pos_errors) if goal_pos_errors else 0.0,
            "max_controller_goal_orientation_difference_rad": max(goal_ori_errors) if goal_ori_errors else 0.0,
            "max_eef_position_difference": max(eef_errors) if eef_errors else 0.0,
            "max_bowl_position_difference": max(bowl_errors) if bowl_errors else 0.0,
            "max_full_state_difference": max(state_errors) if state_errors else 0.0,
            "contact_index": result["first_contact_action_index"],
            "success_index": result["first_success_action_index"],
            "failure_stage": result["failure_stage"],
        }
    return comparison


def select_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    selected = [
        record
        for record in manifest["records"]
        if record.get("accepted") and (int(record["layout"]), str(record["demo_key"])) in SELECTED
    ]
    found = {(int(record["layout"]), str(record["demo_key"])) for record in selected}
    if found != SELECTED:
        raise ValueError(f"selected diagnostic records missing: {sorted(SELECTED - found)}")
    return sorted(selected, key=lambda item: (int(item["layout"]), str(item["demo_key"])) )


def make_pointbridge_delta_environment(pb_suite: Any, upstream: Path, task_name: str) -> Any:
    pb_suite.init_models = lambda: _PrecomputedLanguageEncoder()
    bddl_dir = (
        upstream / "third_party" / "mimiclabs" / "mimiclabs" / "mimiclabs" / "task_suites" / "new_task_suite"
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
        raise RuntimeError(f"expected one Point Bridge environment, got {len(envs)}")
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
    import point_bridge.suite.mimiclabs as pb_suite
    from mimiclabs.mimiclabs.envs.problems import MimicLabs_Lab1_Tabletop_Manipulation  # noqa: F401

    if str(getattr(robosuite, "__version__", "unknown")) != REQUIRED_ROBOSUITE:
        raise RuntimeError(f"expected robosuite {REQUIRED_ROBOSUITE}")
    if str(getattr(mujoco, "__version__", "unknown")) != REQUIRED_MUJOCO:
        raise RuntimeError(f"expected MuJoCo {REQUIRED_MUJOCO}")

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = select_records(manifest)
    all_actions: list[np.ndarray] = []
    loaded: dict[tuple[int, str], tuple[dict[str, np.ndarray], str]] = {}
    for record in records:
        arrays, model_xml = load_record(record)
        loaded[(int(record["layout"]), str(record["demo_key"]))] = (arrays, model_xml)
        all_actions.append(np.asarray(arrays["issued_actions"], dtype=np.float64))
    all_manifest_records = [record for record in manifest["records"] if record.get("accepted")]
    all_action_arrays = []
    for record in all_manifest_records:
        artifact = ROOT / record["artifact"]
        with np.load(artifact, allow_pickle=False) as bundle:
            all_action_arrays.append(np.asarray(bundle["issued_actions"], dtype=np.float64))
    stats = shared_action_stats(all_action_arrays)

    records_out: list[dict[str, Any]] = []
    descriptors: dict[str, Any] = {}
    for record in records:
        layout = int(record["layout"])
        task_name = f"bowl_on_plate_{layout}"
        hdf5_path = upstream / "data" / "mimicgen_data" / "bowl_on_plate" / task_name / "demo" / "demo.hdf5"
        bddl_path = upstream / "third_party" / "mimiclabs" / "mimiclabs" / "mimiclabs" / "task_suites" / "new_task_suite" / f"{task_name}.bddl"
        with h5py.File(hdf5_path, "r") as handle:
            env_spec = json.loads(str(handle["data"].attrs["env_args"]))
        capture_env = make_env(robosuite, env_spec, bddl_path, control_delta=True)
        pb_env = make_pointbridge_delta_environment(pb_suite, upstream, task_name)
        capture_env.reset()
        pb_env.reset()
        descriptors.setdefault("capture_env_args", env_descriptor(capture_env, "capture_env_args", env_spec))
        descriptors.setdefault("pointbridge_delta_pose", env_descriptor(pb_env, "pointbridge_delta_pose"))
        arrays, model_xml = loaded[(layout, str(record["demo_key"]))]
        raw_actions = np.asarray(arrays["issued_actions"], dtype=np.float64)
        normalized, deployment, roundtrip_error = normalize_delta_actions(raw_actions, stats)
        raw_float32 = raw_actions.astype(np.float32)
        conditions = (
            ("A_capture_raw", capture_env, "capture_env_args", raw_actions, raw_actions, None),
            ("B_capture_normalized", capture_env, "capture_env_args", deployment, deployment, roundtrip_error),
            ("C_pointbridge_raw", pb_env, "pointbridge_delta_pose", raw_actions, raw_actions, None),
            ("D_pointbridge_normalized", pb_env, "pointbridge_delta_pose", deployment, deployment, roundtrip_error),
            ("E_capture_float32", capture_env, "capture_env_args", raw_float32, raw_float32, None),
            ("F_pointbridge_float32", pb_env, "pointbridge_delta_pose", raw_float32, raw_float32, None),
        )
        per_condition: dict[str, dict[str, Any]] = {}
        for condition, env, source, actions, input_actions, normalization_error in conditions:
            result = replay_condition(
                env,
                source,
                condition,
                record,
                arrays,
                model_xml,
                actions,
                input_actions,
                normalization_error,
            )
            per_condition[condition] = result
            print(
                f"layout={layout} demo={record['demo_key']} condition={condition} "
                f"bottom={result['bottom_step_call_count']} success={result['success']} "
                f"first_success={result['first_success_action_index']}",
                flush=True,
            )
        records_out.append(
            jsonable(
                {
                    "layout": layout,
                    "demo_key": record["demo_key"],
                    "artifact": record["artifact"],
                    "source_artifact_sha256": record["artifact_sha256"],
                    "raw_actions_sha256": raw_array_sha256(raw_actions),
                    "normalized_actions_sha256": raw_array_sha256(normalized),
                    "deployment_actions_sha256": raw_array_sha256(deployment),
                    "normalization_roundtrip_max_abs_error": roundtrip_error,
                    "conditions": per_condition,
                    "comparison_to_a": compare_to_a(per_condition),
                }
            )
        )
        capture_env.close()
        pb_env.close()

    result = {
        "stage": "V1-R.2J-F",
        "status": "completed",
        "diagnostic": "delta_pose_execution_path_2x3",
        "formal_threshold": {
            "checked_per_formal_gate": 20,
            "required_passed_per_formal_gate": 20,
        },
        "diagnostic_replays": 24,
        "design": {
            "selected_records": sorted([f"layout_{layout}_{demo}" for layout, demo in SELECTED]),
            "conditions": {
                "A_capture_raw": "HDF5 env_args capture environment + original issued_actions",
                "B_capture_normalized": "HDF5 env_args capture environment + shared minmax float32 inverse",
                "C_pointbridge_raw": "Point Bridge delta_pose entry + original issued_actions",
                "D_pointbridge_normalized": "Point Bridge delta_pose entry + shared minmax float32 inverse",
                "E_capture_float32": "HDF5 env_args capture environment + original issued_actions cast directly to float32",
                "F_pointbridge_float32": "Point Bridge delta_pose entry + original issued_actions cast directly to float32",
            },
            "replays": 24,
            "factorial": {
                "environment": ["capture_env_args", "pointbridge_delta_pose"],
                "action_values": [
                    "raw_float64",
                    "raw_float32",
                    "shared_minmax_float32_inverse_float64",
                ],
                "same_selected_records": True,
            },
            "intermediate_state_restore": False,
            "policy_training_or_inference": False,
            "formal_20_demo_gate_changed": False,
        },
        "runtime": {
            "robosuite": REQUIRED_ROBOSUITE,
            "mujoco": REQUIRED_MUJOCO,
            "control_freq_hz": CONTROL_FREQ_HZ,
            "action_mode": "delta_pose",
            "normalization_source": "all_20_v1r_2i_issued_actions",
            "normalization_dtype": "float32",
            "deployment_dtype": "float64",
        },
        "normalization": {
            "shared_min": stats["min"],
            "shared_max": stats["max"],
            "shared_scale": stats["scale"],
        },
        "environment_descriptors": descriptors,
        "source": {
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": file_sha256(manifest_path),
        },
        "records": records_out,
        "decision": "diagnostic_only_no_training_contract_selected",
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(jsonable(result), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(jsonable({key: value for key, value in result.items() if key != "records"}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
