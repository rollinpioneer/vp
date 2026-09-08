#!/usr/bin/env python3
"""Capture V1-R.2I demos by executing expert commands continuously.

This entry point deliberately does not apply the Point Bridge PKL-generation
patches.  It reads the official HDF5 action stream as a candidate command
source, restores a candidate's initial simulator state once, then obtains all
later states from actual ``env.step`` transitions.  A candidate is accepted
only when the formal task success predicate becomes true during that sequence.

The saved bundle keeps three meanings separate:

* ``issued_actions``: the 7-D commands actually sent to the delta OSC;
* ``native_eef_states`` / ``native_gripper_states``: the observation-aligned
  arrays needed to reproduce the original Point Bridge label path;
* ``post_step_eef_states`` and ``states_after``: measurements after each real
  transition, plus the absolute controller target used for that transition.

No state is loaded between two actions in a candidate trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
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
REQUIRED_ROBOSUITE = "1.4.1"
REQUIRED_MUJOCO = "3.3.5"
CONTROL_FREQ_HZ = 20

sys.path.insert(0, str(ROOT / "src"))


def raw_array_sha256(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def text_sha256(value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def controller_for(env: Any) -> Any:
    controller = env.robots[0].controller
    if isinstance(controller, dict):
        if len(controller) != 1:
            raise ValueError("sequential capture expects exactly one arm controller")
        controller = next(iter(controller.values()))
    return controller


def synchronize_runtime_state(env: Any) -> None:
    """Clear stateful controller/gripper caches after the one allowed reset."""

    gripper = env.robots[0].gripper
    gripper.current_action = np.zeros(gripper.dof, dtype=np.float64)
    controller = controller_for(env)
    controller.update(force=True)
    controller.update_initial_joints(controller.joint_pos)


def model_xml_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def robot_base_transform(env: Any, t_robot_base: np.ndarray) -> np.ndarray:
    base = np.eye(4, dtype=np.float64)
    base[:3, :3] = np.asarray(env.sim.data.get_body_xmat("robot0_base"), dtype=np.float64)
    base[:3, 3] = np.asarray(env.sim.data.get_body_xpos("robot0_base"), dtype=np.float64)
    return base @ np.asarray(t_robot_base, dtype=np.float64)


def eef_pose_pointbridge(env: Any, robot_base: np.ndarray, t_gripper: np.ndarray) -> np.ndarray:
    """Return [position, xyzw quaternion] in the Point Bridge base frame."""

    from scipy.spatial.transform import Rotation as rotation

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(env.sim.data.get_body_xmat("gripper0_eef"), dtype=np.float64)
    pose[:3, 3] = np.asarray(env.sim.data.get_body_xpos("gripper0_eef"), dtype=np.float64)
    pose = np.linalg.inv(robot_base) @ pose @ np.asarray(t_gripper, dtype=np.float64)
    return np.concatenate([pose[:3, 3], rotation.from_matrix(pose[:3, :3]).as_quat()])


def controller_target_world(env: Any) -> np.ndarray:
    controller = controller_for(env)
    target = np.eye(4, dtype=np.float64)
    target[:3, :3] = np.asarray(controller.goal_ori, dtype=np.float64)
    target[:3, 3] = np.asarray(controller.goal_pos, dtype=np.float64)
    return target


def controller_target_pointbridge(
    target_world: np.ndarray, robot_base: np.ndarray, t_gripper: np.ndarray
) -> np.ndarray:
    target = np.linalg.inv(robot_base) @ np.asarray(target_world) @ np.asarray(t_gripper)
    return target


def phase_flags(env: Any) -> tuple[bool, bool, bool]:
    bowl = env.objects_dict["bowl"]
    gripper = env.robots[0].gripper
    contact = bool(env.check_contact(gripper, bowl))
    grasped = bool(env._check_grasp(gripper, bowl))
    success = bool(env._check_success())
    return contact, grasped, success


def make_env(
    suite: Any,
    env_spec: dict[str, Any],
    bddl_path: Path,
    control_delta: bool | None = None,
) -> Any:
    env_kwargs = dict(env_spec["env_kwargs"])
    env_kwargs.pop("env_lang", None)
    env_kwargs.update(
        {
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "use_camera_obs": False,
            "camera_depths": False,
            "bddl_file_name": str(bddl_path),
        }
    )
    controller = dict(env_kwargs["controller_configs"])
    if control_delta is not None:
        controller["control_delta"] = control_delta
    env_kwargs["controller_configs"] = controller
    env_kwargs["control_freq"] = CONTROL_FREQ_HZ
    return suite.make(env_name=env_spec["env_name"], **env_kwargs)


def candidate_indices(
    handle: Any,
    demo_start: int,
    demo_stop: int | None,
    candidate_manifests: list[Path] | None,
    layout: int | None = None,
) -> list[int]:
    demos = sorted(handle["data"].keys(), key=lambda key: int(key.rsplit("_", 1)[1]))
    selected = list(range(demo_start, demo_stop if demo_stop is not None else len(demos)))
    if not candidate_manifests:
        return selected
    manifest_paths: list[Path] = []
    for candidate_manifest in candidate_manifests:
        if candidate_manifest.is_dir():
            manifest_paths.extend(sorted(candidate_manifest.glob("*.json")))
        else:
            manifest_paths.append(candidate_manifest)
    accepted: set[int] = set()
    for manifest_path in manifest_paths:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_layouts = {
            int(record.get("shard", record.get("layout")))
            for record in payload.get("records", [])
            if record.get("shard", record.get("layout")) is not None
        }
        if layout is not None and manifest_layouts and layout not in manifest_layouts:
            continue
        accepted.update(
            int(str(record["demo_key"]).rsplit("_", 1)[1])
            for record in payload.get("records", [])
            if (
                layout is None
                or record.get("shard", record.get("layout")) is None
                or int(record.get("shard", record.get("layout"))) == layout
            )
            if record.get("saved_final_success")
            or record.get("saved_final_action_success")
        )
    return [index for index in selected if index in accepted]


def _record_array_hashes(path: Path) -> dict[str, str]:
    with np.load(path, allow_pickle=False) as bundle:
        return {
            key: raw_array_sha256(bundle[key])
            for key in sorted(bundle.files)
            if bundle[key].dtype.kind not in {"U", "S"}
        }


def capture_candidate(
    env: Any,
    demo: Any,
    layout: int,
    demo_key: str,
    output_dir: Path,
    t_robot_base: np.ndarray,
    t_gripper: np.ndarray,
    migrate_saved_model_xml: Any,
) -> dict[str, Any]:
    started = time.monotonic()
    source_states = np.asarray(demo["states"], dtype=np.float64)
    source_actions = np.asarray(demo["actions"], dtype=np.float64)
    if source_states.ndim != 2 or not len(source_states):
        raise ValueError(f"expected non-empty source states, got {source_states.shape}")
    if source_actions.ndim != 2 or source_actions.shape[1] != 7:
        raise ValueError(f"expected source actions shaped (T, 7), got {source_actions.shape}")
    original_xml = model_xml_text(demo.attrs["model_file"])
    migrated_xml = migrate_saved_model_xml(original_xml)
    record: dict[str, Any] = {
        "layout": layout,
        "demo_key": demo_key,
        "source_steps": int(len(source_actions)),
        "source_hdf5_model_xml_sha256": text_sha256(original_xml),
        "runtime_model_xml_sha256": text_sha256(migrated_xml),
        "initial_state_sha256": "",
        "restored_initial_state_sha256": "",
        "restored_initial_state_match": False,
        "actions_issued": 0,
        "task_success": False,
        "first_success_action_index": None,
        "stable_grasp_observed": False,
        "contact_observed": False,
        "closed_without_bowl_observed": False,
        "midtrajectory_state_restore_count": 0,
        "accepted": False,
        "artifact": None,
        "exception": None,
        "failure_stage": "not_run",
    }
    try:
        # This is the only state restore for a candidate.  All subsequent
        # states are produced by env.step(source_actions[index]).
        env.reset_to({"states": source_states[0], "model": migrated_xml})
        synchronize_runtime_state(env)
        actual_initial = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
        record["initial_state_sha256"] = raw_array_sha256(source_states[0])
        record["restored_initial_state_sha256"] = raw_array_sha256(actual_initial)
        record["restored_initial_state_match"] = bool(
            actual_initial.shape == source_states[0].shape
            and np.array_equal(actual_initial, source_states[0])
        )
        if not record["restored_initial_state_match"]:
            raise RuntimeError("initial simulator state mismatch after candidate reset")

        robot_base = robot_base_transform(env, t_robot_base)
        states_before: list[np.ndarray] = []
        states_after: list[np.ndarray] = []
        native_eef_states: list[np.ndarray] = []
        post_step_eef_states: list[np.ndarray] = []
        native_gripper_states: list[float] = []
        issued_actions: list[np.ndarray] = []
        controller_targets_world: list[np.ndarray] = []
        controller_targets_pointbridge: list[np.ndarray] = []
        contact_flags: list[bool] = []
        grasp_flags: list[bool] = []
        success_flags: list[bool] = []

        for action_index, source_action in enumerate(source_actions):
            # Observation aligned with the command.  This mirrors the native
            # generator's eef/gripper arrays without loading a future state.
            states_before.append(np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy())
            native_eef_states.append(eef_pose_pointbridge(env, robot_base, t_gripper))
            native_gripper_states.append(float(source_action[-1]))
            command = np.asarray(source_action, dtype=np.float64).copy()
            issued_actions.append(command)

            env.step(command)
            after_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy()
            after_eef = eef_pose_pointbridge(env, robot_base, t_gripper)
            target_world = controller_target_world(env)
            target_pointbridge = controller_target_pointbridge(
                target_world, robot_base, t_gripper
            )
            contact, grasped, success = phase_flags(env)
            states_after.append(after_state)
            post_step_eef_states.append(after_eef)
            controller_targets_world.append(target_world)
            controller_targets_pointbridge.append(target_pointbridge)
            contact_flags.append(contact)
            grasp_flags.append(grasped)
            success_flags.append(success)

            if contact:
                record["contact_observed"] = True
            if grasped and not record["stable_grasp_observed"]:
                # Diagnostic only.  It is intentionally not a success filter.
                run = 0
                for value in reversed(grasp_flags):
                    if not value:
                        break
                    run += 1
                record["stable_grasp_observed"] = run >= 5
            if command[-1] > 0 and not contact and not grasped:
                record["closed_without_bowl_observed"] = True
            if success:
                record["task_success"] = True
                record["first_success_action_index"] = action_index
                break

        record["actions_issued"] = len(issued_actions)
        if not record["task_success"]:
            record["failure_stage"] = "no_task_success"
            return record

        artifact_stem = f"layout_{layout}_demo_{int(demo_key.rsplit('_', 1)[1]):03d}"
        artifact_path = output_dir / f"{artifact_stem}.npz"
        model_path = output_dir / f"{artifact_stem}.xml"
        model_path.write_text(migrated_xml, encoding="utf-8")
        np.savez_compressed(
            artifact_path,
            initial_state=actual_initial,
            robot_base=robot_base,
            states_before=np.asarray(states_before),
            states_after=np.asarray(states_after),
            issued_actions=np.asarray(issued_actions),
            native_eef_states=np.asarray(native_eef_states),
            native_gripper_states=np.asarray(native_gripper_states),
            post_step_eef_states=np.asarray(post_step_eef_states),
            controller_targets_world=np.asarray(controller_targets_world),
            controller_targets_pointbridge=np.asarray(controller_targets_pointbridge),
            contact_flags=np.asarray(contact_flags, dtype=np.bool_),
            grasp_flags=np.asarray(grasp_flags, dtype=np.bool_),
            success_flags=np.asarray(success_flags, dtype=np.bool_),
        )
        record.update(
            {
                "artifact": str(artifact_path.relative_to(ROOT)),
                "model_xml": str(model_path.relative_to(ROOT)),
                "artifact_sha256": file_sha256(artifact_path),
                "model_xml_sha256": file_sha256(model_path),
                "artifact_array_sha256": _record_array_hashes(artifact_path),
                "states_after_sha256": raw_array_sha256(np.asarray(states_after)),
                "issued_actions_sha256": raw_array_sha256(np.asarray(issued_actions)),
                "native_eef_states_sha256": raw_array_sha256(np.asarray(native_eef_states)),
                "controller_targets_world_sha256": raw_array_sha256(
                    np.asarray(controller_targets_world)
                ),
                "failure_stage": "success",
                "accepted": True,
            }
        )
    except Exception as exc:
        record["exception"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-3000:]}"
        record["failure_stage"] = "exception"
    finally:
        record["wall_clock_seconds"] = time.monotonic() - started
    return jsonable(record)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--layouts", default="1,2,3,4")
    parser.add_argument("--target-per-layout", type=int, default=5)
    parser.add_argument("--demo-start", type=int, default=0)
    parser.add_argument("--demo-stop", type=int)
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        action="append",
        help="saved-state audit JSON (repeatable; a directory reads all JSON files)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "v1r" / "sequential_success_demos_2i",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=ROOT / "outputs" / "v1r" / "sequential_success_demos_2i" / "manifest.json",
    )
    args = parser.parse_args()
    layouts = [int(item) for item in args.layouts.split(",") if item]
    if not layouts or any(layout not in {1, 2, 3, 4} for layout in layouts):
        parser.error("--layouts must contain only 1,2,3,4")
    if args.target_per_layout <= 0:
        parser.error("--target-per-layout must be positive")

    upstream = args.upstream.resolve()
    add_upstream_paths(upstream)
    os.environ.setdefault("MUJOCO_GL", "egl")

    import h5py
    import mujoco
    import robosuite as suite
    from mimiclabs.mimiclabs.envs.problems import (  # noqa: F401
        MimicLabs_Lab1_Tabletop_Manipulation,
    )
    from point_bridge.robot_utils.mimiclabs.utils import T_gripper, T_robot_base
    from vico_point.envs.mimiclabs_compat import migrate_saved_model_xml

    if str(getattr(suite, "__version__", "unknown")) != REQUIRED_ROBOSUITE:
        raise RuntimeError(f"expected robosuite {REQUIRED_ROBOSUITE}")
    if str(getattr(mujoco, "__version__", "unknown")) != REQUIRED_MUJOCO:
        raise RuntimeError(f"expected MuJoCo {REQUIRED_MUJOCO}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_output.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    accepted_by_layout = {str(layout): 0 for layout in layouts}

    for layout in layouts:
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
            source_hdf5_sha256 = file_sha256(hdf5_path)
            indices = candidate_indices(
                handle,
                args.demo_start,
                args.demo_stop,
                args.candidate_manifest,
                layout=layout,
            )
            env = make_env(suite, env_spec, bddl_path, control_delta=True)
            env.reset()
            if not bool(controller_for(env).use_delta):
                raise RuntimeError("sequential capture environment is not delta OSC")
            try:
                for demo_index in indices:
                    if accepted_by_layout[str(layout)] >= args.target_per_layout:
                        break
                    demo_key = f"demo_{demo_index}"
                    if demo_key not in handle["data"]:
                        continue
                    record = capture_candidate(
                        env,
                        handle["data"][demo_key],
                        layout,
                        demo_key,
                        output_dir,
                        np.asarray(T_robot_base, dtype=np.float64),
                        np.asarray(T_gripper, dtype=np.float64),
                        migrate_saved_model_xml,
                    )
                    record.update(
                        {
                            "task_name": task_name,
                            "source_hdf5": str(hdf5_path.relative_to(ROOT)),
                            "source_hdf5_sha256": source_hdf5_sha256,
                            "runtime": {
                                "robosuite": REQUIRED_ROBOSUITE,
                                "mujoco": REQUIRED_MUJOCO,
                                "control_freq_hz": CONTROL_FREQ_HZ,
                                "controller": "OSC_POSE",
                                "control_delta": True,
                            },
                            "action_source": "official_hdf5_delta_actions",
                            "middle_state_restore_count": 0,
                        }
                    )
                    records.append(jsonable(record))
                    if record.get("accepted"):
                        accepted_by_layout[str(layout)] += 1
                    print(
                        f"layout={layout} candidate={demo_key} accepted={record.get('accepted')} "
                        f"count={accepted_by_layout[str(layout)]}/{args.target_per_layout}",
                        flush=True,
                    )
            finally:
                env.close()

    per_layout = {
        str(layout): {
            "target": args.target_per_layout,
            "accepted": accepted_by_layout[str(layout)],
            "passed": accepted_by_layout[str(layout)] == args.target_per_layout,
        }
        for layout in layouts
    }
    result = {
        "stage": "V1-R.2I.1",
        "status": "passed" if all(item["passed"] for item in per_layout.values()) else "incomplete",
        "formal_success_criterion": "goal_achieved_during_continuous_env_step_sequence",
        "stable_grasp_is_diagnostic_only": True,
        "middle_state_restore_count": 0,
        "runtime": {
            "robosuite": REQUIRED_ROBOSUITE,
            "mujoco": REQUIRED_MUJOCO,
            "control_freq_hz": CONTROL_FREQ_HZ,
            "controller": "OSC_POSE",
            "control_delta": True,
        },
        "target_per_layout": args.target_per_layout,
        "per_layout": per_layout,
        "records": records,
        "next_stage": (
            "v1r_2i_2_verify_actual_command_replay_then_s0_s1_label_replay"
            if all(item["passed"] for item in per_layout.values())
            else "obtain_more_continuously_successful_demos_without_retraining"
        ),
    }
    manifest_path.write_text(json.dumps(jsonable(result), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
