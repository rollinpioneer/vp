#!/usr/bin/env python3
"""Capture V1-R.2K demonstrations with quantization before each transition.

This script is deliberately separate from the completed V1-R.2I collector.
Every recorded transition has the meaning
``observation_before --float32_label--> state_after``.  Candidate actions are
read as float64 from the audited HDF5 source, encoded once, decoded once, and
then passed directly to the delta OSC environment.  No state is restored
between actions.
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
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from capture_sequential_success_demos import (  # noqa: E402
    add_upstream_paths,
    controller_for,
    eef_pose_pointbridge,
    file_sha256,
    make_env,
    model_xml_text,
    phase_flags,
    raw_array_sha256,
    robot_base_transform,
    synchronize_runtime_state,
)
from numeric_contract import (  # noqa: E402
    CONTRACT_ID,
    array_sha256,
    decode_sequence,
    encode_sequence,
)


REQUIRED_ROBOSUITE = "1.4.1"
REQUIRED_MUJOCO = "3.3.5"
CONTROL_FREQ_HZ = 20
DEFAULT_UPSTREAM = ROOT / "third_party" / "pointbridge"
DEFAULT_OUTPUT = ROOT / "outputs" / "v1r" / "sequential_success_demos_2k_quantized_at_source"
DEFAULT_MANIFEST = DEFAULT_OUTPUT / "manifest.json"


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


def display_path(path: Path) -> str:
    """Use a repo-relative path when possible, otherwise preserve the source path."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def load_candidate_indices(
    handle: Any, layout: int, manifests: list[Path] | None
) -> tuple[list[int], str]:
    all_indices = sorted(
        (int(key.rsplit("_", 1)[1]) for key in handle["data"].keys()),
    )
    if not manifests:
        return all_indices, "all_hdf5_demo_keys_numeric_ascending"
    selected: set[int] = set()
    used: list[str] = []
    for manifest_path in manifests:
        if not manifest_path.exists():
            continue
        used.append(str(manifest_path))
        if manifest_path.suffix.lower() == ".csv":
            with manifest_path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    if int(row.get("layout", row.get("shard", -1))) != layout:
                        continue
                    selected.add(int(row["demo_index"]))
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in payload.get("records", []):
            row_layout = row.get("layout", row.get("shard"))
            if row_layout is not None and int(row_layout) != layout:
                continue
            if row.get("saved_final_success") or row.get("saved_final_action_success"):
                selected.add(int(str(row["demo_key"]).rsplit("_", 1)[1]))
    if not selected:
        return all_indices, "missing_or_empty_audit_manifest_fallback_all_hdf5_keys"
    return [index for index in all_indices if index in selected], "+".join(used)


def _save_attempt(
    output_dir: Path,
    stem: str,
    model_xml: str,
    arrays: dict[str, Any],
) -> tuple[Path, Path]:
    artifact = output_dir / f"{stem}.npz"
    model_path = output_dir / f"{stem}.xml"
    model_path.write_text(model_xml, encoding="utf-8")
    np.savez_compressed(artifact, **arrays)
    return artifact, model_path


def capture_candidate(
    env: Any,
    demo: Any,
    layout: int,
    demo_key: str,
    output_dir: Path,
    t_robot_base: np.ndarray,
    t_gripper: np.ndarray,
    migrate_saved_model_xml: Any,
    attempt_ordinal: int,
    attempt_kind: str,
) -> dict[str, Any]:
    started = time.monotonic()
    source_states = np.asarray(demo["states"], dtype=np.float64)
    raw_actions = np.asarray(demo["actions"], dtype=np.float64)
    if source_states.ndim != 2 or not len(source_states):
        raise ValueError(f"expected non-empty source states, got {source_states.shape}")
    if raw_actions.ndim != 2 or raw_actions.shape[1] != 7:
        raise ValueError(f"expected source actions shaped (T, 7), got {raw_actions.shape}")
    original_xml = model_xml_text(demo.attrs["model_file"])
    migrated_xml = migrate_saved_model_xml(original_xml)
    labels = encode_sequence(raw_actions)
    commands = decode_sequence(labels)
    record: dict[str, Any] = {
        "layout": layout,
        "demo_key": demo_key,
        "candidate_order": "numeric_demo_index_ascending_within_layout",
        "attempt_ordinal": attempt_ordinal,
        "attempt_kind": attempt_kind,
        "counts_toward_target": attempt_kind == "formal_candidate",
        "source_steps": int(len(raw_actions)),
        "raw_action_sha256": array_sha256(raw_actions),
        "float32_label_sha256": array_sha256(labels),
        "decoded_action_sha256": array_sha256(commands),
        "numeric_contract": CONTRACT_ID,
        "initial_state_sha256": "",
        "restored_initial_state_match": False,
        "actions_issued": 0,
        "task_success": False,
        "first_contact_action_index": None,
        "first_contact_time_seconds": None,
        "first_success_action_index": None,
        "first_success_time_seconds": None,
        "contact_observed": False,
        "grasp_observed": False,
        "closed_without_bowl_observed": False,
        "midtrajectory_state_restore_count": 0,
        "accepted": False,
        "failure_stage": "not_run",
        "artifact": None,
        "model_xml": None,
        "exception": None,
    }
    arrays: dict[str, Any] = {}
    try:
        env.reset_to({"states": source_states[0], "model": migrated_xml})
        synchronize_runtime_state(env)
        actual_initial = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
        record["initial_state_sha256"] = raw_array_sha256(actual_initial)
        record["restored_initial_state_match"] = bool(
            np.array_equal(actual_initial, source_states[0])
        )
        if not record["restored_initial_state_match"]:
            raise RuntimeError("initial simulator state mismatch after candidate reset")
        robot_base = robot_base_transform(env, t_robot_base)
        states_before: list[np.ndarray] = []
        states_after: list[np.ndarray] = []
        eef_before: list[np.ndarray] = []
        eef_after: list[np.ndarray] = []
        contacts: list[bool] = []
        grasps: list[bool] = []
        successes: list[bool] = []
        elapsed: list[float] = []
        for index, (label, command) in enumerate(zip(labels, commands)):
            states_before.append(np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy())
            eef_before.append(eef_pose_pointbridge(env, robot_base, t_gripper))
            started_step = time.monotonic()
            env.step(command)
            elapsed.append(time.monotonic() - started_step)
            states_after.append(np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy())
            eef_after.append(eef_pose_pointbridge(env, robot_base, t_gripper))
            contact, grasp, success = phase_flags(env)
            contacts.append(contact)
            grasps.append(grasp)
            successes.append(success)
            if contact and record["first_contact_action_index"] is None:
                record["first_contact_action_index"] = index
                record["first_contact_time_seconds"] = index / CONTROL_FREQ_HZ
            if contact:
                record["contact_observed"] = True
            if grasp:
                record["grasp_observed"] = True
            if command[-1] > 0 and not contact and not grasp:
                record["closed_without_bowl_observed"] = True
            if success:
                record["task_success"] = True
                record["first_success_action_index"] = index
                record["first_success_time_seconds"] = index / CONTROL_FREQ_HZ
                break
        used = len(states_after)
        record["actions_issued"] = used
        record["source_action_exhausted"] = used == len(raw_actions) and not record["task_success"]
        if record["task_success"]:
            record["failure_stage"] = "success"
            record["accepted"] = True
        elif record["contact_observed"] and not record["grasp_observed"]:
            record["failure_stage"] = "contact_without_grasp"
        elif not record["contact_observed"]:
            record["failure_stage"] = "no_contact"
        else:
            record["failure_stage"] = "other_after_grasp_without_task_success"
        arrays = {
            "initial_state": actual_initial,
            "robot_base": robot_base,
            "states_before": np.asarray(states_before),
            "states_after": np.asarray(states_after),
            "raw_actions_float64": raw_actions[:used],
            "float32_labels": labels[:used],
            "controller_commands_float64": commands[:used],
            "eef_states_before": np.asarray(eef_before),
            "eef_states_after": np.asarray(eef_after),
            "contact_flags": np.asarray(contacts, dtype=np.bool_),
            "grasp_flags": np.asarray(grasps, dtype=np.bool_),
            "success_flags": np.asarray(successes, dtype=np.bool_),
            "step_wall_clock_seconds": np.asarray(elapsed, dtype=np.float64),
        }
        stem = f"layout_{layout}_demo_{int(demo_key.rsplit('_', 1)[1]):03d}_attempt_{attempt_ordinal:03d}"
        artifact, model_path = _save_attempt(output_dir, stem, migrated_xml, arrays)
        record.update(
            {
                "artifact": str(artifact.relative_to(ROOT)),
                "model_xml": str(model_path.relative_to(ROOT)),
                "artifact_sha256": file_sha256(artifact),
                "model_xml_sha256": file_sha256(model_path),
                "array_sha256": {key: array_sha256(value) for key, value in arrays.items()},
                "executed_command_sequence_exact": bool(
                    np.array_equal(arrays["controller_commands_float64"], labels[:used].astype(np.float64))
                ),
            }
        )
    except Exception as exc:
        record["exception"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-3000:]}"
        record["failure_stage"] = "exception"
    record["wall_clock_seconds"] = time.monotonic() - started
    return jsonable(record)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--layouts", default="1,2,3,4")
    parser.add_argument("--target-per-layout", type=int, default=5)
    parser.add_argument("--candidate-manifest", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--diagnostic-demo",
        action="append",
        default=["layout_1:demo_18"],
        help="required quantized attempt kept outside acceptance quota (LAYOUT:demo_KEY)",
    )
    args = parser.parse_args()
    layouts = [int(item) for item in args.layouts.split(",") if item]
    if set(layouts) != {1, 2, 3, 4}:
        parser.error("--layouts must contain exactly 1,2,3,4")
    upstream = args.upstream.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_output.resolve().parent.mkdir(parents=True, exist_ok=True)
    add_upstream_paths(upstream)
    os.environ.setdefault("MUJOCO_GL", "egl")
    import h5py
    import mujoco
    import robosuite as suite
    from mimiclabs.mimiclabs.envs.problems import MimicLabs_Lab1_Tabletop_Manipulation  # noqa: F401
    from point_bridge.robot_utils.mimiclabs.utils import T_gripper, T_robot_base
    from vico_point.envs.mimiclabs_compat import migrate_saved_model_xml

    if str(getattr(suite, "__version__", "unknown")) != REQUIRED_ROBOSUITE:
        raise RuntimeError(f"expected robosuite {REQUIRED_ROBOSUITE}")
    if str(getattr(mujoco, "__version__", "unknown")) != REQUIRED_MUJOCO:
        raise RuntimeError(f"expected MuJoCo {REQUIRED_MUJOCO}")
    diagnostic_keys = {
        (int(item.split(":", 1)[0].split("_", 1)[1]), item.split(":", 1)[1])
        for item in args.diagnostic_demo
    }
    records: list[dict[str, Any]] = []
    accepted_by_layout = {layout: 0 for layout in layouts}
    attempt_ordinal = 0
    for layout in layouts:
        task_name = f"bowl_on_plate_{layout}"
        hdf5_path = upstream / "data" / "mimicgen_data" / "bowl_on_plate" / task_name / "demo" / "demo.hdf5"
        bddl_path = upstream / "third_party" / "mimiclabs" / "mimiclabs" / "mimiclabs" / "task_suites" / "new_task_suite" / f"{task_name}.bddl"
        with h5py.File(hdf5_path, "r") as handle:
            env_spec = json.loads(str(handle["data"].attrs["env_args"]))
            source_hdf5_sha256 = file_sha256(hdf5_path)
            candidates, candidate_source = load_candidate_indices(handle, layout, args.candidate_manifest)
            env = make_env(suite, env_spec, bddl_path, control_delta=True)
            env.reset()
            if not bool(controller_for(env).use_delta):
                raise RuntimeError("capture environment is not delta OSC")
            try:
                for demo_index in candidates:
                    demo_key = f"demo_{demo_index}"
                    if accepted_by_layout[layout] >= args.target_per_layout:
                        break
                    attempt_ordinal += 1
                    record = capture_candidate(env, handle["data"][demo_key], layout, demo_key, output_dir, np.asarray(T_robot_base, dtype=np.float64), np.asarray(T_gripper, dtype=np.float64), migrate_saved_model_xml, attempt_ordinal, "formal_candidate")
                    record.update({"task_name": task_name, "source_hdf5": display_path(hdf5_path), "source_hdf5_sha256": source_hdf5_sha256, "candidate_source": candidate_source, "runtime": {"robosuite": REQUIRED_ROBOSUITE, "mujoco": REQUIRED_MUJOCO, "control_freq_hz": CONTROL_FREQ_HZ, "controller": "OSC_POSE", "control_delta": True}})
                    records.append(record)
                    if record.get("accepted"):
                        accepted_by_layout[layout] += 1
                    print(f"layout={layout} candidate={demo_key} accepted={record.get('accepted')} count={accepted_by_layout[layout]}/{args.target_per_layout}", flush=True)
                for diagnostic_layout, diagnostic_key in sorted(diagnostic_keys):
                    if diagnostic_layout != layout or diagnostic_key not in handle["data"]:
                        continue
                    if any(r["layout"] == layout and r["demo_key"] == diagnostic_key for r in records):
                        continue
                    attempt_ordinal += 1
                    record = capture_candidate(env, handle["data"][diagnostic_key], layout, diagnostic_key, output_dir, np.asarray(T_robot_base, dtype=np.float64), np.asarray(T_gripper, dtype=np.float64), migrate_saved_model_xml, attempt_ordinal, "required_diagnostic")
                    record.update({"task_name": task_name, "source_hdf5": display_path(hdf5_path), "source_hdf5_sha256": source_hdf5_sha256, "candidate_source": candidate_source, "runtime": {"robosuite": REQUIRED_ROBOSUITE, "mujoco": REQUIRED_MUJOCO, "control_freq_hz": CONTROL_FREQ_HZ, "controller": "OSC_POSE", "control_delta": True}})
                    records.append(record)
                    print(f"layout={layout} diagnostic={diagnostic_key} accepted={record.get('accepted')} quota_count={accepted_by_layout[layout]}/{args.target_per_layout}", flush=True)
            finally:
                env.close()
    per_layout = {}
    for layout in layouts:
        layout_records = [record for record in records if record["layout"] == layout]
        per_layout[str(layout)] = {
            "target": args.target_per_layout,
            "attempted": len(layout_records),
            "accepted": sum(bool(record["accepted"]) for record in layout_records),
            "failed_contact_without_grasp": sum(
                record["failure_stage"] == "contact_without_grasp"
                for record in layout_records
            ),
            "failed_no_contact": sum(
                record["failure_stage"] == "no_contact" for record in layout_records
            ),
            "failed_other": sum(
                not record["accepted"]
                and record["failure_stage"]
                not in {"contact_without_grasp", "no_contact"}
                for record in layout_records
            ),
            "passed": accepted_by_layout[layout] == args.target_per_layout,
        }
    result = {"stage": "V1-R.2K", "status": "passed" if all(v["passed"] for v in per_layout.values()) else "incomplete", "numeric_contract": {"id": CONTRACT_ID, "saved_label": "raw_action_float64.astype(float32)", "controller_command": "saved_label.astype(float64)", "dataset_minmax_normalization": False}, "runtime": {"robosuite": REQUIRED_ROBOSUITE, "mujoco": REQUIRED_MUJOCO, "control_freq_hz": CONTROL_FREQ_HZ, "controller": "OSC_POSE", "control_delta": True}, "candidate_order": "numeric_demo_index_ascending_within_layout", "intermediate_state_restore": False, "target_per_layout": args.target_per_layout, "per_layout": per_layout, "records": records, "next_stage": "run_strict_20_of_20_replay_gate" if all(v["passed"] for v in per_layout.values()) else "quantized_at_source_collection_incomplete"}
    args.manifest_output.resolve().write_text(json.dumps(jsonable(result), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
