#!/usr/bin/env python3
"""Strict V1-R.2K replay gate for saved quantized-at-source labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    make_env,
    raw_array_sha256,
    synchronize_runtime_state,
)
from numeric_contract import CONTRACT_ID, array_sha256, decode_sequence, exact_contract_check  # noqa: E402


REQUIRED_ROBOSUITE = "1.4.1"
REQUIRED_MUJOCO = "3.3.5"
CONTROL_FREQ_HZ = 20


def replay(env: Any, initial_state: np.ndarray, model_xml: str, commands: np.ndarray) -> dict[str, Any]:
    env.reset_to({"states": initial_state, "model": model_xml})
    synchronize_runtime_state(env)
    actual_initial = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
    received: list[np.ndarray] = []
    states_after: list[np.ndarray] = []
    success_steps: list[int] = []
    for index, command in enumerate(commands):
        received.append(np.asarray(command, dtype=np.float64).copy())
        env.step(received[-1])
        states_after.append(np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy())
        if bool(env._check_success()):
            success_steps.append(index)
    received_array = np.asarray(received, dtype=np.float64)
    return {"initial_state_match": bool(np.array_equal(actual_initial, initial_state)), "initial_state_max_abs_error": float(np.max(np.abs(actual_initial - initial_state))), "success": bool(success_steps), "first_success_action_index": success_steps[0] if success_steps else None, "steps_executed": len(received_array), "received_commands": received_array, "states_after": np.asarray(states_after)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=ROOT / "third_party" / "pointbridge")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    add_upstream_paths(upstream)
    os.environ.setdefault("MUJOCO_GL", "egl")
    import h5py
    import mujoco
    import robosuite
    from mimiclabs.mimiclabs.envs.problems import MimicLabs_Lab1_Tabletop_Manipulation  # noqa: F401
    if str(getattr(robosuite, "__version__", "unknown")) != REQUIRED_ROBOSUITE:
        raise RuntimeError(f"expected robosuite {REQUIRED_ROBOSUITE}")
    if str(getattr(mujoco, "__version__", "unknown")) != REQUIRED_MUJOCO:
        raise RuntimeError(f"expected MuJoCo {REQUIRED_MUJOCO}")
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    accepted = [r for r in manifest["records"] if r.get("accepted") and r.get("counts_toward_target", True)]
    by_layout: dict[int, list[dict[str, Any]]] = {}
    for record in accepted:
        by_layout.setdefault(int(record["layout"]), []).append(record)
    if any(len(by_layout.get(layout, [])) != 5 for layout in (1, 2, 3, 4)):
        raise ValueError("strict V1-R.2K replay requires exactly five accepted records per layout")
    results: list[dict[str, Any]] = []
    for layout in (1, 2, 3, 4):
        task_name = f"bowl_on_plate_{layout}"
        hdf5_path = upstream / "data" / "mimicgen_data" / "bowl_on_plate" / task_name / "demo" / "demo.hdf5"
        bddl_path = upstream / "third_party" / "mimiclabs" / "mimiclabs" / "mimiclabs" / "task_suites" / "new_task_suite" / f"{task_name}.bddl"
        with h5py.File(hdf5_path, "r") as handle:
            env_spec = json.loads(str(handle["data"].attrs["env_args"]))
        env = make_env(robosuite, env_spec, bddl_path, control_delta=True)
        env.reset()
        if not bool(controller_for(env).use_delta):
            raise RuntimeError("replay environment is not delta OSC")
        try:
            for record in sorted(by_layout[layout], key=lambda r: int(str(r["demo_key"]).rsplit("_", 1)[1])):
                payload: dict[str, Any] = {"layout": layout, "demo_key": record["demo_key"], "artifact": record["artifact"], "exception": None}
                try:
                    artifact_path = ROOT / record["artifact"]
                    model_xml = (ROOT / record["model_xml"]).read_text(encoding="utf-8")
                    with np.load(artifact_path, allow_pickle=False) as bundle:
                        arrays = {key: np.asarray(bundle[key]).copy() for key in bundle.files}
                    labels = arrays["float32_labels"]
                    commands = decode_sequence(labels)
                    source = arrays["raw_actions_float64"]
                    stored_commands = arrays["controller_commands_float64"]
                    contract = exact_contract_check(source, labels, stored_commands)
                    replay_result = replay(env, arrays["initial_state"], model_xml, commands)
                    received = replay_result.pop("received_commands")
                    states_after = replay_result.pop("states_after")
                    captured_after = arrays["states_after"]
                    payload.update({"numeric_contract": contract, "replay": replay_result, "replay_command_exact": bool(np.array_equal(received, commands)), "replay_command_sha256": array_sha256(received), "captured_command_sha256": array_sha256(stored_commands), "state_sequence_exact": bool(np.array_equal(states_after, captured_after)), "state_sequence_max_abs_error": float(np.max(np.abs(states_after - captured_after))) if states_after.shape == captured_after.shape else None})
                    payload["passed"] = bool(contract["passed"] and replay_result["initial_state_match"] and replay_result["success"] and payload["replay_command_exact"])
                except Exception as exc:
                    payload["exception"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-3000:]}"
                    payload["passed"] = False
                results.append(payload)
                print(f"layout={layout} demo={record['demo_key']} passed={payload['passed']}", flush=True)
        finally:
            env.close()
    passed = sum(bool(r["passed"]) for r in results)
    result = {"stage": "V1-R.2K.1", "status": "passed" if passed == 20 else "failed", "numeric_contract": CONTRACT_ID, "formal_gate": {"checked": 20, "passed": passed, "required": 20}, "training_authorized": False, "seed0_training_authorized": False, "confirm_rollouts_authorized": False, "v2_formal_experiment_authorized": False, "v3_formal_experiment_authorized": False, "records": results, "source_manifest": str(manifest_path.relative_to(ROOT))}
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
