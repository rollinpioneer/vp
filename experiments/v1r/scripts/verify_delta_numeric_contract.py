#!/usr/bin/env python3
"""Verify the V1-R.2J-N float32-label / float64-controller candidate.

The default scope is the four frozen V1-R.2J-F records.  It does not run the
20-record expansion automatically: the caller must explicitly pass
``--scope all`` after the selected-record gate reaches 4/4.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPSTREAM = ROOT / "third_party" / "pointbridge"
DEFAULT_MANIFEST = ROOT / "outputs" / "v1r" / "sequential_success_demos_2i" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "v1r" / "delta_numeric_contract_2j_n" / "selected_results.json"
REQUIRED_ROBOSUITE = "1.4.1"
REQUIRED_MUJOCO = "3.3.5"
SELECTED = {
    (1, "demo_18"),
    (2, "demo_27"),
    (3, "demo_10"),
    (4, "demo_2"),
}
CONDITION = "G_pointbridge_float32_label_float64_controller"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_delta_pose_path_diagnostic import (  # noqa: E402
    add_upstream_paths,
    file_sha256,
    float32_labels_to_float64_controller,
    jsonable,
    load_record,
    make_pointbridge_delta_environment,
    raw_array_sha256,
    replay_condition,
)
from verify_delta_pose_contract import _PrecomputedLanguageEncoder  # noqa: E402


def select_records(manifest: dict[str, Any], scope: str) -> list[dict[str, Any]]:
    accepted = [record for record in manifest["records"] if record.get("accepted")]
    if scope == "all":
        records = accepted
    else:
        records = [
            record
            for record in accepted
            if (int(record["layout"]), str(record["demo_key"])) in SELECTED
        ]
        found = {(int(record["layout"]), str(record["demo_key"])) for record in records}
        if found != SELECTED:
            raise ValueError(f"selected records missing: {sorted(SELECTED - found)}")
    return sorted(records, key=lambda item: (int(item["layout"]), str(item["demo_key"])))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scope", choices=("selected", "all"), default="selected")
    args = parser.parse_args()

    upstream = args.upstream.resolve()
    add_upstream_paths(upstream)
    os.environ.setdefault("MUJOCO_GL", "egl")
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
    records = select_records(manifest, args.scope)
    if args.scope == "all" and len(records) != 20:
        raise ValueError(f"all-record scope requires 20 accepted records, got {len(records)}")

    pb_suite.init_models = lambda: _PrecomputedLanguageEncoder()
    output_records: list[dict[str, Any]] = []
    for record in records:
        layout = int(record["layout"])
        arrays, model_xml = load_record(record)
        raw = np.asarray(arrays["issued_actions"], dtype=np.float64)
        labels, controller_actions, max_error = float32_labels_to_float64_controller(raw)
        env = make_pointbridge_delta_environment(pb_suite, upstream, f"bowl_on_plate_{layout}")
        env.reset()
        replay = replay_condition(
            env=env,
            environment_source="pointbridge_delta_pose",
            condition=CONDITION,
            record=record,
            arrays=arrays,
            model_xml=model_xml,
            actions=controller_actions,
            input_actions=labels,
            normalization_error=None,
        )
        env.close()
        output_records.append(
            {
                "layout": layout,
                "demo_key": record["demo_key"],
                "artifact": record["artifact"],
                "source_artifact_sha256": record["artifact_sha256"],
                "raw_actions_sha256": raw_array_sha256(raw),
                "label_float32_sha256": raw_array_sha256(labels),
                "controller_float64_sha256": raw_array_sha256(controller_actions),
                "max_abs_error_vs_raw": max_error,
                "replay": replay,
            }
        )
        print(
            f"layout={layout} demo={record['demo_key']} label={labels.dtype} "
            f"controller={controller_actions.dtype} success={replay['success']}",
            flush=True,
        )

    passed = sum(bool(record["replay"]["success"]) for record in output_records)
    checked = len(output_records)
    scope_gate_passed = passed == checked and checked == (4 if args.scope == "selected" else 20)
    selected_gate_passed = args.scope == "selected" and scope_gate_passed
    formal_gate_passed = args.scope == "all" and scope_gate_passed
    result = {
        "stage": "V1-R.2J-N",
        "status": (
            "completed_selected_gate_passed"
            if selected_gate_passed
            else "completed_formal_gate_passed"
            if formal_gate_passed
            else "completed_numeric_candidate_failed"
        ),
        "scope": args.scope,
        "candidate": {
            "label": "raw_actions.astype(float32)",
            "controller_input": "label_float32.astype(float64)",
            "dataset_minmax_normalization": False,
            "action_shape": [7],
        },
        "runtime": {
            "robosuite": REQUIRED_ROBOSUITE,
            "mujoco": REQUIRED_MUJOCO,
            "control_freq_hz": 20,
            "entrypoint": "point_bridge.suite.mimiclabs.make",
            "action_mode": "delta_pose",
        },
        "gate": {"checked": checked, "passed": passed, "required_passed": checked},
        "selected_gate_passed": selected_gate_passed,
        "formal_gate_passed": formal_gate_passed,
        "all_20_expansion_authorized": selected_gate_passed,
        "policy_training_or_inference": False,
        "intermediate_state_restore": False,
        "formal_20_demo_gate_changed": False,
        "source": {
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": file_sha256(manifest_path),
        },
        "records": output_records,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(jsonable(result), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    return 0 if scope_gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
