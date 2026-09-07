#!/usr/bin/env python3
"""Verify frozen clean state files, manifest references, and runtime restores."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPSTREAM = ROOT / "third_party" / "pointbridge"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from state_utils import (
    load_state_bundle,
    load_state_index,
    raw_array_sha256,
)


def load_legacy_runner():
    path = ROOT / "scripts" / "evaluate_pointbridge_paired.py"
    spec = importlib.util.spec_from_file_location("vico_pointbridge_state_verifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_manifest(path: Path, state_index: dict[str, dict[str, str]]) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_scenario: dict[str, list[dict[str, str]]] = defaultdict(list)
    errors = []
    for row in rows:
        by_scenario[row["scenario_id"]].append(row)
        state = state_index.get(row["scenario_id"])
        if state is None:
            errors.append(f"missing state index row: {row['scenario_id']}")
            continue
        expected = {
            "state_path": state["state_path"],
            "saved_state_sha256": state["state_sha256"],
            "restored_state_sha256": state["restored_state_sha256"],
            "state_file_sha256": state["state_file_sha256"],
        }
        for key, value in expected.items():
            if row.get(key) != value:
                errors.append(f"{row['scenario_id']} manifest {key} does not match state index")
    for scenario_id, values in by_scenario.items():
        seeds = {int(row["training_seed"]) for row in values}
        if seeds != {0, 1, 2}:
            errors.append(f"{scenario_id} does not contain training seeds 0,1,2")
        for key in (
            "state_path",
            "saved_state_sha256",
            "restored_state_sha256",
            "state_file_sha256",
        ):
            if len({row.get(key) for row in values}) != 1:
                errors.append(f"{scenario_id} changes {key} across training seeds")
    return {
        "path": str(path.relative_to(ROOT)),
        "rows": len(rows),
        "unique_scenarios": len(by_scenario),
        "errors": errors,
    }


def verify_files(index: dict[str, dict[str, str]]) -> tuple[list[dict[str, Any]], list[str]]:
    records = []
    errors = []
    for scenario_id, row in index.items():
        try:
            bundle = load_state_bundle(ROOT, row)
            file_hash_match = bundle["file_sha256"] == row["state_file_sha256"]
            if not file_hash_match:
                errors.append(f"{scenario_id} state file SHA-256 mismatch")
            records.append(
                {
                    "scenario_id": scenario_id,
                    "state_path": row["state_path"],
                    "saved_state_sha256": row["state_sha256"],
                    "restored_state_sha256": row["restored_state_sha256"],
                    "state_file_sha256": row["state_file_sha256"],
                    "file_hash_match": file_hash_match,
                    "state_length": int(row.get("state_length", "0")),
                }
            )
        except Exception as exc:
            errors.append(f"{scenario_id}: {type(exc).__name__}: {exc}")
    return records, errors


def verify_runtime(
    index: dict[str, dict[str, str]],
    checkpoint: Path,
    upstream: Path,
    repeats: int,
) -> list[dict[str, Any]]:
    legacy = load_legacy_runner()
    legacy._add_paths(upstream)
    eval_module = legacy._load_eval_module(upstream)
    cfg = legacy._compose_config(upstream, checkpoint, 0, device="cpu")
    cfg.save_video = False
    cfg.use_tb = False
    records = []
    old_cwd = Path.cwd()
    os.chdir(upstream)
    try:
        workspace = eval_module.Workspace(cfg)
        for item_index, (scenario_id, row) in enumerate(index.items()):
            bundle = load_state_bundle(ROOT, row)
            env = workspace.env[int(row["layout"]) - 1]
            hashes = []
            exception = ""
            try:
                for _ in range(repeats):
                    env.reset()
                    env.sim.set_state_from_flattened(bundle["sim_state"])
                    env.sim.forward()
                    hashes.append(raw_array_sha256(env.sim.get_state().flatten()))
            except Exception as exc:
                exception = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}"
            records.append(
                {
                    "scenario_id": scenario_id,
                    "expected_restored_state_sha256": row["restored_state_sha256"],
                    "actual_restored_state_sha256": hashes,
                    "repeat_equal": len(set(hashes)) == 1 if hashes else False,
                    "expected_equal": bool(hashes)
                    and all(value == row["restored_state_sha256"] for value in hashes),
                    "exception": exception,
                }
            )
            print(
                f"{item_index + 1}/{len(index)} {scenario_id} "
                f"match={records[-1]['expected_equal']} repeat={records[-1]['repeat_equal']}",
                flush=True,
            )
    finally:
        if "workspace" in locals():
            for env in workspace.env:
                env.close()
        os.chdir(old_cwd)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-state-index", type=Path, required=True)
    parser.add_argument("--runner-state-index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TENSORBOARD_NO_TF", "1")
    clean_index = load_state_index(args.clean_state_index.resolve())
    runner_index = load_state_index(args.runner_state_index.resolve())
    file_records, errors = verify_files({**clean_index, **runner_index})
    manifest_records = [
        verify_manifest(path.resolve(), clean_index) for path in args.manifest
    ]
    errors.extend(
        error
        for manifest in manifest_records
        for error in manifest["errors"]
    )
    runtime_records = (
        verify_runtime(
            {**clean_index, **runner_index},
            args.checkpoint.resolve(),
            args.upstream.resolve(),
            args.repeats,
        )
        if args.runtime
        else []
    )
    if args.runtime:
        errors.extend(
            f"runtime restore failed: {record['scenario_id']}"
            for record in runtime_records
            if not record["expected_equal"]
            or not record["repeat_equal"]
            or record["exception"]
        )
    result = {
        "stage": "V1-R.2F",
        "audit": "initial_state_pairing",
        "status": "passed" if not errors and args.runtime else "static_passed" if not errors else "failed",
        "clean_states": len(clean_index),
        "runner_parity_states": len(runner_index),
        "splits": dict(Counter(row.get("split", "audit") for row in clean_index.values())),
        "runtime_repeats": args.repeats if args.runtime else 0,
        "manifest_records": manifest_records,
        "file_records": file_records,
        "runtime_records": runtime_records,
        "errors": errors,
        "seed_only_reproduction": {
            "runner_parity_states_checked": len(runner_index),
            "historical_hashes_reproduced": sum(
                row.get("source_old_initial_state_sha256") == row["state_sha256"]
                for row in runner_index.values()
            ),
            "conclusion": "simulator_seed_is_not_a_complete_state_identifier",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# V1-R.2F Initial State Audit\n\n"
        f"状态：`{result['status']}`。clean 状态：{len(clean_index)}；runner parity 状态：{len(runner_index)}。\n\n"
        f"dev/confirm 唯一状态：`{result['splits']}`；每个 clean scenario 的 training seed 0/1/2 引用同一状态路径和 SHA-256。\n\n"
        f"旧 V1 的 10 个 seed-only 场景历史 hash 可重现数：`{result['seed_only_reproduction']['historical_hashes_reproduced']}/10`，因此后续不再把 simulator seed 当作完整初态。\n\n"
        + (f"运行时对每个状态重复恢复 {args.repeats} 次，并严格比较 `H_expected == H_actual`。\n" if args.runtime else "尚未执行运行时恢复。\n")
        + ("\n错误：\n" + "\n".join(f"- {item}" for item in errors) + "\n" if errors else ""),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in result.items() if key not in {"file_records", "runtime_records"}}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
