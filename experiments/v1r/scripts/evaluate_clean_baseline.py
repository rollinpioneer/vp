#!/usr/bin/env python3
"""Evaluate an independent clean manifest or record an auditable blocker.

The script intentionally does not reuse the legacy V1 E00 rows.  A rollout CSV
can be supplied by a real Point Bridge runner; without it the output is a
blocked gate rather than a guessed success rate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_audits(
    audit_results: dict[str, dict[str, object]] | None,
) -> dict[str, dict[str, object]]:
    field_names = {
        "runner_parity": (
            "status",
            "device",
            "scenarios",
            "passed_scenarios",
            "source_old_outcomes",
            "path_success",
        ),
        "cpu_cuda_parity": (
            "status",
            "deployment_device",
            "deployment_device_protocol_frozen",
            "formal_evaluation_device_constraint",
            "blockers",
        ),
        "expert_replay": (
            "status",
            "checked",
            "passed",
            "per_layout",
            "restored_initial_state_matches",
            "failed_layouts",
            "failed_records_diverging_at_step_1",
            "failure_localization",
        ),
    }
    return {
        name: {key: payload[key] for key in field_names[name] if key in payload}
        for name, payload in (audit_results or {}).items()
        if name in field_names
    }


def evaluate(
    manifest: Path,
    checkpoints: dict[int, Path],
    rollouts: list[Path],
    requested_seeds: list[int] | None,
    audit_results: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if not manifest_rows:
        raise ValueError("clean manifest is empty")
    if any(row.get("condition") != "E00_CLEAN" for row in manifest_rows):
        raise ValueError("clean baseline manifest contains non-clean rows")
    manifest_seeds = sorted({int(row["training_seed"]) for row in manifest_rows})
    seeds = sorted(requested_seeds if requested_seeds is not None else manifest_seeds)
    manifest_splits = sorted({row.get("split", "") for row in manifest_rows})
    evaluation_stage = (
        "dev_precheck" if seeds == [0] and manifest_splits == ["dev"] else "confirm"
    )
    result: dict[str, object] = {
        "stage": "V1-R.2",
        "status": "blocked_clean_baseline",
        "evaluation_stage": evaluation_stage,
        "training_seeds": seeds,
        "manifest": str(manifest),
        "manifest_rows": len(manifest_rows),
        "selected_manifest_rows": sum(1 for row in manifest_rows if int(row["training_seed"]) in seeds),
        "checkpoints": {
            str(seed): str(checkpoints[seed]) if seed in checkpoints else None
            for seed in seeds
        },
        "checkpoint_sha256": {
            str(seed): sha256(checkpoints[seed])
            if seed in checkpoints and checkpoints[seed].is_file()
            else None
            for seed in seeds
        },
        "rollouts": [str(path) for path in rollouts],
        "rollout_sha256": {
            str(path): sha256(path) if path.is_file() else None for path in rollouts
        },
        "success_rate": None,
        "pooled_success_rate": None,
        "per_seed": {},
        "simulator_exception_count": None,
        "action_decode_error_count": None,
        "paired_initial_state_check": "unresolved",
        "audit_gates": summarize_audits(audit_results),
        "thresholds": {
            "pooled_success_rate_min": 0.50,
            "seeds_at_or_above_0_45_min": 2,
            "per_seed_success_rate_min": 0.35,
        },
        "blockers": [],
    }
    blockers = result["blockers"]
    assert isinstance(blockers, list)
    missing_checkpoints = [seed for seed in seeds if seed not in checkpoints or not checkpoints[seed].is_file()]
    if missing_checkpoints:
        blockers.append(f"missing frozen checkpoint(s) for training seed(s): {missing_checkpoints}")
    if not rollouts:
        blockers.append("no independent clean-baseline rollout CSV was supplied")
        return result
    rollout_rows: dict[tuple[int, str], dict[str, str]] = {}
    for rollout in rollouts:
        if not rollout.is_file():
            blockers.append(f"rollout CSV does not exist: {rollout}")
            continue
        with rollout.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        required = {"scenario_id", "success", "simulator_exception", "action_decode_error"}
        missing = required.difference(rows[0] if rows else ())
        if missing:
            blockers.append(f"rollout CSV {rollout} is missing columns: {sorted(missing)}")
            continue
        for row in rows:
            row_seed = int(row.get("training_seed", "-1"))
            key = (row_seed, row["scenario_id"])
            if key in rollout_rows:
                blockers.append(f"duplicate rollout row: seed={row_seed}, scenario={row['scenario_id']}")
            rollout_rows[key] = row

    manifest_keys = [(int(row["training_seed"]), row["scenario_id"]) for row in manifest_rows if int(row["training_seed"]) in seeds]
    missing_ids = [key for key in manifest_keys if key not in rollout_rows]
    if missing_ids:
        blockers.append(f"rollout CSV is missing {len(missing_ids)} manifest scenarios")
        return result
    selected = [rollout_rows[key] for key in manifest_keys]
    result["success_rate"] = sum(int(row["success"]) for row in selected) / len(selected)
    result["pooled_success_rate"] = result["success_rate"]
    result["simulator_exception_count"] = sum(int(row["simulator_exception"]) for row in selected)
    result["action_decode_error_count"] = sum(int(row["action_decode_error"]) for row in selected)
    state_checks = []
    for row in selected:
        expected = row.get("expected_initial_state_sha256", "")
        actual = row.get("actual_initial_state_sha256", "")
        explicit = row.get("initial_state_match", "")
        state_checks.append(
            bool(expected)
            and bool(actual)
            and expected == actual
            and explicit in {"1", "true", "True", "passed"}
        )
    result["paired_initial_state_check"] = (
        "passed" if state_checks and all(state_checks) else "failed_or_unresolved"
    )
    per_seed: dict[str, dict[str, object]] = {}
    for seed in seeds:
        seed_rows = [
            row
            for row in selected
            if int(row.get("training_seed", "-1")) == seed
        ]
        if not seed_rows:
            continue
        per_seed[str(seed)] = {
            "rows": len(seed_rows),
            "success_rate": sum(int(row["success"]) for row in seed_rows) / len(seed_rows),
            "simulator_exception_count": sum(int(row["simulator_exception"]) for row in seed_rows),
            "action_decode_error_count": sum(int(row["action_decode_error"]) for row in seed_rows),
        }
    result["per_seed"] = per_seed
    seed_rates = [float(item["success_rate"]) for item in per_seed.values()]
    audits = audit_results or {}
    runner_parity_passed = audits.get("runner_parity", {}).get("status") == "passed"
    cpu_cuda_status = audits.get("cpu_cuda_parity", {}).get("status")
    deployment_protocol_frozen = bool(
        audits.get("cpu_cuda_parity", {}).get("deployment_device_protocol_frozen", False)
    )
    device_audit_passed = cpu_cuda_status == "passed" or deployment_protocol_frozen
    expert_replay_passed = audits.get("expert_replay", {}).get("status") == "passed"
    integrity_passed = (
        result["simulator_exception_count"] == 0
        and result["action_decode_error_count"] == 0
        and result["paired_initial_state_check"] == "passed"
    )
    passed = (
        not blockers
        and len(per_seed) == len(seeds) == 3
        and float(result["pooled_success_rate"]) >= 0.50
        and sum(rate >= 0.45 for rate in seed_rates) >= 2
        and min(seed_rates, default=0.0) >= 0.35
        and result["simulator_exception_count"] == 0
        and result["action_decode_error_count"] == 0
        and result["paired_initial_state_check"] == "passed"
        and runner_parity_passed
        and device_audit_passed
        and expert_replay_passed
    )
    if evaluation_stage == "dev_precheck":
        dev_passed = (
            not blockers
            and len(per_seed) == 1
            and float(result["pooled_success_rate"]) >= 0.50
            and integrity_passed
            and runner_parity_passed
            and device_audit_passed
            and expert_replay_passed
        )
        result["status"] = "dev_precheck_passed" if dev_passed else "blocked_clean_baseline"
        result["confirm_gate_status"] = "not_run"
        if float(result["pooled_success_rate"]) < 0.50:
            blockers.append(
                f"seed-0 dev success rate {result['pooled_success_rate']:.3f} is below the predeclared 0.50 threshold"
            )
        if not integrity_passed:
            blockers.append("simulator/action/initial-state integrity checks did not all pass")
        if not runner_parity_passed:
            blockers.append("runner parity audit is not passed")
        if not device_audit_passed:
            blockers.append("CPU/CUDA parity is not passed and no deployment device protocol is frozen")
        if not expert_replay_passed:
            blockers.append("expert replay audit is not passed")
    elif not blockers and len(per_seed) == len(seeds) == 3:
        result["status"] = "passed" if passed else "failed"
        if not runner_parity_passed:
            blockers.append("runner parity audit is not passed")
        if not device_audit_passed:
            blockers.append("CPU/CUDA parity is not passed and no deployment device protocol is frozen")
        if not expert_replay_passed:
            blockers.append("expert replay audit is not passed")
    elif not blockers:
        result["status"] = "incomplete_multi_seed_evaluation"
        blockers.append("three training seeds are required for the confirm clean-baseline gate")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append")
    parser.add_argument("--rollouts", type=Path, nargs="+")
    parser.add_argument("--training-seed", type=int, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("experiments/v1r/reports/clean_baseline_report.md"))
    parser.add_argument("--runner-parity", type=Path)
    parser.add_argument("--cpu-cuda-parity", type=Path)
    parser.add_argument("--expert-replay", type=Path)
    args = parser.parse_args()
    checkpoint_paths = args.checkpoint or []
    requested_seeds = args.training_seed
    if requested_seeds is None and checkpoint_paths and len(checkpoint_paths) == 1:
        requested_seeds = [0]
    checkpoints = {
        (requested_seeds[index] if requested_seeds and len(requested_seeds) == len(checkpoint_paths) else index): path
        for index, path in enumerate(checkpoint_paths)
    }
    audit_results = {}
    for name, path in (
        ("runner_parity", args.runner_parity),
        ("cpu_cuda_parity", args.cpu_cuda_parity),
        ("expert_replay", args.expert_replay),
    ):
        if path is not None:
            audit_results[name] = json.loads(path.read_text(encoding="utf-8"))
    result = evaluate(
        args.manifest,
        checkpoints,
        args.rollouts or [],
        requested_seeds,
        audit_results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    blockers = result.get("blockers", [])
    selected_rows = result.get("selected_manifest_rows", result["manifest_rows"])
    success_rate = result.get("success_rate")
    args.report.write_text(
        "# V1-R.2 Clean Baseline\n\n"
        f"状态：`{result['status']}`。\n\n"
        f"本次评测集行数：{selected_rows}。\n\n"
        + (f"success_rate：`{success_rate}`。\n\n" if success_rate is not None else "")
        + ("阻塞原因：\n" + "\n".join(f"- {item}" for item in blockers) + "\n" if blockers else "结果已写入 gate JSON。\n"),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
