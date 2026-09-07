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


def evaluate(manifest: Path, checkpoint: Path | None, rollouts: Path | None, seed: int) -> dict[str, object]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if not manifest_rows:
        raise ValueError("clean manifest is empty")
    if any(row.get("condition") != "E00_CLEAN" for row in manifest_rows):
        raise ValueError("clean baseline manifest contains non-clean rows")
    result: dict[str, object] = {
        "stage": "V1-R.2",
        "status": "blocked_clean_baseline",
        "training_seed": seed,
        "manifest": str(manifest),
        "manifest_rows": len(manifest_rows),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": sha256(checkpoint) if checkpoint and checkpoint.is_file() else None,
        "success_rate": None,
        "simulator_exception_count": None,
        "action_decode_error_count": None,
        "paired_initial_state_check": "unresolved",
        "blockers": [],
    }
    blockers = result["blockers"]
    assert isinstance(blockers, list)
    if checkpoint is None or not checkpoint.is_file():
        blockers.append("a frozen Point Bridge checkpoint was not supplied")
    if rollouts is None or not rollouts.is_file():
        blockers.append("no independent clean-baseline rollout CSV was supplied")
        return result
    with rollouts.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"scenario_id", "success", "simulator_exception", "action_decode_error"}
    missing = required.difference(rows[0] if rows else ())
    if missing:
        blockers.append(f"rollout CSV is missing columns: {sorted(missing)}")
        return result
    by_id = {row["scenario_id"]: row for row in rows}
    missing_ids = [row["scenario_id"] for row in manifest_rows if row["scenario_id"] not in by_id]
    if missing_ids:
        blockers.append(f"rollout CSV is missing {len(missing_ids)} manifest scenarios")
        return result
    selected = [by_id[row["scenario_id"]] for row in manifest_rows]
    successes = sum(int(row["success"]) for row in selected)
    result["success_rate"] = successes / len(selected)
    result["simulator_exception_count"] = sum(int(row["simulator_exception"]) for row in selected)
    result["action_decode_error_count"] = sum(int(row["action_decode_error"]) for row in selected)
    result["paired_initial_state_check"] = "passed" if all(row.get("initial_state_sha256") for row in selected) else "unresolved"
    if result["simulator_exception_count"] == 0 and result["action_decode_error_count"] == 0 and result["paired_initial_state_check"] == "passed":
        result["status"] = "passed" if float(result["success_rate"]) >= 0.50 else "failed"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--rollouts", type=Path)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("experiments/v1r/reports/clean_baseline_report.md"))
    args = parser.parse_args()
    result = evaluate(args.manifest, args.checkpoint, args.rollouts, args.training_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    blockers = result.get("blockers", [])
    args.report.write_text(
        "# V1-R.2 Clean Baseline\n\n"
        f"状态：`{result['status']}`。\n\n"
        f"确认集行数：{result['manifest_rows']}。\n\n"
        + ("阻塞原因：\n" + "\n".join(f"- {item}" for item in blockers) + "\n" if blockers else "结果已写入 gate JSON。\n"),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
