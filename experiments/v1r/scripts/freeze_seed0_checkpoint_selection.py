#!/usr/bin/env python3
"""Freeze the primary and diagnostic checkpoints before clean-dev evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-step", type=int, default=300000)
    parser.add_argument("--intermediate-step", type=int, action="append", default=[100000, 200000])
    args = parser.parse_args()

    root = args.training_root.resolve()
    metadata_path = root / "launcher_metadata.json"
    config_path = root / ".hydra" / "config.yaml"
    if not metadata_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("training output is missing launcher metadata or resolved Hydra config")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed" or metadata.get("return_code") != 0:
        raise ValueError("training output is not a successful completed run")

    steps = sorted(set(args.intermediate_step + [args.primary_step]))
    checkpoints = {}
    for step in steps:
        path = root / "snapshot" / f"{step}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing checkpoint: {path}")
        checkpoints[str(step)] = {
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "role": "primary_clean_dev" if step == args.primary_step else "diagnostic_only",
        }

    report = {
        "stage": "V1-R.2K.seed0_checkpoint_freeze",
        "status": "frozen_before_clean_dev",
        "training_root": str(root),
        "primary_checkpoint_step": args.primary_step,
        "intermediate_checkpoint_steps": [step for step in steps if step != args.primary_step],
        "checkpoint_selection_by_clean_dev": False,
        "selection_rule": "primary checkpoint is the final frozen 300000-step snapshot; intermediate snapshots are diagnostic only",
        "training_config": str(config_path),
        "training_config_sha256": sha256(config_path),
        "dataset_manifest_sha256": metadata.get("frozen", {}).get("sha256", {}).get("dataset_manifest"),
        "checkpoints": checkpoints,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    mirror = root / "checkpoint_selection.json"
    mirror.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
