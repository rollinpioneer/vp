#!/usr/bin/env python3
"""Shared identity and runtime helpers for V1-R.2K.seed0-D1."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPSTREAM = Path("/home/xushijie/vico-point/third_party/pointbridge")
PROTOCOL = ROOT / "experiments/v1r/configs/v1r_2k_seed0_d1_protocol.yaml"
DATASET_MANIFEST = ROOT / "outputs/v1r/b1_2k_point_pkls_f/manifest.json"
RESOLVED_CONFIG = ROOT / "outputs/v1r/training/v1r_b1_2k_20_seed0/.hydra/config.yaml"
CHECKPOINT_SELECTION = ROOT / "experiments/v1r/reports/v1r_2k_seed0_checkpoint_selection.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def load_protocol() -> dict[str, Any]:
    return load_yaml(PROTOCOL)


def configure_runtime(upstream: Path) -> None:
    paths = [
        Path("/tmp/v1r_mujoco335"), ROOT / "src", upstream,
        upstream / "third_party/mimiclabs", upstream / "third_party/LIBERO",
        upstream / "third_party/mimicgen", upstream / "third_party/robocasa",
    ]
    existing = [item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item]
    merged = [str(path) for path in paths if path.is_dir()] + existing
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(merged))
    for path in reversed(merged):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TENSORBOARD_NO_TF", "1")


def validate_preflight(upstream: Path = DEFAULT_UPSTREAM) -> dict[str, Any]:
    protocol = load_protocol()
    frozen = protocol["frozen_artifacts"]
    files = {
        "resolved_config": RESOLVED_CONFIG,
        "dataset_manifest": DATASET_MANIFEST,
    }
    for step in (100000, 200000, 300000):
        files[f"checkpoint_{step}"] = ROOT / f"outputs/v1r/training/v1r_b1_2k_20_seed0/snapshot/{step}.pt"
    for step, rel in {
        100000: "outputs/v1r/checkpoint_diagnostics/seed0_100000.csv",
        200000: "outputs/v1r/checkpoint_diagnostics/seed0_200000.csv",
        300000: "outputs/v1r/training/v1r_b1_2k_20_seed0/clean_dev_seed0_verified.csv",
    }.items():
        files[f"clean_dev_{step}"] = ROOT / rel
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing D1 frozen artifacts: " + ", ".join(missing))
    expected = {
        "resolved_config": frozen["resolved_config_sha256"],
        "dataset_manifest": frozen["dataset_manifest_sha256"],
        **{f"checkpoint_{k}": v for k, v in frozen["checkpoint_sha256"].items()},
        **{f"clean_dev_{k}": v for k, v in frozen["clean_dev_csv_sha256"].items()},
    }
    observed = {key: sha256(path) for key, path in files.items()}
    mismatches = {key: (expected[key], value) for key, value in observed.items() if expected[key] != value}
    if mismatches:
        raise ValueError(f"D1 frozen artifact hash mismatch: {mismatches}")
    manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("ordered_episode_mapping_sha256") != frozen["ordered_episode_mapping_sha256"]:
        raise ValueError("ordered episode mapping hash mismatch")
    authorization = protocol["authorization"]
    required_false = (
        "seed0_retraining_authorized", "confirm_rollouts_authorized",
        "remaining_training_seeds_authorized", "v2_formal_experiment_authorized",
        "v3_formal_experiment_authorized", "pb_r0_authorized",
    )
    if not authorization.get("seed0_original_training_authorization_consumed"):
        raise ValueError("seed-0 training authorization is not marked consumed")
    if any(authorization.get(key) is not False for key in required_false):
        raise ValueError("D1 authorization boundary has changed")
    source_anchor = protocol["source_anchor"]["vp_source_commit"]
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", source_anchor, "HEAD"], check=True)
    critical = protocol["pointbridge_identity"]["critical_file_sha256"]
    for relative, expected_hash in critical.items():
        path = upstream / relative
        if sha256(path) != expected_hash:
            raise ValueError(f"Point Bridge critical file hash mismatch: {relative}")
    return {"protocol": protocol, "files": files, "sha256": observed, "manifest": manifest}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
