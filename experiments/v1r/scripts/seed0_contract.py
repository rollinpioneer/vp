#!/usr/bin/env python3
"""Shared validation and configuration helpers for the authorized seed-0 run."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_UPSTREAM_HEAD = "491db4c4652ebfbcfce241c1c5e83c6d0c9eea75"
HISTORICAL_PROTOCOL_HEAD = "5d567a62d62b5a97c5960d45024e065349680cda"
PATCHES = (
    ROOT / "patches/pointbridge/0005-mimiclabs-delta-pose-float32-identity-training.patch",
    ROOT / "patches/pointbridge/0006-mimiclabs-delta-pose-chunk-contract.patch",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return value


def upstream_head(upstream: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "--verify", "HEAD"],
        text=True,
    ).strip()


def validate_frozen_contract(config_path: Path, upstream: Path) -> dict[str, Any]:
    """Validate all tracked and local identities needed by B1-2K-20."""

    cfg = load_yaml(config_path)
    training = cfg.get("training", {})
    freeze = cfg.get("artifact_freeze", {})
    authorization = cfg.get("authorization", {})
    required = {
        "variant": "B1-2K-20",
        "action_mode": "delta_pose",
        "selected_label_contract": "delta_pose_float32_identity",
        "dataset_minmax_normalization": False,
        "action_chunking": True,
        "num_queries": 40,
        "history_len": 1,
        "batch_size": 16,
        "train_steps": 300010,
        "seed": 0,
        "layouts": [1, 2, 3, 4],
        "num_demos_per_layout": 5,
    }
    actual = {
        "variant": cfg.get("variant"),
        **{key: training.get(key) for key in required if key != "variant"},
    }
    mismatches = {
        key: (expected, actual.get(key))
        for key, expected in required.items()
        if actual.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"frozen training contract mismatch: {mismatches}")
    if authorization.get("seed0_training_authorized") is not True:
        raise ValueError("seed-0 training is not authorized by the frozen config")
    for key in (
        "b0_b1_training_authorized",
        "confirm_rollouts_authorized",
        "v2_formal_experiment_authorized",
        "v3_formal_experiment_authorized",
    ):
        if authorization.get(key) is not False:
            raise ValueError(f"authorization boundary changed: {key}")

    dataset_manifest = ROOT / "outputs/v1r/b1_2k_point_pkls_f/manifest.json"
    verification = ROOT / "outputs/v1r/b1_2k_point_pkls_f/verification.json"
    index = ROOT / "experiments/v1r/manifests/b1_2k_dataset_artifact_index.csv"
    paths = {
        "config": config_path,
        "dataset_manifest": dataset_manifest,
        "verification": verification,
        "dataset_index": index,
        "patch_0005": PATCHES[0],
        "patch_0006": PATCHES[1],
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen artifact(s): " + ", ".join(missing))

    expected_hashes = {
        "dataset_manifest": freeze.get("manifest_sha256"),
        "verification": freeze.get("verification_sha256"),
        "config": freeze.get("training_config_sha256"),
        "patch_0005": freeze.get("patch_0005_sha256"),
        "patch_0006": freeze.get("patch_0006_sha256"),
    }
    observed_hashes = {name: sha256(path) for name, path in paths.items()}
    for name, expected in expected_hashes.items():
        if expected and observed_hashes[name] != expected:
            raise ValueError(
                f"{name} SHA-256 mismatch: expected {expected}, got {observed_hashes[name]}"
            )

    dataset_payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    records = [
        record
        for record in dataset_payload.get("records", [])
        if record.get("layout") is not None
    ]
    if len(records) != 20 or any(
        sum(int(record["layout"]) == layout for record in records) != 5
        for layout in range(1, 5)
    ):
        raise ValueError("dataset manifest is not balanced at five episodes per layout")
    ordered_mapping = dataset_payload.get("ordered_episode_mapping_sha256")
    if ordered_mapping != freeze.get("ordered_episode_mapping_sha256"):
        raise ValueError(
            "ordered episode mapping SHA-256 mismatch: "
            f"expected {freeze.get('ordered_episode_mapping_sha256')}, got {ordered_mapping}"
        )
    verification_payload = json.loads(verification.read_text(encoding="utf-8"))
    if verification_payload.get("status") != "passed":
        raise ValueError("frozen Point Bridge dataset verification is not passed")

    current_head = upstream_head(upstream)
    if current_head != EXPECTED_UPSTREAM_HEAD:
        raise ValueError(
            "Point Bridge checkout identity mismatch: "
            f"expected current source {EXPECTED_UPSTREAM_HEAD}, got {current_head}"
        )
    identity = {
        "current_source_commit": current_head,
        "historical_protocol_commit": HISTORICAL_PROTOCOL_HEAD,
        "source_commit_matches_historical_protocol": current_head
        == HISTORICAL_PROTOCOL_HEAD,
    }
    return {
        "config": cfg,
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
        "sha256": observed_hashes,
        "pointbridge_identity": identity,
        "dataset_records": len(records),
        "ordered_episode_mapping_sha256": ordered_mapping,
    }


def training_overrides(cfg: dict[str, Any], device: str, experiment: str) -> list[str]:
    training = cfg["training"]
    return [
        "agent=pb",
        "suite=mimiclabs",
        "dataloader=mimiclabs",
        "suite.task_make_fn._target_=point_bridge.suite.mimiclabs.make",
        "eval=false",
        f"device={device}",
        "save_video=false",
        "use_tb=true",
        f"batch_size={training['batch_size']}",
        f"num_demos_per_task={training['num_demos_per_layout']}",
        f"suite.num_train_steps={training['train_steps']}",
        f"suite.save_every_steps={training['save_every_steps']}",
        f"suite.log_every_steps={training['log_every_steps']}",
        "use_language=false",
        "use_proprio=true",
        "action_chunking=true",
        "num_queries=40",
        "suite.history_len=1",
        "suite.obs_type=[points]",
        "suite.action_mode=delta_pose",
        f"dataloader.bc_dataset.path={training['dataset_path']}",
        "dataloader.bc_dataset.suffix=null",
        "dataloader.bc_dataset.task_indices=[0,1,2,3]",
        f"dataloader.bc_dataset.noise_object_points={str(training['noise_object_points']).lower()}",
        "suite.num_points_per_obj=128",
        f"experiment={experiment}",
        "seed=0",
    ]
