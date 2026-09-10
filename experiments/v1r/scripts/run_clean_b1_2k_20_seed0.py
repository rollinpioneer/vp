#!/usr/bin/env python3
"""Evaluate a B1-2K-20 checkpoint using its saved resolved config.

Only evaluation-specific fields are overridden. In particular, the action
mode, point inputs, history, chunking, and dataset path are taken from the
training config rather than the legacy clean runner's hard-coded pose setup.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_clean_pointbridge import load_snapshot, run_episode, sha256  # noqa: E402
from seed0_contract import load_yaml, validate_frozen_contract  # noqa: E402
from state_utils import load_state_index  # noqa: E402


def _configure_runtime(upstream: Path) -> None:
    """Make the clean runner self-contained under the frozen runtime."""

    paths = [
        Path("/tmp/v1r_mujoco335"),
        ROOT / "src",
        upstream,
        upstream / "third_party/mimiclabs",
        upstream / "third_party/LIBERO",
        upstream / "third_party/mimicgen",
        upstream / "third_party/robocasa",
    ]
    existing = [item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item]
    merged = [str(path) for path in paths if path.is_dir()]
    merged.extend(existing)
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(merged))
    for path in reversed(merged):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TENSORBOARD_NO_TF", "1")


def _resolved_config(checkpoint: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    candidate = checkpoint.parent.parent / ".hydra" / "config.yaml"
    if candidate.is_file():
        return candidate
    candidate = checkpoint.parent.parent / "resolved_config.yaml"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError("could not locate training .hydra/config.yaml beside checkpoint")


def _check_config(cfg: dict[str, object]) -> None:
    suite = cfg.get("suite", {})
    if not isinstance(suite, dict):
        raise ValueError("resolved config has no suite mapping")
    checks = {
        "suite.action_mode": (suite.get("action_mode"), "delta_pose"),
        "action_chunking": (cfg.get("action_chunking"), True),
        "num_queries": (cfg.get("num_queries"), 40),
        "suite.history_len": (suite.get("history_len"), 1),
        "suite.obs_type": (suite.get("obs_type"), ["points"]),
        "use_proprio": (cfg.get("use_proprio"), True),
        "use_language": (cfg.get("use_language"), False),
        "dataset_minmax_normalization": (
            cfg.get("dataset_minmax_normalization", False),
            False,
        ),
    }
    errors = {key: value for key, value in checks.items() if value[0] != value[1]}
    if errors:
        raise ValueError(f"resolved config violates clean-dev contract: {errors}")


def _check_checkpoint_selection(
    checkpoint: Path, selection_path: Path
) -> dict[str, object]:
    """Require the primary checkpoint frozen before clean-dev evaluation."""

    selection = load_yaml(selection_path)
    if selection.get("status") != "frozen_before_clean_dev":
        raise ValueError(
            "checkpoint selection must be frozen_before_clean_dev before evaluation"
        )
    if selection.get("checkpoint_selection_by_clean_dev") is not False:
        raise ValueError("checkpoint selection must not depend on clean-dev outcomes")
    primary_step = int(selection.get("primary_checkpoint_step", -1))
    if primary_step != 300000 or checkpoint.name != f"{primary_step}.pt":
        raise ValueError(
            "clean-dev must use the frozen 300000-step primary checkpoint, "
            f"got {checkpoint.name}"
        )
    primary = selection.get("checkpoints", {}).get(str(primary_step), {})
    if not isinstance(primary, dict) or not primary.get("sha256"):
        raise ValueError("checkpoint selection is missing the primary checkpoint SHA-256")
    observed = sha256(checkpoint)
    if observed != primary["sha256"]:
        raise ValueError(
            "primary checkpoint SHA-256 mismatch: "
            f"expected {primary['sha256']}, got {observed}"
        )
    return {
        "path": str(selection_path.resolve()),
        "sha256": sha256(selection_path),
        "primary_checkpoint_step": primary_step,
        "primary_checkpoint_sha256": observed,
    }


def _check_smoke_checkpoint(checkpoint: Path) -> dict[str, object]:
    """Allow only a checkpoint produced by the isolated technical smoke run."""

    run_root = checkpoint.parent.parent
    if run_root.name != "v1r_b1_2k_20_seed0_smoke":
        raise ValueError(
            "--smoke evaluation requires a checkpoint from "
            "v1r_b1_2k_20_seed0_smoke"
        )
    return {
        "smoke": True,
        "path": str(checkpoint.resolve()),
        "sha256": sha256(checkpoint),
        "selection_rule": "technical_smoke_checkpoint_not_used_for_scientific_gate",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-selection",
        type=Path,
        default=ROOT / "experiments/v1r/reports/v1r_2k_seed0_checkpoint_selection.yaml",
    )
    parser.add_argument("--resolved-config", type=Path)
    parser.add_argument("--upstream", type=Path, default=Path("/home/xushijie/vico-point/third_party/pointbridge"))
    parser.add_argument("--state-index", type=Path, default=ROOT / "experiments/v1r/manifests/clean_state_index.csv")
    parser.add_argument("--split", choices=("dev", "confirm"), default="dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--cuda-visible-devices",
        help="Set CUDA_VISIBLE_DEVICES before importing torch (for example, 3).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    device = args.device or ("cpu" if args.smoke else "cuda")
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA clean-dev evaluation requested but torch.cuda.is_available() is false; "
                "run with a visible GPU or explicitly select --device cpu for diagnostics"
            )

    checkpoint = args.checkpoint.resolve()
    upstream = args.upstream.resolve()
    state_index_path = args.state_index.resolve()
    output_path = args.output.resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    frozen = validate_frozen_contract(ROOT / "experiments/v1r/configs/b1_2k_20_seed0.yaml", upstream)
    resolved_path = _resolved_config(checkpoint, args.resolved_config)
    resolved = load_yaml(resolved_path)
    _check_config(resolved)
    if args.smoke:
        if args.limit is None:
            args.limit = 1
        if args.limit < 1 or args.limit > 2:
            parser.error("--smoke requires --limit between 1 and 2")
        checkpoint_selection = _check_smoke_checkpoint(checkpoint)
    else:
        checkpoint_selection = _check_checkpoint_selection(
            checkpoint, args.checkpoint_selection.resolve()
        )
    _configure_runtime(upstream)

    # Reuse the legacy runner's audited rollout loop, but load the actual
    # training config through a small temporary Hydra composition shim.
    import importlib.util

    legacy_path = ROOT / "scripts" / "evaluate_pointbridge_paired.py"
    spec = importlib.util.spec_from_file_location("v1r_clean_legacy", legacy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {legacy_path}")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    legacy._add_paths(upstream)
    eval_module = legacy._load_eval_module(upstream)
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(resolved_path)
    cfg.eval = True
    cfg.device = device
    cfg.save_video = False
    cfg.use_tb = False
    cfg.bc_weight = str(checkpoint)
    cfg.suite.num_eval_episodes = 1
    cfg.expert_dataset = cfg.dataloader.bc_dataset
    old_cwd = Path.cwd()
    os.chdir(upstream)
    rows = []
    try:
        workspace = eval_module.Workspace(cfg)
        load_snapshot(workspace, checkpoint, device)
        workspace.agent.train(False)
        state_index = load_state_index(state_index_path)
        state_rows = [row for row in state_index.values() if row.get("split") == args.split]
        state_rows.sort(key=lambda row: row["scenario_id"])
        if args.limit is not None:
            state_rows = state_rows[: args.limit]
        if not args.smoke and args.split == "dev" and len(state_rows) != 40 and args.limit is None:
            raise ValueError(f"frozen clean dev must contain 40 rows, got {len(state_rows)}")
        for index, state_row in enumerate(state_rows):
            env = workspace.env[int(state_row["layout"]) - 1]
            row = {
                "training_seed": "0",
                "scenario_id": state_row["scenario_id"],
                "layout": state_row["layout"],
                "simulator_seed": state_row["simulator_seed"],
            }
            result = run_episode(workspace, env, row, 0, state_index, max_steps=None)
            result["checkpoint_sha256"] = sha256(checkpoint)
            result["evaluation_device"] = device
            if not np_action_shape_ok(result):
                raise ValueError(f"non-7D action was observed for {row['scenario_id']}")
            rows.append(result)
            print(f"{index + 1}/{len(state_rows)} {row['scenario_id']} success={result['success']}", flush=True)
    finally:
        if "workspace" in locals():
            for env in workspace.env:
                try:
                    env.close()
                except Exception:
                    pass
        os.chdir(old_cwd)
    if not rows:
        raise ValueError("clean evaluation selected no rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "stage": "V1-R.2K.seed0.smoke_eval" if args.smoke else "V1-R.2K.seed0.clean_dev",
        "smoke": args.smoke,
        "status": "complete",
        "rollouts": len(rows),
        "successes_total": sum(int(row["success"]) for row in rows),
        "successes_per_layout": {
            str(layout): sum(int(row["success"]) for row in rows if int(row["layout"]) == layout)
            for layout in range(1, 5)
        },
        "initial_state_matches": sum(row["initial_state_match"] == "passed" for row in rows),
        "historical_initial_state_matches": sum(
            row.get("historical_initial_state_match") == "passed" for row in rows
        ),
        "simulator_exceptions": sum(int(row["simulator_exception"]) for row in rows),
        "action_decode_errors": sum(int(row["action_decode_error"]) for row in rows),
        "failure_stage_counts": {
            stage: sum(row["failure_stage"] == stage for row in rows)
            for stage in sorted({str(row["failure_stage"]) for row in rows})
        },
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_selection": checkpoint_selection,
        "resolved_config_sha256": sha256(resolved_path),
        "dataset_manifest_sha256": frozen["sha256"]["dataset_manifest"],
        "evaluation_device": device,
        "pass": (
            bool(rows)
            and all(
                int(row["simulator_exception"]) == 0
                and int(row["action_decode_error"]) == 0
                for row in rows
            )
            if args.smoke
            else len(rows) == 40 and sum(int(row["success"]) for row in rows) >= 20
        ),
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["pass"] else 2


def np_action_shape_ok(result: dict[str, object]) -> bool:
    """The rollout loop records first actions; enforce the deployment shape."""

    first = result.get("first_20_actions_json", "[]")
    try:
        values = json.loads(str(first))
    except json.JSONDecodeError:
        return False
    return all(len(action) == 7 for action in values)


if __name__ == "__main__":
    raise SystemExit(main())
