#!/usr/bin/env python3
"""Run paired E00, E10, and diagnostic-oracle Point Bridge rollouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM = ROOT / "third_party" / "pointbridge"
DEFAULT_SCENARIOS = ROOT / "manifests" / "v1_e00_e10_scenarios.csv"


def _add_paths(upstream: Path) -> None:
    paths = (
        ROOT / "src",
        upstream,
        upstream / "point_bridge",
        upstream / "third_party" / "LIBERO",
        upstream / "third_party" / "mimicgen",
        upstream / "third_party" / "mimiclabs",
        upstream / "third_party" / "robocasa",
    )
    for path in reversed(paths):
        sys.path.insert(0, str(path))


def _load_eval_module(upstream: Path):
    path = upstream / "point_bridge" / "eval.py"
    spec = importlib.util.spec_from_file_location("vico_pointbridge_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_scenarios(path: Path, training_seed: int, limit: int) -> list[dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["training_seed"]) != training_seed or row["condition"] != "E10":
                continue
            by_id[row["scenario_id"]] = row
    scenarios = [by_id[key] for key in sorted(by_id)]
    if len(scenarios) < limit:
        raise ValueError(f"seed {training_seed} has only {len(scenarios)} paired scenarios")
    return scenarios[:limit]


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compose_config(upstream: Path, checkpoint: Path, training_seed: int):
    from hydra import compose, initialize_config_dir

    overrides = [
        "agent=pb",
        "suite=mimiclabs",
        "dataloader=mimiclabs",
        "eval=true",
        "device=cuda",
        "save_video=false",
        "use_tb=false",
        "batch_size=16",
        "num_demos_per_task=44",
        "use_language=false",
        "use_proprio=true",
        "num_queries=40",
        "suite.history_len=1",
        "suite.eval_history_len=1",
        "suite.obs_type=[points]",
        "suite.pixel_keys=[pixels_left,pixels_right]",
        "suite.action_mode=pose",
        "suite.num_points_per_obj=128",
        "dataloader.bc_dataset.suffix=_no_images",
        "dataloader.bc_dataset.task_indices=[0,1,2,3]",
        "dataloader.bc_dataset.noise_object_points=false",
        f"seed={training_seed}",
        f"bc_weight={checkpoint}",
        f"experiment=v1_paired_seed{training_seed}",
    ]
    with initialize_config_dir(
        config_dir=str(upstream / "point_bridge" / "cfgs"), version_base=None
    ):
        return compose(config_name="config_eval", overrides=overrides)


def _state_sha256(env: Any) -> str:
    state = np.asarray(env.sim.get_state().flatten())
    return hashlib.sha256(state.tobytes()).hexdigest()


def _adapt_time_step(
    time_step: Any,
    env: Any,
    adapter: Any,
    window: Any,
    *,
    step: int,
    branch: str,
) -> tuple[Any, dict[str, float | int]]:
    from point_bridge.robot_utils.common.utils import depthimg2Meters
    from vico_point.policy.pointbridge_adapter import (
        fill_unknown_from_visible_group_centroids,
    )

    point_key = f"{env._object_points_key}_3d"
    object_points = np.asarray(time_step.observation[point_key])
    flat_points = object_points.reshape(-1, 3)
    pixel_key = env._pixel_keys[0]
    camera = env._pixelkey2camera[pixel_key]
    rgb, raw_depth = env.sim.render(
        env._width, env._height, camera_name=camera, depth=True
    )
    rgb = rgb[::-1]
    depth_m = depthimg2Meters(env._env, raw_depth[::-1])
    oracle = branch == "E10_ORACLE_HIDDEN_GT"
    condition = "E00" if branch == "E00" else "E10"
    adapted = adapter.adapt(
        flat_points,
        rgb,
        depth_m,
        env.intr[camera],
        env.extr[camera],
        step=step,
        condition=condition,
        window=window,
        oracle_hidden_truth=oracle,
    )
    if not oracle:
        adapter.audit_no_hidden_truth(adapted)
    policy_points, filled = fill_unknown_from_visible_group_centroids(
        adapted.policy_points,
        adapted.visible_mask,
        group_size=object_points.shape[-2],
    )
    observation = dict(time_step.observation)
    observation[point_key] = policy_points.reshape(object_points.shape).astype(
        object_points.dtype, copy=False
    )
    pixels = observation.get(pixel_key)
    if pixels is not None and pixels.ndim == 3 and pixels.shape[0] in (1, 3, 4):
        observation[pixel_key] = np.transpose(adapted.visibility.rgb, (2, 0, 1))
    diagnostics: dict[str, float | int] = {
        "visible_fraction": float(np.mean(adapted.visible_mask)),
        "last_reliable_hold_fraction": adapted.source.count("last_reliable_hold") / len(adapted.source),
        "unknown_fraction": sum(source == "unknown" for source in adapted.source)
        / len(adapted.source),
        "finite_fill_count": filled,
        "hidden_truth_reads": sum(source == "oracle_hidden_gt" for source in adapted.source),
    }
    return time_step._replace(observation=observation), diagnostics


def _run_branch(
    workspace: Any,
    env: Any,
    *,
    branch: str,
    simulator_seed: int,
    occlusion_start: int,
    occlusion_duration: int,
    fixed_object_points: dict[str, np.ndarray] | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import torch
    from point_bridge import utils
    from vico_point.envs.visibility import OcclusionWindow
    from vico_point.policy.pointbridge_adapter import CausalPointBridgeAdapter

    _set_seed(simulator_seed)
    kwargs = {"object_points": fixed_object_points} if fixed_object_points else {}
    time_step = env.reset(**kwargs)
    object_points = {
        key: np.array(value, copy=True) for key, value in env.object_points.items()
    }
    initial_state_sha256 = _state_sha256(env)
    point_key = f"{env._object_points_key}_3d"
    point_count = int(np.asarray(time_step.observation[point_key]).reshape(-1, 3).shape[0])
    adapter = CausalPointBridgeAdapter(tuple(f"point_{i:04d}" for i in range(point_count)))
    window = OcclusionWindow(occlusion_start, occlusion_duration)
    visible: list[float] = []
    held: list[float] = []
    unknown: list[float] = []
    hidden_reads = 0
    finite_fills = 0
    action_prefix = hashlib.sha256()
    step = 0
    workspace.agent.buffer_reset()
    while not time_step.last():
        policy_time_step, diagnostics = _adapt_time_step(
            time_step, env, adapter, window, step=step, branch=branch
        )
        with torch.no_grad(), utils.eval_mode(workspace.agent):
            action = workspace.agent.act(
                policy_time_step.observation,
                workspace.stats,
                step,
                workspace.global_step,
            )
        if step < occlusion_start:
            action_prefix.update(np.asarray(action).tobytes())
        visible.append(float(diagnostics["visible_fraction"]))
        held.append(float(diagnostics["last_reliable_hold_fraction"]))
        unknown.append(float(diagnostics["unknown_fraction"]))
        hidden_reads += int(diagnostics["hidden_truth_reads"])
        finite_fills += int(diagnostics["finite_fill_count"])
        time_step = env.step(action)
        step += 1
    return (
        {
            "branch": branch,
            "success": int(bool(time_step.observation["goal_achieved"])),
            "steps": step,
            "initial_state_sha256": initial_state_sha256,
            "pre_occlusion_action_sha256": action_prefix.hexdigest(),
            "mean_visible_fraction": float(np.mean(visible)) if visible else 0.0,
            "mean_last_reliable_hold_fraction": float(np.mean(held)) if held else 0.0,
            "mean_unknown_fraction": float(np.mean(unknown)) if unknown else 0.0,
            "finite_fill_count": finite_fills,
            "hidden_truth_reads": hidden_reads,
        },
        object_points,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--scenario-count", type=int, default=50)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    upstream = args.upstream.resolve()
    if not checkpoint.exists():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    if args.scenario_count <= 0:
        parser.error("--scenario-count must be positive")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TENSORBOARD_NO_TF", "1")
    _add_paths(upstream)
    eval_module = _load_eval_module(upstream)
    cfg = _compose_config(upstream, checkpoint, args.training_seed)
    old_cwd = Path.cwd()
    os.chdir(upstream)
    try:
        workspace = eval_module.Workspace(cfg)
        workspace.load_snapshot({"bc": checkpoint})
        workspace.agent.train(False)
        scenarios = _load_scenarios(
            args.scenarios.resolve(), args.training_seed, args.scenario_count
        )
        rows: list[dict[str, Any]] = []
        for index, scenario in enumerate(scenarios):
            env_index = index % len(workspace.env)
            env = workspace.env[env_index]
            simulator_seed = int(scenario["simulator_seed"])
            start = int(scenario["occlusion_start_step"])
            duration = int(scenario["occlusion_duration_steps"])
            fixed_points = None
            scenario_rows: list[dict[str, Any]] = []
            for branch in ("E00", "E10", "E10_ORACLE_HIDDEN_GT"):
                result, sampled_points = _run_branch(
                    workspace,
                    env,
                    branch=branch,
                    simulator_seed=simulator_seed,
                    occlusion_start=start,
                    occlusion_duration=duration,
                    fixed_object_points=fixed_points,
                )
                if fixed_points is None:
                    fixed_points = sampled_points
                result.update(
                    {
                        "training_seed": args.training_seed,
                        "scenario_id": scenario["scenario_id"],
                        "simulator_seed": simulator_seed,
                        "layout": f"bowl_on_plate_{env_index + 1}",
                        "occlusion_start_step": start,
                        "occlusion_duration_steps": duration,
                    }
                )
                scenario_rows.append(result)
                rows.append(result)
            state_hashes = {row["initial_state_sha256"] for row in scenario_rows}
            if len(state_hashes) != 1:
                raise RuntimeError(
                    f"paired initial state mismatch for {scenario['scenario_id']}"
                )
            action_prefix_hashes = {
                row["pre_occlusion_action_sha256"] for row in scenario_rows
            }
            if len(action_prefix_hashes) != 1:
                raise RuntimeError(
                    f"pre-occlusion action mismatch for {scenario['scenario_id']}"
                )
            if any(
                row["hidden_truth_reads"] for row in scenario_rows if row["branch"] != "E10_ORACLE_HIDDEN_GT"
            ):
                raise RuntimeError("non-oracle branch read hidden simulator truth")
            print(
                f"finished {scenario['scenario_id']} layout={env_index + 1} "
                + " ".join(f"{row['branch']}={row['success']}" for row in scenario_rows),
                flush=True,
            )
    finally:
        if "workspace" in locals():
            for env in workspace.env:
                env.close()
        os.chdir(old_cwd)

    output_dir = args.output_dir or (
        ROOT / "outputs" / "v1" / f"pointbridge_seed{args.training_seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (output_dir / "paired_rollouts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    by_branch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_branch[row["branch"]].append(row)
    summary = {
        "stage": "V1",
        "scientific_status": "pointbridge_checkpoint_paired_policy_evaluation",
        "training_seed": args.training_seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint),
        "paired_scenarios": len(scenarios),
        "branches": {
            branch: {
                "n": len(values),
                "success_rate": float(np.mean([row["success"] for row in values])),
                "mean_visible_fraction": float(
                    np.mean([row["mean_visible_fraction"] for row in values])
                ),
                "hidden_truth_reads": int(
                    sum(row["hidden_truth_reads"] for row in values)
                ),
            }
            for branch, values in sorted(by_branch.items())
        },
        "invariants": {
            "paired_initial_state_hashes_match": True,
            "pre_occlusion_action_hashes_match": True,
            "fixed_object_point_identity_within_pair": True,
            "non_oracle_hidden_truth_reads": 0,
            "oracle_is_diagnostic_only": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
