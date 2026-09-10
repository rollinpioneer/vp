#!/usr/bin/env python3
"""Run D1 E2: offline action-block prediction without simulator rollouts."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_clean_b1_2k_20_seed0 import load_snapshot  # noqa: E402
from seed0_d1_common import (  # noqa: E402
    DEFAULT_UPSTREAM,
    RESOLVED_CONFIG,
    configure_runtime,
    read_csv,
    sha256,
    validate_preflight,
)


def _load_workspace(upstream: Path, checkpoint: Path, device: str, config: Path) -> Any:
    import importlib.util

    legacy_path = ROOT / "scripts/evaluate_pointbridge_paired.py"
    spec = importlib.util.spec_from_file_location("d1_offline_legacy", legacy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {legacy_path}")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    legacy._add_paths(upstream)
    eval_module = legacy._load_eval_module(upstream)
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config)
    cfg.eval = True; cfg.device = device; cfg.save_video = False; cfg.use_tb = False
    cfg.bc_weight = str(checkpoint); cfg.suite.num_eval_episodes = 1; cfg.expert_dataset = cfg.dataloader.bc_dataset
    old_cwd = Path.cwd()
    os.chdir(upstream)
    try:
        workspace = eval_module.Workspace(cfg)
        load_snapshot(workspace, checkpoint, device)
        workspace.agent.train(False)
        return workspace
    finally:
        os.chdir(old_cwd)


def _target_block(actions: np.ndarray, start: int, queries: int = 40) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(actions[start : start + queries], dtype=np.float32)
    valid = np.zeros(queries, dtype=bool)
    valid[: len(actual)] = True
    if len(actual) < queries:
        padding = np.zeros((queries - len(actual), actions.shape[1]), dtype=np.float32)
        padding[:, -1] = actions[-1, -1]
        actual = np.concatenate([actual, padding], axis=0)
    return actual, valid


def _predict(agent: Any, dataset: Any, episode: dict[str, Any], sample_idx: int, seed: int) -> np.ndarray:
    import torch
    from point_bridge import utils

    obs = episode["observation"]
    robot = np.asarray(obs["robot_tracks_3d"][sample_idx : sample_idx + 1, -dataset._num_robot_points :], dtype=np.float32)
    obj = np.asarray(obs["object_tracks_128_3d"][sample_idx : sample_idx + 1, :, : dataset._num_points_per_obj], dtype=np.float32).copy()
    # Reproduce the dataset's enabled object-point augmentation with a stable
    # diagnostic seed. This is a deterministic re-evaluation, not a claim to
    # reconstruct the RNG state of a particular training minibatch.
    np.random.seed(seed)
    if dataset._noise_object_points:
        obj += np.random.normal(0, dataset._noise_std, obj.shape).astype(np.float32)
    proprio = np.asarray(episode["observation"]["eef_states"][sample_idx : sample_idx + 1], dtype=np.float32)
    robot = dataset.preprocess["past_tracks"](robot)
    obj = dataset.preprocess["past_tracks"](obj)
    proprio = dataset.preprocess["proprioceptive"](proprio)
    with torch.no_grad(), utils.eval_mode(agent):
        robot_feat = agent.encoder(torch.as_tensor(robot, device=agent.device).float())
        object_feat = agent.encoder(torch.as_tensor(obj.reshape(1, -1, 3), device=agent.device).float())
        proprio_feat = agent.proprio_projector(torch.as_tensor(proprio, device=agent.device).float())
        features = torch.cat([robot_feat, object_feat, proprio_feat], dim=-1).view(1, -1, agent.repr_dim)
        predicted = agent.actor(features, 0, utils.schedule(agent.stddev_schedule, 0))
        if agent.policy_head == "deterministic":
            predicted = predicted.mean
        elif agent.policy_head == "diffusion":
            predicted = predicted["action_pred"] if isinstance(predicted, dict) else predicted
    return predicted.detach().cpu().numpy().reshape(agent.num_queries, agent._act_dim).astype(np.float32)


def _stats(values: list[np.ndarray], targets: list[np.ndarray], labels: list[str]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    pred = np.concatenate(values, axis=0)
    target = np.concatenate(targets, axis=0)
    err = pred - target
    result: dict[str, Any] = {
        "count": int(len(pred)),
        "mse": float(np.mean(err ** 2)),
        "mae": float(np.mean(np.abs(err))),
        "component_mae": [float(value) for value in np.mean(np.abs(err), axis=0)],
    }
    if pred.shape[1] == 7:
        result["per_component_mae"] = {
            "translation": float(np.mean(np.abs(err[:, :3]))),
            "rotation": float(np.mean(np.abs(err[:, 3:6]))),
            "gripper": float(np.mean(np.abs(err[:, 6]))),
        }
        result["gripper_sign_accuracy"] = float(
            np.mean((pred[:, 6] >= 0) == (target[:, 6] >= 0))
        )
    elif pred.shape[1] == 1:
        result["gripper_sign_accuracy"] = float(
            np.mean((pred[:, 0] >= 0) == (target[:, 0] >= 0))
        )
    if labels:
        result["labels"] = labels
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, default=RESOLVED_CONFIG)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    frozen = validate_preflight(args.upstream.resolve())
    if args.dataset.resolve() != Path(frozen["manifest"]["pkl_files"][0]["path"]).resolve().parent:
        raise ValueError("E2 dataset path must be the frozen b1_2k_point_pkls_f directory")
    rows = read_csv(args.manifest.resolve())
    if len(rows) != 20: raise ValueError("E2 requires the frozen 20-row trainfit manifest")
    configure_runtime(args.upstream.resolve())
    from omegaconf import OmegaConf
    workspace_by_step: dict[int, Any] = {}
    summaries: dict[str, Any] = {}
    old_cwd = Path.cwd()
    try:
        for checkpoint_arg in args.checkpoints:
            checkpoint = checkpoint_arg.resolve(); step = int(checkpoint.stem)
            if step not in (100000, 200000, 300000): raise ValueError("E2 checkpoint is not frozen")
            workspace = _load_workspace(args.upstream.resolve(), checkpoint, args.device, args.resolved_config.resolve())
            workspace_by_step[step] = workspace
            dataset = workspace.expert_replay_loader.dataset
            by_task = {name: idx for idx, name in enumerate(dataset.tasks)}
            all_values: list[np.ndarray] = []; all_targets: list[np.ndarray] = []
            all_terminal_values: list[np.ndarray] = []; all_terminal_targets: list[np.ndarray] = []
            horizon_values: dict[str, list[np.ndarray]] = defaultdict(list); horizon_targets: dict[str, list[np.ndarray]] = defaultdict(list)
            component_values: dict[str, list[np.ndarray]] = defaultdict(list); component_targets: dict[str, list[np.ndarray]] = defaultdict(list)
            layout_values: dict[int, list[np.ndarray]] = defaultdict(list); layout_targets: dict[int, list[np.ndarray]] = defaultdict(list)
            switch_near_values: list[np.ndarray] = []; switch_near_targets: list[np.ndarray] = []
            per_episode: list[dict[str, Any]] = []
            for row in rows:
                layout = int(row["layout"]); task = f"bowl_on_plate_{layout}"
                episode = dataset._episodes[by_task[task]][int(row["episode_index"])]
                actions = np.asarray(episode["action"], dtype=np.float32)
                episode_values=[]; episode_targets=[]; terminal_values=[]; terminal_targets=[]
                switch_indices = np.flatnonzero(np.diff(actions[:, 6], prepend=actions[:1, 6]) != 0)
                for sample_idx in range(len(actions)):
                    target, valid = _target_block(actions, sample_idx, 40)
                    pred = _predict(workspace.agent, dataset, episode, sample_idx, 700000 + layout * 10000 + sample_idx)
                    valid_pred = pred[valid]; valid_target = target[valid]
                    all_values.append(valid_pred); all_targets.append(valid_target)
                    episode_values.append(valid_pred); episode_targets.append(valid_target)
                    terminal_values.append(pred[~valid]); terminal_targets.append(target[~valid])
                    all_terminal_values.append(pred[~valid]); all_terminal_targets.append(target[~valid])
                    layout_values[layout].append(valid_pred); layout_targets[layout].append(valid_target)
                    if switch_indices.size:
                        valid_indices = np.flatnonzero(valid)
                        horizon_indices = sample_idx + valid_indices
                        near = np.any(np.abs(horizon_indices[:, None] - switch_indices[None, :]) <= 2, axis=1)
                        if near.any():
                            switch_near_values.append(pred[valid][near]); switch_near_targets.append(target[valid][near])
                    for h0, h1, name in ((0,1,"h0"),(1,10,"h1_9"),(10,20,"h10_19"),(20,40,"h20_39")):
                        mask = valid[h0:h1];
                        if mask.any():
                            horizon_values[name].append(pred[h0:h1][mask]); horizon_targets[name].append(target[h0:h1][mask])
                    for name, sl in (("translation", slice(0,3)),("rotation",slice(3,6)),("gripper",slice(6,7))):
                        component_values[name].append(valid_pred[:, sl]); component_targets[name].append(valid_target[:, sl])
                per_episode.append({
                    "layout": layout, "episode_key": row["episode_key"], "samples": len(actions),
                    "valid_action_count": int(sum(len(x) for x in episode_values)),
                    "terminal_padding_count": int(sum(len(x) for x in terminal_values)),
                    "metrics": _stats(episode_values, episode_targets, []),
                "terminal_padding_metrics": _stats(terminal_values, terminal_targets, []),
                })
            summaries[str(step)] = {
                "checkpoint_sha256": sha256(checkpoint), "valid_action_metrics": _stats(all_values, all_targets, []),
                "horizon_metrics": {name: _stats(horizon_values[name], horizon_targets[name], []) for name in ("h0","h1_9","h10_19","h20_39")},
                "component_metrics": {name: _stats(component_values[name], component_targets[name], []) for name in component_values},
                "per_layout": {str(layout): _stats(layout_values[layout], layout_targets[layout], []) for layout in sorted(layout_values)},
                "gripper_switch_near_metrics": _stats(switch_near_values, switch_near_targets, []),
                "terminal_padding_metrics": _stats(all_terminal_values, all_terminal_targets, []),
                "per_episode": per_episode,
                "noise_object_points": bool(dataset._noise_object_points), "noise_std": float(dataset._noise_std),
            }
            print(json.dumps({"checkpoint": step, "valid_mse": summaries[str(step)]["valid_action_metrics"]["mse"]}, indent=2), flush=True)
    finally:
        for workspace in workspace_by_step.values():
            for env in workspace.env:
                try: env.close()
                except Exception: pass
        os.chdir(old_cwd)
    result = {"stage":"V1-R.2K.seed0-D1.E2", "status":"complete", "diagnostic_only":True, "checkpoints":summaries,
              "dataset_manifest_sha256":frozen["sha256"]["dataset_manifest"], "resolved_config_sha256":sha256(args.resolved_config.resolve()),
              "terminal_padding_excluded_from_valid_metrics":True, "gate_impact":"diagnostic_only_not_a_clean_dev_gate"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
