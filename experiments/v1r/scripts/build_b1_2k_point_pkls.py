#!/usr/bin/env python3
"""Build the V1-R.2K.3 Point Bridge point-input PKLs.

This builder materializes observations from the causal ``states_before`` saved
by the quantized-at-source collector. It never restores a state between
actions during collection and never calls the historical PKL generator.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPSTREAM = Path("/home/xushijie/vico-point/third_party/pointbridge")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture_sequential_success_demos import add_upstream_paths, file_sha256, synchronize_runtime_state  # noqa: E402
from state_utils import refresh_pointbridge_observation  # noqa: E402


def _accepted_records(manifest: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    records = [r for r in payload["records"] if r.get("accepted") and r.get("counts_toward_target", True)]
    records.sort(key=lambda r: (int(r["layout"]), int(str(r["demo_key"]).rsplit("_", 1)[1])))
    if len(records) != 20 or any(sum(int(r["layout"]) == i for r in records) != 5 for i in range(1, 5)):
        raise ValueError("expected exactly five accepted quantized demos per layout")
    return records


class _ZeroLanguageEncoder:
    def encode(self, _text: str) -> np.ndarray:
        return np.zeros(384, dtype=np.float32)


def _make_env(pb_suite: Any, upstream: Path, task_name: str) -> Any:
    bddl_dir = upstream / "third_party/mimiclabs/mimiclabs/mimiclabs/task_suites/new_task_suite"
    pb_suite.init_models = lambda: _ZeroLanguageEncoder()
    envs, _ = pb_suite.make(
        bddl_dir=str(bddl_dir), task_names=[task_name], action_repeat=1,
        height=128, width=128, seed=0, max_episode_len=300, max_state_dim=100,
        eval=True, pixel_keys=["pixels_right"], num_robot_points=8,
        num_points_per_obj=128, robot_points_key="robot_tracks",
        object_points_key="object_tracks", obs_type=["points"],
        action_mode="delta_pose", use_vlm_points=False, vlm_mode="segment_depth",
        depth_type="gt", add_camera_from_extrinsics=False,
        camera_extrinsics_file="", temporal_agg_strategy="exponential_average",
        visualize_3d=False, use_full_scene_pcd=False, full_scene_num_points=512,
        max_depth_meters=2.0, min_z_robot_frame=0.0, full_scene_camera_name="cam_8_left",
    )
    if len(envs) != 1:
        raise RuntimeError(f"expected one Point Bridge environment, got {len(envs)}")
    return envs[0]


def _episode(env: Any, record: dict[str, Any], root: Path) -> dict[str, Any]:
    artifact = root / record["artifact"]
    model_xml = (root / record["model_xml"]).read_text(encoding="utf-8")
    with np.load(artifact, allow_pickle=False) as bundle:
        states = np.asarray(bundle["states_before"], dtype=np.float64).copy()
        labels = np.asarray(bundle["float32_labels"], dtype=np.float32).copy()
        eef = np.asarray(bundle["eef_states_before"], dtype=np.float64).copy()
    if len(states) != len(labels) or len(eef) != len(labels):
        raise ValueError(f"length mismatch in {artifact}")

    timestep = env.reset()
    core = env
    while hasattr(core, "_env"):
        if hasattr(core, "get_gt_points"):
            break
        core = core._env
    observations: dict[str, list[Any]] = {
        "robot_tracks_3d": [], "object_tracks_128_3d": [],
        "eef_states": [], "gripper_states": [], "states": [],
    }
    gripper = np.empty(len(labels), dtype=np.float32)
    gripper[0] = -1.0
    if len(labels) > 1:
        gripper[1:] = labels[:-1, -1]
    for index, state in enumerate(states):
        core._env.sim.set_state_from_flattened(state)
        core._env.sim.forward()
        timestep = refresh_pointbridge_observation(env, timestep, None)
        obs = timestep.observation
        observations["robot_tracks_3d"].append(np.asarray(obs["robot_tracks_3d"], dtype=np.float32))
        observations["object_tracks_128_3d"].append(np.asarray(obs["object_tracks_128_3d"], dtype=np.float32))
        observations["eef_states"].append(eef[index].astype(np.float32))
        observations["gripper_states"].append(gripper[index])
        observations["states"].append(state.astype(np.float32))
    packed = {key: np.asarray(value) for key, value in observations.items()}
    return {
        "observation": packed,
        "actions": labels,
        "task_emb": np.zeros(384, dtype=np.float32),
        "source_artifact": str(artifact.relative_to(root)),
        "source_artifact_sha256": file_sha256(artifact),
        "model_xml": model_xml,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/v1r/sequential_success_demos_2k_quantized_at_source/manifest.json")
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/v1r/b1_2k_point_pkls")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    upstream = args.upstream.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    add_upstream_paths(upstream)
    os.environ.setdefault("MUJOCO_GL", "egl")
    import point_bridge.suite.mimiclabs as pb_suite

    records = _accepted_records(manifest)
    by_layout: dict[int, list[dict[str, Any]]] = {i: [] for i in range(1, 5)}
    for record in records:
        by_layout[int(record["layout"])].append(record)
    manifest_records = []
    for layout in range(1, 5):
        env = _make_env(pb_suite, upstream, f"bowl_on_plate_{layout}")
        try:
            episodes = []
            for record in by_layout[layout]:
                episode = _episode(env, record, ROOT)
                episodes.append({k: v for k, v in episode.items() if k in {"observation", "actions", "task_emb"}})
                manifest_records.append({"layout": layout, "demo_key": record["demo_key"], "artifact": episode["source_artifact"], "artifact_sha256": episode["source_artifact_sha256"], "steps": len(episode["actions"])})
        finally:
            env.close()
        pkl_path = output / f"bowl_on_plate_{layout}.pkl"
        pkl_path.write_bytes(pickle.dumps({"observations": [e["observation"] for e in episodes], "actions": [e["actions"] for e in episodes], "task_emb": np.zeros(384, dtype=np.float32)}))
    result = {"stage": "V1-R.2K.3", "status": "built", "source_manifest": str(manifest.relative_to(ROOT)), "records": manifest_records, "dataset_contract": {"action_mode": "delta_pose", "history_len": 1, "action_chunking": True, "num_queries": 40, "gripper_proprio": "previous_issued_command", "terminal_padding": "zero_motion_hold_last_gripper"}, "seed0_training_authorized": False}
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
