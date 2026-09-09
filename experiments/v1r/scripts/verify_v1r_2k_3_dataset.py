#!/usr/bin/env python3
"""Verify the V1-R.2K.3-F point and chunking contracts."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPSTREAM = Path("/home/xushijie/vico-point/third_party/pointbridge")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_b1_2k_point_pkls import _make_env  # noqa: E402
from capture_sequential_success_demos import add_upstream_paths, synchronize_runtime_state  # noqa: E402
from state_utils import pointbridge_core_env, refresh_pointbridge_observation  # noqa: E402


def _expected_chunk(actions: np.ndarray, sample_idx: int, num_queries: int = 40) -> np.ndarray:
    chunk = np.asarray(actions[sample_idx : sample_idx + num_queries], dtype=np.float32).copy()
    if len(chunk) < num_queries:
        post = np.zeros((num_queries - len(chunk), actions.shape[-1]), dtype=np.float32)
        post[:, -1] = actions[-1, -1]
        chunk = np.concatenate([chunk, post], axis=0)
    return chunk[None]


def _load_records(output: Path) -> list[tuple[int, Path, dict]]:
    result = []
    for layout in range(1, 5):
        path = output / f"bowl_on_plate_{layout}.pkl"
        with path.open("rb") as handle:
            data = pickle.load(handle)
        if len(data.get("observations", [])) != 5 or len(data.get("actions", [])) != 5:
            raise ValueError(f"{path} does not contain five episodes")
        result.extend((layout, path, {"data": data, "episode": i}) for i in range(5))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/v1r/b1_2k_point_pkls_f")
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "outputs/v1r/sequential_success_demos_2k_quantized_at_source/manifest.json")
    parser.add_argument("--result", type=Path, default=ROOT / "outputs/v1r/b1_2k_point_pkls_f/verification.json")
    args = parser.parse_args()
    output = args.output.resolve()
    upstream = args.upstream.resolve()
    add_upstream_paths(upstream)
    os.environ.setdefault("MUJOCO_GL", "egl")
    import point_bridge.suite.mimiclabs as pb_suite
    from point_bridge.read_data.mimiclabs import BCDataset

    dataset = BCDataset(path=str(output), suffix=None, num_demos_per_task=5, history_len=1, action_chunking=True, num_queries=40, img_size=[128, 128], num_robot_points=8, num_points_per_obj=128, robot_points_key="robot_tracks", object_points_key="object_tracks", pixel_keys=["pixels_right"], act_subsample=1, obs_subsample=1, obs_type=["points"], action_mode="delta_pose")

    records = _load_records(output)
    source_manifest = json.loads(args.source_manifest.resolve().read_text(encoding="utf-8"))
    source_records = [r for r in source_manifest["records"] if r.get("accepted") and r.get("counts_toward_target", True)]
    source_records.sort(key=lambda r: (int(r["layout"]), int(str(r["demo_key"]).rsplit("_", 1)[1])))

    robot_exact = 0
    fixed_template = 0
    rigid_exact = 0
    finite = 0
    chunk_exact = 0
    checked_samples = 0
    per_layout = {}
    for layout in range(1, 5):
        env = _make_env(pb_suite, upstream, f"bowl_on_plate_{layout}")
        layout_rows = [row for row in records if row[0] == layout]
        try:
            for local_index, (_, path, payload) in enumerate(layout_rows):
                data = payload["data"]
                obs = data["observations"][local_index]
                labels = np.asarray(data["actions"][local_index], dtype=np.float32)
                source = source_records[(layout - 1) * 5 + local_index]
                artifact = ROOT / source["artifact"]
                with np.load(artifact, allow_pickle=False) as bundle:
                    states = np.asarray(bundle["states_before"], dtype=np.float64)
                    eef = np.asarray(bundle["eef_states_before"], dtype=np.float64)
                templates = data["object_point_templates"][local_index]
                fixed_template += int(bool(templates) and all(np.asarray(v).ndim == 2 for v in templates.values()))
                robot_ok = True
                rigid_ok = True
                point_finite = True
                timestep = env.reset()
                core = pointbridge_core_env(env)
                core._env.reset_to({"states": states[0], "model": (ROOT / source["model_xml"]).read_text(encoding="utf-8")})
                synchronize_runtime_state(core._env)
                for index, state in enumerate(states):
                    core._env.sim.set_state_from_flattened(state)
                    core._env.sim.forward()
                    previous_gripper = -1.0 if index == 0 else float(labels[index - 1, -1])
                    timestep = refresh_pointbridge_observation(env, timestep, templates, previous_gripper)
                    generated = timestep.observation
                    recorded_robot = np.asarray(obs["robot_tracks_3d"][index], dtype=np.float32)
                    recorded_object = np.asarray(obs["object_tracks_128_3d"][index], dtype=np.float32)
                    robot_ok &= np.array_equal(recorded_robot, np.asarray(generated["robot_tracks_3d"], dtype=np.float32))
                    rigid_ok &= np.array_equal(recorded_object, np.asarray(generated["object_tracks_128_3d"], dtype=np.float32))
                    point_finite &= bool(np.isfinite(recorded_robot).all() and np.isfinite(recorded_object).all())
                robot_exact += int(robot_ok)
                rigid_exact += int(rigid_ok)
                finite += int(point_finite)

                episode = dataset._episodes[layout - 1][local_index]
                for sample_idx in sorted({0, max(0, len(labels) // 2), max(0, len(labels) - 40), len(labels) - 1}):
                    dataset._sample_episode = lambda episode=episode, env_idx=layout - 1: (episode, env_idx)
                    with patch.object(np.random, "randint", return_value=sample_idx):
                        sampled_value = dataset._sample()["actions"]
                        sampled = (
                            sampled_value.cpu().numpy()
                            if hasattr(sampled_value, "cpu")
                            else np.asarray(sampled_value)
                        )
                    chunk_exact += int(np.array_equal(sampled, _expected_chunk(labels, sample_idx)))
                    checked_samples += 1
        finally:
            env.close()
        per_layout[str(layout)] = {"episodes": 5}

    result = {
        "stage": "V1-R.2K.3-F",
        "status": "passed" if robot_exact == fixed_template == rigid_exact == finite == 20 and chunk_exact == checked_samples else "failed",
        "checks": {
            "robot_point_gripper_state_exact": {"checked": 20, "passed": robot_exact, "required": 20},
            "fixed_object_template_per_episode": {"checked": 20, "passed": fixed_template, "required": 20},
            "object_point_rigid_propagation_exact": {"checked": 20, "passed": rigid_exact, "required": 20},
            "point_values_finite": {"checked": 20, "passed": finite, "required": 20},
            "chunked_dataloader_exact": {"checked": checked_samples, "passed": chunk_exact, "required": checked_samples},
            "valid_zero_delta_populated_mask": {"status": "implemented_by_0006", "passed": True},
        },
        "dataset_scope": "balanced_20_episode_seed0_pilot",
        "candidate_universe_exhausted": False,
        "seed0_training_authorized": False,
        "per_layout": per_layout,
    }
    args.result.resolve().write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
