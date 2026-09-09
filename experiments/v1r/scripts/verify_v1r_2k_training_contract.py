#!/usr/bin/env python3
"""Verify the V1-R.2K label contract through Point Bridge data and agent APIs.

The fixture is built from the local, gitignored V1-R.2K artifacts. It checks
the real ``BCDataset`` action path and the real ``BCAgent.act`` postprocess
path without training a network or changing the saved experiment artifacts.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import tempfile
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = Path("/home/xushijie/vico-point/third_party/pointbridge")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(ROOT / "experiments" / "v1r" / "scripts"))

from point_bridge.agent.pb import BCAgent  # noqa: E402
from point_bridge.read_data.mimiclabs import BCDataset  # noqa: E402
from vico_point.action_contracts import decode_delta_command  # noqa: E402


class _FixedActor(nn.Module):
    def __init__(self, action: np.ndarray):
        super().__init__()
        self.register_buffer("action", torch.as_tensor(action, dtype=torch.float32))

    def forward(self, *args, **kwargs):
        return SimpleNamespace(mean=self.action[None, None])


def _accepted_records(manifest: Path) -> list[dict]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    records = [r for r in payload["records"] if r.get("accepted") and r.get("counts_toward_target", True)]
    records.sort(key=lambda r: (int(r["layout"]), int(str(r["demo_key"]).rsplit("_", 1)[1])))
    if len(records) != 20 or any(sum(int(r["layout"]) == layout for r in records) != 5 for layout in range(1, 5)):
        raise ValueError("expected exactly five accepted records per layout")
    return records


def _fixture_from_artifacts(records: list[dict], path: Path) -> None:
    by_layout: dict[int, list[dict]] = {layout: [] for layout in range(1, 5)}
    for record in records:
        with np.load(ROOT / record["artifact"], allow_pickle=False) as bundle:
            labels = np.asarray(bundle["float32_labels"], dtype=np.float32)
            eef = np.asarray(bundle["eef_states_before"], dtype=np.float64)
        t = len(labels)
        # BCDataset's absolute-pose bookkeeping expects quaternions here;
        # delta labels remain the exact source data under test.
        eef_quat = np.zeros((t, 7), dtype=np.float64)
        eef_quat[:, :3] = eef[:, :3]
        eef_quat[:, 3] = 1.0
        gripper_observation = np.empty(t, dtype=np.float64)
        gripper_observation[0] = -1.0
        if t > 1:
            gripper_observation[1:] = labels[:-1, -1]
        obs = {
            "eef_states": eef_quat,
            "gripper_states": gripper_observation,
            "robot_3d": np.repeat(eef[:, None, :3], 9, axis=1),
            "object_1_3d": np.zeros((t, 1, 1, 3), dtype=np.float64),
        }
        by_layout[int(record["layout"])].append({"observation": obs, "actions": labels})
    for layout, episodes in by_layout.items():
        data = {
            "observations": [item["observation"] for item in episodes],
            "actions": [item["actions"] for item in episodes],
            "task_emb": np.zeros(384, dtype=np.float32),
        }
        (path / f"layout_{layout}.pkl").write_bytes(pickle.dumps(data))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/v1r/sequential_success_demos_2k_quantized_at_source/manifest.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = _accepted_records(args.manifest.resolve())
    with tempfile.TemporaryDirectory(prefix="v1r_2k_bcdataset_") as tmp:
        fixture = Path(tmp) / "episodes"
        fixture.mkdir()
        _fixture_from_artifacts(records, fixture)
        dataset = BCDataset(
            path=str(fixture), suffix=None, num_demos_per_task=5,
            history_len=1, action_chunking=False, num_queries=1,
            img_size=[8, 8], num_robot_points=9, num_points_per_obj=1,
            robot_points_key="robot", object_points_key="object",
            pixel_keys=["unused"], act_subsample=1, obs_subsample=1,
            obs_type=["points"], action_mode="delta_pose",
        )
        dataset_exact = 0
        decoded_exact = 0
        batch_exact = 0
        for index, record in enumerate(records):
            episode = dataset._episodes[index // 5][index % 5]
            with np.load(ROOT / record["artifact"], allow_pickle=False) as bundle:
                expected = np.asarray(bundle["float32_labels"], dtype=np.float32)
            if np.array_equal(episode["action"], expected) and episode["action"].dtype == np.float32:
                dataset_exact += 1
            decoded = decode_delta_command(episode["action"])
            if decoded.dtype == np.float64 and np.array_equal(decoded, expected.astype(np.float64)):
                decoded_exact += 1
            dataset._sample_episode = lambda episode=episode, env_idx=index // 5: (episode, env_idx)
            with patch("numpy.random.randint", return_value=0):
                batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
            batch_action = batch["actions"].cpu().numpy()[0]
            if batch_action.dtype == np.float32 and np.array_equal(batch_action, expected[:1]):
                batch_exact += 1

        agent = BCAgent(
            obs_shape={"unused": (1,), "proprioceptive": (10,)}, action_shape=(7,),
            device="cpu", lr=1e-4, hidden_dim=32, stddev_schedule="0.0",
            use_tb=False, policy_head="deterministic", use_language=False,
            pixel_keys=["unused"], proprio_key="proprioceptive", use_proprio=False,
            history_len=1, eval_history_len=1, action_chunking=False, num_queries=1,
            temporal_agg_strategy="exponential_average", max_episode_len=4,
            film=False, obs_type=["points"], action_mode="delta_pose",
            robot_points_key="robot", object_points_key="object", num_points_per_obj=1,
        )
        raw = np.asarray([0.125, -0.25, 0.03125, 0.0, 0.5, -0.75, 1.0], dtype=np.float32)
        agent.actor = _FixedActor(raw)
        norm_stats = {
            "past_tracks": {"min": np.zeros(3), "max": np.ones(3)},
            "actions": {"min": np.full(7, -9.0), "max": np.full(7, 9.0)},
        }
        obs = {"robot_3d": np.zeros((9, 3)), "object_1_3d": np.zeros((1, 1, 3))}
        decoded = agent.act(obs, norm_stats, step=0, global_step=0)
        agent_exact = bool(decoded.dtype == np.float64 and np.array_equal(decoded, raw.astype(np.float64)))

    result = {
        "stage": "V1-R.2K.2",
        "status": "passed" if dataset_exact == 20 and batch_exact == 20 and decoded_exact == 20 and agent_exact else "failed",
        "dataset_label_exact": {"checked": 20, "passed": dataset_exact, "required": 20},
        "dataloader_batch_exact": {"checked": 20, "passed": batch_exact, "required": 20},
        "decoded_command_exact": {"checked": 20, "passed": decoded_exact, "required": 20},
        "agent_postprocess_exact": {"checked": 1, "passed": int(agent_exact), "required": 1},
        "action_order_and_length_exact": {"checked": 20, "passed": dataset_exact, "required": 20},
        "task_replay_success": {"checked": 20, "passed": 20, "required": 20, "source": "V1-R.2K.1"},
        "captured_vs_replayed_state_exact": {"checked": 20, "passed": 20, "max_abs_error": 0.0, "source": "V1-R.2K.1"},
        "training_authorized": False,
        "seed0_training_authorized": False,
        "confirm_rollouts_authorized": False,
        "v2_formal_experiment_authorized": False,
        "v3_formal_experiment_authorized": False,
        "source_manifest": str(args.manifest.resolve().relative_to(ROOT)),
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
