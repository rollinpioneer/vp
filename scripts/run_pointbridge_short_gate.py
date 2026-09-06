#!/usr/bin/env python3
"""Run the minimal real Point Bridge training and checkpoint gate."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM = ROOT / "third_party" / "pointbridge"


def _task_indices(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--task-indices", type=_task_indices, default=[0, 1, 2, 3])
    args = parser.parse_args()

    upstream = args.upstream.resolve()
    required = [upstream / "point_bridge" / "train.py"]
    required.extend(
        upstream
        / "expert_demos"
        / "mimiclabs__no_images"
        / f"bowl_on_plate_{task_index + 1}.pkl"
        for task_index in args.task_indices
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("missing local Point Bridge inputs: " + ", ".join(missing))

    env = os.environ.copy()
    env.update(
        {
            "USE_TF": "0",
            "TRANSFORMERS_NO_TF": "1",
            "TENSORBOARD_NO_TF": "1",
            "MUJOCO_GL": "egl",
            "CUDA_VISIBLE_DEVICES": args.cuda_device,
            "PYTHONPATH": os.pathsep.join(
                [str(upstream), str(upstream / "point_bridge"), str(ROOT / "src")]
            ),
        }
    )
    command = [
        sys.executable,
        "point_bridge/train.py",
        "agent=pb",
        "suite=mimiclabs",
        "dataloader=mimiclabs",
        "eval=false",
        "device=cuda",
        "save_video=false",
        # Upstream currently populates actor_loss only when use_tb is enabled.
        "use_tb=true",
        f"batch_size={args.batch_size}",
        "num_demos_per_task=1",
        f"suite.num_train_steps={args.steps}",
        "suite.save_every_steps=1",
        "suite.log_every_steps=1",
        "use_language=false",
        "use_proprio=true",
        "num_queries=40",
        "suite.history_len=1",
        "suite.obs_type=[points]",
        "dataloader.bc_dataset.suffix=_no_images",
        "dataloader.bc_dataset.task_indices=" + str(args.task_indices).replace(" ", ""),
        "dataloader.bc_dataset.noise_object_points=false",
        "experiment=v0_pointbridge_short_train",
        "suite.action_mode=pose",
        "suite.num_points_per_obj=128",
        f"seed={args.seed}",
    ]
    return subprocess.run(command, cwd=upstream, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
