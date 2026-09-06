#!/usr/bin/env python3
"""Launch the frozen balanced Point Bridge bowl_on_plate baseline."""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM = ROOT / "third_party" / "pointbridge"


def _integers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _strings(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--seeds", type=_integers, default=[0, 1, 2])
    parser.add_argument("--cuda-devices", type=_strings, default=["0", "1", "2"])
    parser.add_argument("--steps", type=int, default=300010)
    parser.add_argument("--save-every", type=int, default=100000)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-demos-per-task", type=int, default=44)
    args = parser.parse_args()
    if len(args.cuda_devices) < len(args.seeds):
        parser.error("provide at least one CUDA device per concurrent seed")

    upstream = args.upstream.resolve()
    data_dir = upstream / "expert_demos" / "mimiclabs__no_images"
    missing = [
        str(data_dir / f"bowl_on_plate_{index}.pkl")
        for index in range(1, 5)
        if not (data_dir / f"bowl_on_plate_{index}.pkl").exists()
    ]
    if missing:
        parser.error("missing generated PKLs: " + ", ".join(missing))
    for index in range(1, 5):
        path = data_dir / f"bowl_on_plate_{index}.pkl"
        with path.open("rb") as handle:
            episodes = len(pickle.load(handle)["observations"])
        if episodes < args.num_demos_per_task:
            parser.error(
                f"{path} has {episodes} episodes; "
                f"{args.num_demos_per_task} are required"
            )

    base_env = os.environ.copy()
    base_env.update(
        {
            "USE_TF": "0",
            "TRANSFORMERS_NO_TF": "1",
            "TENSORBOARD_NO_TF": "1",
            "MUJOCO_GL": "egl",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": os.pathsep.join(
                [str(upstream), str(upstream / "point_bridge"), str(ROOT / "src")]
            ),
        }
    )

    processes: list[tuple[int, Path, object, subprocess.Popen[str]]] = []
    for seed, device in zip(args.seeds, args.cuda_devices):
        env = base_env.copy()
        env["CUDA_VISIBLE_DEVICES"] = device
        experiment = f"v0_pointbridge_bowl_on_plate_seed{seed}"
        command = [
            sys.executable,
            "point_bridge/train.py",
            "agent=pb",
            "suite=mimiclabs",
            "dataloader=mimiclabs",
            "eval=false",
            "device=cuda",
            "save_video=false",
            "use_tb=true",
            f"batch_size={args.batch_size}",
            f"num_demos_per_task={args.num_demos_per_task}",
            f"suite.num_train_steps={args.steps}",
            f"suite.save_every_steps={args.save_every}",
            f"suite.log_every_steps={args.log_every}",
            "use_language=false",
            "use_proprio=true",
            "num_queries=40",
            "suite.history_len=1",
            "suite.obs_type=[points]",
            "dataloader.bc_dataset.suffix=_no_images",
            "dataloader.bc_dataset.task_indices=[0,1,2,3]",
            "dataloader.bc_dataset.noise_object_points=true",
            f"experiment={experiment}",
            "suite.action_mode=pose",
            "suite.num_points_per_obj=128",
            f"seed={seed}",
        ]
        log_path = upstream / f"{experiment}.launcher.log"
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=upstream,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((seed, log_path, log_handle, process))
        print(f"started seed={seed} pid={process.pid} cuda={device} log={log_path}")

    failed = False
    try:
        for seed, log_path, log_handle, process in processes:
            returncode = process.wait()
            log_handle.close()
            print(f"finished seed={seed} returncode={returncode} log={log_path}")
            failed = failed or returncode != 0
    except KeyboardInterrupt:
        for _, _, log_handle, process in processes:
            if process.poll() is None:
                process.terminate()
            log_handle.close()
        raise
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
