#!/usr/bin/env python3
"""Generate Point Bridge PKLs for selected MimicLabs layouts in parallel."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM = ROOT / "third_party" / "pointbridge"
DEFAULT_TASKS = [f"bowl_on_plate_{index}" for index in range(1, 5)]


def _strings(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--tasks", type=_strings, default=DEFAULT_TASKS)
    parser.add_argument("--demo-start", type=int, default=0)
    parser.add_argument("--demo-stop", type=int, default=300)
    parser.add_argument("--max-accepted", type=int, default=100)
    parser.add_argument("--cuda-devices", type=_strings, default=["0", "1", "2", "3"])
    parser.add_argument("--audit-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    if args.demo_stop <= args.demo_start:
        parser.error("--demo-stop must be greater than --demo-start")
    if not args.cuda_devices:
        parser.error("at least one CUDA device is required")

    upstream = args.upstream.resolve()
    patch_command = [
        sys.executable,
        str(ROOT / "scripts" / "apply_pointbridge_patches.py"),
        "--upstream",
        str(upstream),
    ]
    if subprocess.run(patch_command, check=False).returncode != 0:
        return 1

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = upstream / "exp_local" / "pkl_generation" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
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

    processes: list[tuple[str, Path, object, subprocess.Popen[str]]] = []
    for index, task in enumerate(args.tasks):
        device = args.cuda_devices[index % len(args.cuda_devices)]
        env = base_env.copy()
        env["CUDA_VISIBLE_DEVICES"] = device
        command = [
            sys.executable,
            "point_bridge/robot_utils/mimiclabs/generate_pkl.py",
            "--task-name",
            "bowl_on_plate",
            "--task-names",
            task,
            "--demo-start",
            str(args.demo_start),
            "--demo-stop",
            str(args.demo_stop),
            "--max-accepted",
            str(args.max_accepted),
        ]
        shard = int(task.rsplit("_", 1)[1])
        audit_path = args.audit_dir / f"v0_success_audit_shard{shard}.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            accepted_indices = [
                int(record["demo_key"].rsplit("_", 1)[1])
                for record in audit["records"]
                if record["saved_final_success"]
                or record["saved_final_action_success"]
            ][: args.max_accepted]
            if len(accepted_indices) < args.max_accepted:
                parser.error(
                    f"{audit_path} has only {len(accepted_indices)} accepted demos"
                )
            command.extend(
                ["--demo-indices", ",".join(str(index) for index in accepted_indices)]
            )
        log_path = log_dir / f"{task}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=upstream,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((task, log_path, log_handle, process))
        print(f"started {task} pid={process.pid} cuda={device} log={log_path}")

    failed = False
    try:
        for task, log_path, log_handle, process in processes:
            returncode = process.wait()
            log_handle.close()
            print(f"finished {task} returncode={returncode} log={log_path}")
            failed = failed or returncode != 0
    except KeyboardInterrupt:
        for _, _, log_handle, process in processes:
            if process.poll() is None:
                process.terminate()
            log_handle.close()
        raise

    output_dir = upstream / "expert_demos" / "mimiclabs__no_images"
    for task in args.tasks:
        output = output_dir / f"{task}.pkl"
        if not output.exists():
            print(f"missing output {output}")
            failed = True
            continue
        with output.open("rb") as handle:
            data = pickle.load(handle)
        episodes = len(data["observations"])
        print(f"validated {task} episodes={episodes} output={output}")
        failed = failed or episodes < args.max_accepted
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
