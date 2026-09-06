#!/usr/bin/env python3
"""Compare saved-state success with action replay for MimicLabs demos."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM = ROOT / "third_party" / "pointbridge"
sys.path.insert(0, str(ROOT / "src"))

from vico_point.envs.mimiclabs_compat import migrate_saved_model_xml


def _indices(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _add_upstream_paths(upstream: Path) -> None:
    paths = (
        upstream,
        upstream / "third_party" / "LIBERO",
        upstream / "third_party" / "mimicgen",
        upstream / "third_party" / "mimiclabs",
        upstream / "third_party" / "robocasa",
    )
    for path in reversed(paths):
        sys.path.insert(0, str(path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--shards", type=_indices, default=[1, 2, 3, 4])
    parser.add_argument("--demo-indices", type=_indices)
    parser.add_argument("--demo-start", type=int, default=0)
    parser.add_argument("--demo-stop", type=int)
    parser.add_argument("--saved-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    upstream = args.upstream.resolve()
    os.environ.setdefault("MUJOCO_GL", "egl")
    _add_upstream_paths(upstream)

    import robosuite as suite
    from mimiclabs.mimiclabs.envs.problems import (  # noqa: F401
        MimicLabs_Lab1_Tabletop_Manipulation,
    )

    controller = suite.load_controller_config(default_controller="OSC_POSE")
    controller["control_delta"] = True
    records: list[dict[str, object]] = []
    for shard in args.shards:
        task_name = f"bowl_on_plate_{shard}"
        dataset = (
            upstream
            / "data"
            / "mimicgen_data"
            / "bowl_on_plate"
            / task_name
            / "demo"
            / "demo.hdf5"
        )
        bddl = (
            upstream
            / "third_party"
            / "mimiclabs"
            / "mimiclabs"
            / "mimiclabs"
            / "task_suites"
            / "new_task_suite"
            / f"{task_name}.bddl"
        )
        env = suite.make(
            env_name="MimicLabs_Lab1_Tabletop_Manipulation",
            has_renderer=False,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=False,
            control_freq=20,
            controller_configs=controller,
            robots=["Panda"],
            bddl_file_name=str(bddl),
        )
        env.reset()
        with h5py.File(dataset, "r") as handle:
            demos = sorted(
                handle["data"].keys(), key=lambda key: int(key.rsplit("_", 1)[1])
            )
            demo_indices = (
                args.demo_indices
                if args.demo_indices is not None
                else list(range(args.demo_start, args.demo_stop or len(demos)))
            )
            for demo_index in demo_indices:
                demo_key = demos[demo_index]
                demo = handle["data"][demo_key]
                states = np.asarray(demo["states"])
                actions = np.asarray(demo["actions"])
                model_xml = migrate_saved_model_xml(demo.attrs["model_file"])

                env.reset_to({"states": states[-1], "model": model_xml})
                saved_final_success = bool(env._check_success())
                _, saved_final_reward, _, _ = env.step(actions[-1])
                saved_final_action_success = bool(env._check_success())

                generator_reward = generator_success = None
                full_replay_reward = full_replay_success = None
                if not args.saved_only:
                    env.reset_to({"states": states[0], "model": model_xml})
                    generator_reward = float(env.reward())
                    for action in actions[:-1]:
                        _, generator_reward, _, _ = env.step(action)
                    generator_success = bool(env._check_success())

                    env.reset_to({"states": states[0], "model": model_xml})
                    full_replay_reward = float(env.reward())
                    for action in actions:
                        _, full_replay_reward, _, _ = env.step(action)
                    full_replay_success = bool(env._check_success())

                record = {
                    "shard": shard,
                    "demo_key": demo_key,
                    "steps": len(actions),
                    "saved_final_success": saved_final_success,
                    "saved_final_action_success": saved_final_action_success,
                    "saved_final_action_reward": float(saved_final_reward),
                    "generator_replay_success": generator_success,
                    "generator_replay_reward": generator_reward,
                    "full_replay_success": full_replay_success,
                    "full_replay_reward": full_replay_reward,
                }
                records.append(record)
                if not args.quiet:
                    print(json.dumps(record, sort_keys=True))
        env.close()

    accepted = [
        record
        for record in records
        if record["saved_final_success"] or record["saved_final_action_success"]
    ]
    result = {
        "summary": {
            "checked": len(records),
            "accepted": len(accepted),
            "acceptance_rate": len(accepted) / len(records) if records else 0.0,
        },
        "records": records,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
