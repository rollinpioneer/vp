#!/usr/bin/env python3
"""Capture RGB-D and stable GT point identities from a pinned MimicLabs replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPSTREAM = ROOT / "third_party" / "pointbridge"


def add_upstream_paths(upstream: Path) -> None:
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
    parser.add_argument("--shard", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--points-per-object", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-reset", action="store_true", help="Smoke-test current assets without restoring HDF5 XML")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "v1" / "capture_demo0.npz")
    args = parser.parse_args()
    add_upstream_paths(args.upstream)

    import robosuite as suite
    from point_bridge.robot_utils.common.mujoco_transforms import MujocoTransforms
    from point_bridge.robot_utils.common.utils import depthimg2Meters, farthest_point_sampling, transform_points
    from point_bridge.robot_utils.mimiclabs.sample_object_points import extract_object_mesh, sample_points_from_mesh
    from point_bridge.robot_utils.mimiclabs.utils import T_robot_base
    from mimiclabs.mimiclabs.envs.problems import MimicLabs_Lab1_Tabletop_Manipulation  # noqa: F401

    np.random.seed(args.seed)
    dataset = (
        args.upstream
        / "data"
        / "mimicgen_data"
        / "bowl_on_plate"
        / f"bowl_on_plate_{args.shard}"
        / "demo"
        / "demo.hdf5"
    )
    bddl = (
        args.upstream
        / "third_party"
        / "mimiclabs"
        / "mimiclabs"
        / "mimiclabs"
        / "task_suites"
        / "new_task_suite"
        / f"bowl_on_plate_{args.shard}.bddl"
    )
    with h5py.File(dataset, "r") as handle:
        demos = sorted(handle["data"].keys())
        demo_key = demos[args.demo_index]
        states = np.asarray(handle["data"][demo_key]["states"])
        actions = np.asarray(handle["data"][demo_key]["actions"])
        model_file = handle["data"][demo_key].attrs["model_file"]

    controller = suite.load_controller_config(default_controller="OSC_POSE")
    controller["control_delta"] = True
    env = suite.make(
        env_name="MimicLabs_Lab1_Tabletop_Manipulation",
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        control_freq=20,
        controller_configs=controller,
        robots=["Panda"],
        bddl_file_name=str(bddl),
    )
    env.reset()
    if not args.random_reset:
        env.reset_to({"states": states[0], "model": model_file})

    robot_base = np.eye(4)
    robot_base[:3, 3] = env.sim.data.get_body_xpos("robot0_base")
    robot_base[:3, :3] = env.sim.data.get_body_xmat("robot0_base")
    robot_base = robot_base @ T_robot_base

    object_local: dict[str, np.ndarray] = {}
    object_groups: list[list[str]] = []
    for object_name in env.obj_of_interest:
        bodies: list[str] = []
        for body_name in env.sim.model.body_names:
            if body_name is None or object_name not in body_name:
                continue
            try:
                if extract_object_mesh(env, body_name, visualize=False, in_base_frame=True) is not None:
                    bodies.append(body_name)
            except Exception:
                continue
        if bodies:
            object_groups.append(bodies)

    for bodies in object_groups:
        allocations = [args.points_per_object // len(bodies)] * len(bodies)
        allocations[-1] += args.points_per_object - sum(allocations)
        for body_name, count in zip(bodies, allocations):
            mesh = extract_object_mesh(env, body_name, visualize=False, in_base_frame=True)
            sampled = sample_points_from_mesh(mesh, n_points=max(1000, count), surface_only=True)
            sampled = farthest_point_sampling(sampled, count)
            sampled = transform_points(sampled, np.linalg.inv(T_robot_base))
            pose = np.eye(4)
            pose[:3, 3] = env.sim.data.get_body_xpos(body_name)
            pose[:3, :3] = env.sim.data.get_body_xmat(body_name)
            pose = np.linalg.inv(robot_base) @ pose
            object_local[body_name] = transform_points(sampled, np.linalg.inv(pose))

    rgb_frames: list[np.ndarray] = []
    depth_frames: list[np.ndarray] = []
    point_frames: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    extrinsics: list[np.ndarray] = []
    steps = min(args.max_steps, len(actions))
    final_reward = 0.0
    for step in range(steps):
        transforms = MujocoTransforms(env, [args.camera], args.height, args.width)
        camera_pose = transforms.transforms["camera_projection_matrix"][args.camera]
        intrinsic = transforms.camera_intrinsics[args.camera]
        target_to_camera = np.linalg.inv(camera_pose) @ robot_base
        rgb, raw_depth = env.sim.render(args.width, args.height, camera_name=args.camera, depth=True)
        rgb = rgb[::-1]
        depth_m = depthimg2Meters(env, raw_depth[::-1])

        points: list[np.ndarray] = []
        for bodies in object_groups:
            group: list[np.ndarray] = []
            for body_name in bodies:
                pose = np.eye(4)
                pose[:3, 3] = env.sim.data.get_body_xpos(body_name)
                pose[:3, :3] = env.sim.data.get_body_xmat(body_name)
                pose = np.linalg.inv(robot_base) @ pose
                group.append(transform_points(object_local[body_name], pose)[:, :3])
            points.append(np.concatenate(group, axis=0))

        rgb_frames.append(rgb)
        depth_frames.append(depth_m)
        point_frames.append(np.concatenate(points, axis=0))
        intrinsics.append(intrinsic)
        extrinsics.append(target_to_camera)
        _, final_reward, _, _ = env.step(actions[step])
    env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        rgb=np.asarray(rgb_frames),
        depth_m=np.asarray(depth_frames),
        gt_points=np.asarray(point_frames),
        intrinsic=np.asarray(intrinsics),
        target_to_camera=np.asarray(extrinsics),
    )
    metadata = {
        "upstream_commit": "5d567a62d62b5a97c5960d45024e065349680cda",
        "dataset": str(dataset),
        "demo_key": demo_key,
        "recorded_state_restored": not args.random_reset,
        "steps": steps,
        "camera": args.camera,
        "point_count": int(np.asarray(point_frames).shape[1]),
        "expert_replay_final_reward": float(final_reward),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
