#!/usr/bin/env python3
"""Shared helpers for V1-R frozen simulator-state audits.

The state index is intentionally a small, tracked CSV while the actual state
arrays live below ``outputs/`` and remain gitignored.  A state entry may also
carry the sampled object points used by Point Bridge so that restoring a state
does not silently resample a different point cloud.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def array_sha256(value: Any) -> str:
    """Hash shape, dtype and contiguous bytes for an array-like value."""

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def raw_array_sha256(value: Any) -> str:
    """Hash only bytes, matching the legacy runner's simulator-state hash."""

    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state_index(path: Path) -> dict[str, dict[str, str]]:
    """Load and validate a state index keyed by scenario id."""

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "scenario_id",
        "layout",
        "simulator_seed",
        "initial_state_key",
        "state_path",
        "state_sha256",
        "restored_state_sha256",
    }
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"state index {path} is missing columns: {sorted(missing)}")
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        scenario_id = row["scenario_id"]
        if scenario_id in result:
            raise ValueError(f"duplicate state index scenario_id: {scenario_id}")
        result[scenario_id] = row
    return result


def resolve_state_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def load_state_bundle(repo_root: Path, row: dict[str, str]) -> dict[str, Any]:
    """Read a state bundle and return its simulator state and object points."""

    path = resolve_state_path(repo_root, row["state_path"])
    if not path.is_file():
        raise FileNotFoundError(f"frozen state file does not exist: {path}")
    with np.load(path, allow_pickle=False) as bundle:
        if "sim_state" not in bundle:
            raise ValueError(f"state bundle {path} has no sim_state array")
        state = np.asarray(bundle["sim_state"]).copy()
        object_points: dict[str, np.ndarray] = {}
        if "object_point_keys" in bundle:
            keys = [str(item) for item in np.asarray(bundle["object_point_keys"]).tolist()]
            for index, key in enumerate(keys):
                object_points[key] = np.asarray(bundle[f"object_points_{index}"]).copy()
    actual_hash = raw_array_sha256(state)
    if actual_hash != row["state_sha256"]:
        raise ValueError(
            f"state hash mismatch for {row['scenario_id']}: "
            f"index={row['state_sha256']} file={actual_hash}"
        )
    return {
        "path": path,
        "sim_state": state,
        "object_points": object_points,
        "state_sha256": actual_hash,
        "restored_state_sha256": row["restored_state_sha256"],
        "file_sha256": file_sha256(path),
    }


def body_state_hash(env: Any) -> str:
    """Hash simulator qpos/qvel and body transforms for parity diagnostics."""

    sim = env.sim
    values = [
        np.asarray(sim.data.qpos),
        np.asarray(sim.data.qvel),
        np.asarray(sim.data.xpos),
        np.asarray(sim.data.xmat),
        np.asarray(sim.data.xquat),
    ]
    return array_sha256(np.concatenate([value.reshape(-1) for value in values]))


def robot_pose_hash(env: Any) -> str:
    """Hash the end-effector pose used by the point observation."""

    body_id = env.sim.model.body_name2id("gripper0_eef")
    values = [env.sim.data.xpos[body_id], env.sim.data.xmat[body_id]]
    return array_sha256(np.concatenate([np.asarray(value).reshape(-1) for value in values]))


def pointbridge_core_env(env: Any) -> Any:
    """Return the RGBArrayAsObservationWrapper below the dm_env wrappers."""

    core = env
    while hasattr(core, "_env"):
        if "_pixel_keys" in vars(core) and callable(
            getattr(type(core), "get_gt_points", None)
        ):
            return core
        core = core._env
    raise RuntimeError("could not locate Point Bridge RGB observation wrapper")


def refresh_pointbridge_observation(
    env: Any,
    time_step: Any,
    object_points: dict[str, np.ndarray] | None,
    gripper_state: float = -1.0,
) -> Any:
    """Refresh a wrapper observation after setting a frozen state without stepping.

    Point Bridge's wrapper performs ten controller warm-up steps inside reset.
    The frozen state is captured after that warm-up.  Calling ``reset(state=...)``
    would warm it up a second time, so this function reconstructs the wrapper's
    observation at the restored state while preserving the simulator hash.
    """

    from point_bridge.robot_utils.common.utils import matrix_to_rotation_6d
    from point_bridge.robot_utils.mimiclabs.utils import T_gripper

    core = env
    frame_stack = None
    while hasattr(core, "_env"):
        if "_pixel_keys" in vars(core) and callable(
            getattr(type(core), "get_gt_points", None)
        ):
            break
        if "_frames" in vars(core) and "pixel_keys" in vars(core):
            frame_stack = core
        core = core._env
    if not hasattr(core, "get_gt_points"):
        raise RuntimeError("could not locate Point Bridge RGB observation wrapper")

    core._step = 0
    core.prev_gripper_state = float(gripper_state)
    if object_points:
        core.object_points = {
            key: np.array(value, copy=True) for key, value in object_points.items()
        }
        core.fixed_points = True
    else:
        core.object_points = {}
        core.fixed_points = False

    observation: dict[str, Any] = {"task_emb": core.task_emb, "goal_achieved": False}
    for key in core._pixel_keys:
        camera_name = core._pixelkey2camera[key]
        observation[key] = core._env.sim.render(core._width, core._height, camera_name=camera_name)[::-1]

    position = core._env.sim.data.get_body_xpos(core._bodynames[0])
    orientation = core._env.sim.data.get_body_xmat(core._bodynames[0])
    transform = np.eye(4)
    transform[:3, 3], transform[:3, :3] = position, orientation
    transform = transform @ T_gripper
    transform = np.linalg.inv(core.robot_base) @ transform
    position, orientation = transform[:3, 3], transform[:3, :3]
    core.robot_base_orientation = orientation
    orientation_6d = matrix_to_rotation_6d(orientation)
    core._current_pose = np.concatenate(
        [position, orientation_6d, [float(gripper_state)]]
    )
    observation["proprioceptive"] = core._current_pose
    observation["features"] = core._current_pose

    if "points" in core._obs_type:
        points3d_robot, points3d_objects = core.get_gt_points()
        observation[f"{core._robot_points_key}_3d"] = points3d_robot
        observation[f"{core._object_points_key}_3d"] = points3d_objects
    core.observation = observation

    if frame_stack is not None:
        for key in frame_stack.pixel_keys:
            frame_stack._frames[key].clear()
            frame_stack._frames[key].append(observation[key].transpose(2, 0, 1))
        stacked = dict(observation)
        for key in frame_stack.pixel_keys:
            stacked[key] = np.concatenate(list(frame_stack._frames[key]), axis=0)
        return time_step._replace(observation=stacked)
    return time_step._replace(observation=observation)
