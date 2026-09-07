"""Small, dependency-light adapter for auditable point-tracking outputs.

The upstream tracker can be imported by a caller, but this module owns the
conversion from per-camera tracks and visibility probabilities to the common
PointEvidence schema. It never consumes future frames.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .evidence import PointEvidence, PointSource


def _projection_matrix(intrinsic: np.ndarray, camera_to_target: np.ndarray) -> np.ndarray:
    target_to_camera = np.linalg.inv(np.asarray(camera_to_target, dtype=np.float64))
    return np.asarray(intrinsic, dtype=np.float64) @ target_to_camera[:3, :]


def triangulate_points(
    tracks_by_camera: Mapping[str, np.ndarray],
    visibility_by_camera: Mapping[str, np.ndarray],
    intrinsics: Mapping[str, np.ndarray],
    camera_to_target: Mapping[str, np.ndarray],
    *,
    point_ids: tuple[str, ...],
    timestamp_seconds: float,
    visibility_threshold: float = 0.5,
) -> PointEvidence:
    """Triangulate current 2-D tracks and preserve visibility evidence.

    A point with one supporting camera is marked unknown unless the caller
    supplies a depth-based implementation separately. This avoids silently
    treating a monocular pixel as a metric 3-D observation.
    """

    cameras = tuple(sorted(tracks_by_camera))
    n = len(point_ids)
    xyz = np.full((n, 3), np.nan, dtype=np.float64)
    visible = np.zeros(n, dtype=bool)
    confidence = np.full(n, np.nan, dtype=np.float64)
    camera_mask = np.zeros((n, len(cameras)), dtype=bool)
    source = [PointSource.UNKNOWN] * n
    for camera_index, camera in enumerate(cameras):
        tracks = np.asarray(tracks_by_camera[camera], dtype=np.float64)
        visibility = np.asarray(visibility_by_camera[camera], dtype=np.float64)
        if tracks.shape != (n, 2) or visibility.shape != (n,):
            raise ValueError(f"camera {camera} shape mismatch")
        camera_mask[:, camera_index] = np.isfinite(tracks).all(axis=1) & (visibility >= visibility_threshold)

    projections = {
        camera: _projection_matrix(intrinsics[camera], camera_to_target[camera])
        for camera in cameras
    }
    for point_index in range(n):
        supporting = [index for index, camera in enumerate(cameras) if camera_mask[point_index, index]]
        if len(supporting) < 2:
            continue
        equations = []
        for camera_index in supporting:
            camera = cameras[camera_index]
            u, v = np.asarray(tracks_by_camera[camera])[point_index]
            projection = projections[camera]
            equations.extend((u * projection[2] - projection[0], v * projection[2] - projection[1]))
        _, _, vh = np.linalg.svd(np.asarray(equations), full_matrices=False)
        homogeneous = vh[-1]
        if abs(homogeneous[3]) < 1e-10:
            continue
        candidate = homogeneous[:3] / homogeneous[3]
        if not np.isfinite(candidate).all():
            continue
        xyz[point_index] = candidate
        visible[point_index] = True
        probabilities = [float(np.asarray(visibility_by_camera[cameras[i]])[point_index]) for i in supporting]
        confidence[point_index] = float(np.mean(probabilities))
        source[point_index] = PointSource.TRACK_CURRENT
    evidence = PointEvidence(
        xyz=xyz,
        visible=visible,
        confidence=confidence,
        age_seconds=np.where(visible, 0.0, np.inf),
        source=tuple(source),
        timestamp_seconds=float(timestamp_seconds),
        camera_mask=camera_mask,
        point_ids=point_ids,
    )
    evidence.validate()
    return evidence


__all__ = ["triangulate_points"]
