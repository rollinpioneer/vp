"""Causal segmentation-plus-depth point extraction with explicit fallback."""

from __future__ import annotations

import numpy as np

from .evidence import PointEvidence, PointSource


def _unproject(pixel_xy: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(np.asarray(intrinsic, dtype=np.float64))
    homogeneous = np.column_stack((pixel_xy, np.ones(len(pixel_xy))))
    return (inverse @ homogeneous.T).T * depth[:, None]


def extract_segment_depth_evidence(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_target: np.ndarray,
    *,
    point_ids: tuple[str, ...],
    timestamp_seconds: float,
    previous: PointEvidence | None = None,
    depth_min_m: float = 1e-5,
) -> PointEvidence:
    """Extract deterministic points from one current mask/depth frame.

    Missing masks, invalid depth, and insufficient samples are represented as
    ``previous_fallback`` or ``unknown`` rather than as current observations.
    """

    mask = np.asarray(mask, dtype=bool)
    depth_m = np.asarray(depth_m, dtype=np.float64)
    if mask.ndim != 2 or depth_m.shape != mask.shape:
        raise ValueError("mask and depth_m must have the same 2-D shape")
    n = len(point_ids)
    xyz = np.full((n, 3), np.nan, dtype=np.float64)
    visible = np.zeros(n, dtype=bool)
    confidence = np.full(n, np.nan, dtype=np.float64)
    age = np.full(n, np.inf, dtype=np.float64)
    source = [PointSource.UNKNOWN] * n
    camera_mask = np.zeros((n, 1), dtype=bool)
    pixels = np.column_stack(np.nonzero(mask))[:, ::-1] if mask.any() else np.empty((0, 2))
    if len(pixels):
        indices = np.linspace(0, len(pixels) - 1, n).round().astype(int)
        pixels = pixels[indices]
        depths = depth_m[pixels[:, 1], pixels[:, 0]]
        valid = np.isfinite(depths) & (depths > depth_min_m)
        if valid.any():
            camera_points = _unproject(pixels[valid], depths[valid], intrinsic)
            homogeneous = np.column_stack((camera_points, np.ones(len(camera_points))))
            target_points = (np.asarray(camera_to_target, dtype=np.float64) @ homogeneous.T).T[:, :3]
            valid_indices = np.flatnonzero(valid)
            xyz[valid_indices] = target_points
            visible[valid_indices] = True
            confidence[valid_indices] = 1.0
            age[valid_indices] = 0.0
            camera_mask[valid_indices, 0] = True
            for index in valid_indices:
                source[index] = PointSource.SEGMENT_CURRENT

    if previous is not None:
        if previous.point_ids != point_ids:
            raise ValueError("previous evidence point IDs do not match")
        missing = ~visible
        previous_xyz = np.asarray(previous.xyz)
        usable_previous = missing & np.isfinite(previous_xyz).all(axis=1)
        xyz[usable_previous] = previous_xyz[usable_previous]
        age[usable_previous] = np.maximum(
            0.0, float(timestamp_seconds) - previous.timestamp_seconds + previous.age_seconds[usable_previous]
        )
        confidence[usable_previous] = previous.confidence[usable_previous]
        source = [
            PointSource.PREVIOUS_FALLBACK if usable_previous[index] else source[index]
            for index in range(n)
        ]

    evidence = PointEvidence(
        xyz=xyz,
        visible=visible,
        confidence=confidence,
        age_seconds=age,
        source=tuple(source),
        timestamp_seconds=float(timestamp_seconds),
        camera_mask=camera_mask,
        point_ids=point_ids,
    )
    evidence.validate()
    return evidence


__all__ = ["extract_segment_depth_evidence"]
