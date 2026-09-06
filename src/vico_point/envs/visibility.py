"""Camera visibility and synchronized RGB-D occlusion utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OcclusionWindow:
    start_step: int
    duration_steps: int
    box_xyxy: tuple[float, float, float, float] = (0.25, 0.2, 0.75, 0.8)
    depth_m: float = 0.25
    rgb: tuple[int, int, int] = (24, 24, 24)

    def active(self, step: int) -> bool:
        return self.start_step <= step < self.start_step + self.duration_steps


@dataclass(frozen=True)
class VisibilityResult:
    rgb: np.ndarray
    depth_m: np.ndarray
    visible_mask: np.ndarray
    pixels_xy: np.ndarray
    point_depth_m: np.ndarray
    occlusion_active: bool


def project_points(
    points_target: np.ndarray,
    intrinsic: np.ndarray,
    target_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project target-frame XYZ points to image pixels and camera depth."""

    points = np.asarray(points_target, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_target must have shape (N, 3)")
    homogeneous = np.column_stack((points, np.ones(len(points))))
    camera = (np.asarray(target_to_camera, dtype=np.float64) @ homogeneous.T).T[:, :3]
    depth = camera[:, 2]
    pixels_h = (np.asarray(intrinsic, dtype=np.float64) @ camera.T).T
    safe_depth = np.where(np.abs(depth) > 1e-12, depth, 1.0)
    pixels = pixels_h[:, :2] / safe_depth[:, None]
    return pixels, depth


def _pixel_box(shape: tuple[int, int], box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    height, width = shape
    x0, y0, x1, y1 = box
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError("occlusion box must be normalized xyxy coordinates")
    return int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)


def apply_synchronized_occlusion(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    window: OcclusionWindow,
    *,
    step: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Apply the same predeclared image-plane occluder to RGB and metric depth."""

    rgb_out = np.array(rgb, copy=True)
    depth_out = np.array(depth_m, copy=True)
    if not window.active(step):
        return rgb_out, depth_out, False
    x0, y0, x1, y1 = _pixel_box(depth_out.shape, window.box_xyxy)
    rgb_out[y0:y1, x0:x1] = np.asarray(window.rgb, dtype=rgb_out.dtype)
    depth_out[y0:y1, x0:x1] = window.depth_m
    return rgb_out, depth_out, True


def visible_points_from_depth(
    points_target: np.ndarray,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    target_to_camera: np.ndarray,
    *,
    tolerance_m: float = 0.015,
    window: OcclusionWindow | None = None,
    step: int = 0,
) -> VisibilityResult:
    """Return points whose projection agrees with the current depth image."""

    if depth_m.ndim != 2 or rgb.shape[:2] != depth_m.shape:
        raise ValueError("RGB and depth must have identical image dimensions")
    rgb_used, depth_used, active = (
        apply_synchronized_occlusion(rgb, depth_m, window, step=step)
        if window is not None
        else (np.array(rgb, copy=True), np.array(depth_m, copy=True), False)
    )
    pixels, point_depth = project_points(points_target, intrinsic, target_to_camera)
    height, width = depth_used.shape
    rounded = np.rint(pixels).astype(np.int64)
    in_view = (
        np.isfinite(pixels).all(axis=1)
        & np.isfinite(point_depth)
        & (point_depth > 0.0)
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    sampled = np.full(len(point_depth), np.nan, dtype=np.float64)
    sampled[in_view] = depth_used[rounded[in_view, 1], rounded[in_view, 0]]
    visible = in_view & np.isfinite(sampled) & (sampled > 0.0) & (np.abs(sampled - point_depth) <= tolerance_m)
    return VisibilityResult(rgb_used, depth_used, visible, pixels, point_depth, active)
