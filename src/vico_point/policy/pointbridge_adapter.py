"""Leakage-safe observation adapter for a pinned Point Bridge checkout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vico_point.envs.visibility import OcclusionWindow, VisibilityResult, visible_points_from_depth


@dataclass(frozen=True)
class AdaptedPointObservation:
    policy_points: np.ndarray
    visible_mask: np.ndarray
    age_steps: np.ndarray
    source: tuple[str, ...]
    visibility: VisibilityResult


class CausalPointBridgeAdapter:
    """Keep Point Bridge tensors fixed-size without filling from hidden current GT."""

    def __init__(self, point_ids: tuple[str, ...], *, control_hz: float = 20.0) -> None:
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("point_ids must be unique")
        self.point_ids = point_ids
        self.control_hz = float(control_hz)
        self._held = np.full((len(point_ids), 3), np.nan, dtype=np.float64)
        self._age = np.full(len(point_ids), np.inf, dtype=np.float64)

    def reset(self) -> None:
        self._held[:] = np.nan
        self._age[:] = np.inf

    def adapt(
        self,
        gt_points: np.ndarray,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        intrinsic: np.ndarray,
        target_to_camera: np.ndarray,
        *,
        step: int,
        condition: str,
        window: OcclusionWindow | None = None,
        oracle_hidden_truth: bool = False,
    ) -> AdaptedPointObservation:
        points = np.asarray(gt_points, dtype=np.float64)
        if points.shape != self._held.shape:
            raise ValueError(f"expected GT point shape {self._held.shape}, got {points.shape}")
        if condition not in {"E00", "E10"}:
            raise ValueError("real visibility gate supports E00 and E10 only")
        applied_window = window if condition == "E10" else None
        visibility = visible_points_from_depth(
            points,
            rgb,
            depth_m,
            intrinsic,
            target_to_camera,
            window=applied_window,
            step=step,
        )
        baseline_visible = visibility.visible_mask
        if oracle_hidden_truth and applied_window is not None and applied_window.active(step):
            baseline_visible = visible_points_from_depth(
                points,
                rgb,
                depth_m,
                intrinsic,
                target_to_camera,
                step=step,
            ).visible_mask
        synthetic_hidden = baseline_visible & ~visibility.visible_mask
        source: list[str] = []
        for index, is_visible in enumerate(visibility.visible_mask):
            if is_visible:
                self._held[index] = points[index]
                self._age[index] = 0.0
                source.append("current_visible_depth")
            elif oracle_hidden_truth and synthetic_hidden[index]:
                self._held[index] = points[index]
                self._age[index] = 0.0
                source.append("oracle_gt_hidden")
            elif np.isfinite(self._held[index]).all():
                self._age[index] += 1.0
                source.append("causal_hold")
            else:
                source.append("unknown")
        return AdaptedPointObservation(
            policy_points=np.array(self._held, copy=True),
            visible_mask=np.array(visibility.visible_mask, copy=True),
            age_steps=np.array(self._age, copy=True),
            source=tuple(source),
            visibility=visibility,
        )

    def audit_no_hidden_truth(self, observation: AdaptedPointObservation) -> None:
        forbidden = {"oracle_gt_hidden"}
        if forbidden.intersection(observation.source):
            raise RuntimeError("non-oracle policy observation contains simulator truth")


def fill_unknown_from_visible_group_centroids(
    points: np.ndarray,
    visible_mask: np.ndarray,
    *,
    group_size: int,
) -> tuple[np.ndarray, int]:
    """Replace non-finite policy points without consulting hidden current truth.

    Point Bridge has no missing-point mask and cannot consume NaNs. Each missing
    point is therefore replaced by the centroid of currently visible points in
    the same object group. A fully hidden, never-observed group uses zeros.
    """

    values = np.asarray(points, dtype=np.float64)
    visible = np.asarray(visible_mask, dtype=bool)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if visible.shape != (len(values),):
        raise ValueError("visible_mask must have shape (N,)")
    if group_size <= 0 or len(values) % group_size:
        raise ValueError("group_size must evenly divide the point count")

    output = np.array(values, copy=True)
    filled = 0
    for start in range(0, len(output), group_size):
        stop = start + group_size
        group = output[start:stop]
        group_visible = visible[start:stop] & np.isfinite(group).all(axis=1)
        missing = ~np.isfinite(group).all(axis=1)
        if not missing.any():
            continue
        centroid = (
            np.mean(group[group_visible], axis=0)
            if group_visible.any()
            else np.zeros(3, dtype=np.float64)
        )
        group[missing] = centroid
        filled += int(missing.sum())
    return output, filled
