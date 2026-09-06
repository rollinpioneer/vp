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
        source: list[str] = []
        for index, is_visible in enumerate(visibility.visible_mask):
            if oracle_hidden_truth:
                self._held[index] = points[index]
                self._age[index] = 0.0
                source.append("oracle_gt_visible" if is_visible else "oracle_gt_hidden")
            elif is_visible:
                self._held[index] = points[index]
                self._age[index] = 0.0
                source.append("current_visible_depth")
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
        forbidden = {"oracle_gt_hidden", "oracle_gt_visible"}
        if forbidden.intersection(observation.source):
            raise RuntimeError("non-oracle policy observation contains simulator truth")
