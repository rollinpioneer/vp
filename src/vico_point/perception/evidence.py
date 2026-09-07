"""Auditable point evidence shared by perception and policy adapters.

The schema separates the point coordinates from the evidence used to obtain
them.  In particular, a held or filled point must never be serialized as a
current observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class PointSource(str, Enum):
    GT_CURRENT = "gt_current"
    DEPTH_CURRENT = "depth_current"
    SEGMENT_CURRENT = "segment_current"
    TRACK_CURRENT = "track_current"
    PREVIOUS_FALLBACK = "previous_fallback"
    LAST_RELIABLE_HOLD = "last_reliable_hold"
    RIGID_PROPAGATION = "rigid_propagation"
    MOTION_PREDICTION = "motion_prediction"
    GROUP_CENTROID_FILL = "group_centroid_fill"
    UNKNOWN = "unknown"
    ORACLE_HIDDEN_GT = "oracle_hidden_gt"


@dataclass(frozen=True)
class PointEvidence:
    """One causal point frame in the robot-base coordinate system."""

    xyz: np.ndarray
    visible: np.ndarray
    confidence: np.ndarray
    age_seconds: np.ndarray
    source: tuple[PointSource, ...]
    timestamp_seconds: float
    camera_mask: np.ndarray
    point_ids: tuple[str, ...]

    def validate(self) -> None:
        n = len(self.point_ids)
        if self.xyz.shape != (n, 3):
            raise ValueError(f"xyz shape mismatch: {self.xyz.shape}, expected {(n, 3)}")
        for name, value in (
            ("visible", self.visible),
            ("confidence", self.confidence),
            ("age_seconds", self.age_seconds),
        ):
            if value.shape != (n,):
                raise ValueError(f"{name} shape mismatch: {value.shape}")
        if len(self.source) != n:
            raise ValueError("source length mismatch")
        if self.camera_mask.ndim != 2 or self.camera_mask.shape[0] != n:
            raise ValueError("camera_mask shape mismatch")
        if not np.isfinite(self.timestamp_seconds):
            raise ValueError("timestamp_seconds must be finite")
        if np.any(self.age_seconds < 0):
            raise ValueError("negative point age")
        finite_confidence = np.isfinite(self.confidence)
        if np.any((self.confidence[finite_confidence] < 0) | (self.confidence[finite_confidence] > 1)):
            raise ValueError("finite confidence values must be in [0, 1]")
        if np.any(self.visible & ~np.isfinite(self.xyz).all(axis=1)):
            raise ValueError("visible points must have finite coordinates")
        if any(not isinstance(item, PointSource) for item in self.source):
            raise ValueError("source entries must be PointSource values")


__all__ = ["PointEvidence", "PointSource"]
