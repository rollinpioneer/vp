"""Fail-fast checks for the V0 frozen interfaces."""

from __future__ import annotations

from dataclasses import asdict

from .types import PointBudget, PointFrame, RunMetadata, ensure_unique_ids


def validate_frame(frame: PointFrame, budget: PointBudget | None = None) -> None:
    points = frame.all_points()
    ensure_unique_ids(points)
    if frame.calibration.depth_unit not in {"meter", "millimeter"}:
        raise ValueError("depth_unit must be 'meter' or 'millimeter'")
    for point in points:
        if any(value != value for value in point.xyz):
            raise ValueError(f"point {point.point_id} contains NaN")
        if point.valid and point.state.value == "unknown":
            raise ValueError(f"valid point {point.point_id} cannot have unknown state")
    if budget is not None:
        counts = frame.count_by_group()
        expected = {"robot": budget.robot, "task": budget.task, "context": budget.context}
        for group, expected_count in expected.items():
            if counts[group] > expected_count:
                raise ValueError(f"{group} points exceed frozen budget")


def validate_metadata(metadata: RunMetadata) -> None:
    if len(metadata.upstream_commit) != 40:
        raise ValueError("upstream_commit must be a full 40-character commit SHA")
    if metadata.history_frames < 0:
        raise ValueError("history_frames cannot be negative")
    if not metadata.data_split:
        raise ValueError("data_split is required")
    if not metadata.hardware:
        raise ValueError("hardware is required")
    # Force recursive dataclass validation to fail early if fields are malformed.
    asdict(metadata)
