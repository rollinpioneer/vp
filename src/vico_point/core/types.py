"""Stable data contracts shared by perception, policy and evaluation code."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional, Tuple

XYZ = Tuple[float, float, float]


class InputLevel(str, Enum):
    O = "O"  # full-state oracle; diagnostics only
    P = "P"  # visible depth with simulator role labels
    V = "V"  # causal visual extraction


class ObservationState(str, Enum):
    OBSERVED = "observed"
    PREDICTED = "predicted"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Point:
    point_id: str
    xyz: XYZ
    role: str
    valid: bool = True
    observed: bool = True
    confidence: float = 1.0
    timestamp: float = 0.0
    state: ObservationState = ObservationState.OBSERVED

    def __post_init__(self) -> None:
        if len(self.xyz) != 3:
            raise ValueError("xyz must have exactly three coordinates")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.state == ObservationState.OBSERVED and not self.observed:
            raise ValueError("observed points must have observed=True")


@dataclass(frozen=True)
class CameraCalibration:
    """Calibration is explicit so depth units and frame transforms are not implicit."""

    frame: str = "camera"
    target_frame: str = "robot_base"
    depth_unit: str = "meter"
    camera_to_target: Tuple[Tuple[float, ...], ...] = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


@dataclass(frozen=True)
class PointFrame:
    frame_id: str
    timestamp: float
    input_level: InputLevel
    robot_points: Tuple[Point, ...] = ()
    task_points: Tuple[Point, ...] = ()
    context_points: Tuple[Point, ...] = ()
    calibration: CameraCalibration = field(default_factory=CameraCalibration)

    def all_points(self) -> Tuple[Point, ...]:
        return self.robot_points + self.task_points + self.context_points

    def valid_points(self) -> Tuple[Point, ...]:
        return tuple(point for point in self.all_points() if point.valid)

    def count_by_group(self) -> dict[str, int]:
        return {
            "robot": len(self.robot_points),
            "task": len(self.task_points),
            "context": len(self.context_points),
            "total": len(self.all_points()),
        }


@dataclass(frozen=True)
class PointBudget:
    total: int = 64
    robot: int = 9
    task: int = 16
    context: int = 39

    def __post_init__(self) -> None:
        if min(self.total, self.robot, self.task, self.context) < 0:
            raise ValueError("point budget values cannot be negative")
        if self.robot + self.task + self.context != self.total:
            raise ValueError("robot + task + context must equal total")


@dataclass(frozen=True)
class ActionProtocol:
    action_dim: int = 7
    prediction_horizon: int = 1
    execution_horizon: int = 1
    aggregation: str = "first_action"
    position_unit: str = "meter"
    rotation_unit: str = "radian"


@dataclass(frozen=True)
class RunMetadata:
    upstream_commit: str
    own_commit: str
    input_level: InputLevel
    point_budget: PointBudget
    history_frames: int
    action_protocol: ActionProtocol
    data_split: str
    hardware: str


def ensure_unique_ids(points: Iterable[Point]) -> None:
    ids = [point.point_id for point in points]
    if len(ids) != len(set(ids)):
        raise ValueError("point IDs must be unique within a frame")
