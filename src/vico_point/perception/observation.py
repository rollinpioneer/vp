"""Construction helpers for causal visible point frames.

The module deliberately accepts already extracted points. RGB-D extraction and
tracking belong behind this boundary and can be supplied by Point Bridge later.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from vico_point.core.types import (
    CameraCalibration,
    InputLevel,
    ObservationState,
    Point,
    PointFrame,
    XYZ,
)


def make_point(
    point_id: str,
    xyz: XYZ,
    role: str,
    *,
    timestamp: float,
    observed: bool = True,
    confidence: float = 1.0,
    valid: bool | None = None,
) -> Point:
    if valid is None:
        valid = observed
    state = ObservationState.OBSERVED if observed else ObservationState.UNKNOWN
    return Point(
        point_id=point_id,
        xyz=xyz,
        role=role,
        valid=valid,
        observed=observed,
        confidence=confidence,
        timestamp=timestamp,
        state=state,
    )


def build_frame(
    frame_id: str,
    timestamp: float,
    input_level: InputLevel,
    *,
    robot_points: Iterable[Point] = (),
    task_points: Iterable[Point] = (),
    context_points: Iterable[Point] = (),
    calibration: CameraCalibration | None = None,
) -> PointFrame:
    """Build one frame without filling missing observations with fake zeros."""

    return PointFrame(
        frame_id=frame_id,
        timestamp=timestamp,
        input_level=input_level,
        robot_points=tuple(robot_points),
        task_points=tuple(task_points),
        context_points=tuple(context_points),
        calibration=calibration or CameraCalibration(),
    )


def extract_role_points(
    detections: Iterable[Mapping[str, object]], *, timestamp: float
) -> tuple[Point, ...]:
    """Convert a causal detector output into stable role-labelled points."""

    points = []
    for detection in detections:
        points.append(
            make_point(
                str(detection["point_id"]),
                tuple(float(value) for value in detection["xyz"]),  # type: ignore[arg-type]
                str(detection["role"]),
                timestamp=timestamp,
                observed=bool(detection.get("observed", True)),
                confidence=float(detection.get("confidence", 1.0)),
            )
        )
    return tuple(points)
