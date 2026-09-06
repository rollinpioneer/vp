"""Causal task-point belief updates for V1 diagnostics and V2 extension."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import inf

from vico_point.core.types import ObservationState, Point, XYZ


@dataclass(frozen=True)
class TaskBelief:
    point_id: str
    role: str
    mean: XYZ | None = None
    variance: XYZ = (inf, inf, inf)
    last_seen: float | None = None
    age: float = inf
    initialized: bool = False
    state: ObservationState = ObservationState.UNKNOWN
    update_source: str = "none"


def update_belief(
    belief: TaskBelief,
    observation: Point | None,
    *,
    timestamp: float,
    hold_last_observation: bool = True,
) -> TaskBelief:
    """Update only from current or past observations; never uses future data."""

    if observation is not None and observation.valid and observation.observed:
        return replace(
            belief,
            mean=observation.xyz,
            variance=(0.0, 0.0, 0.0),
            last_seen=timestamp,
            age=0.0,
            initialized=True,
            state=ObservationState.OBSERVED,
            update_source="current_observation",
        )
    if hold_last_observation and belief.initialized:
        age = inf if belief.last_seen is None else max(0.0, timestamp - belief.last_seen)
        return replace(
            belief,
            age=age,
            state=ObservationState.PREDICTED,
            update_source="causal_hold",
        )
    return replace(belief, age=inf, state=ObservationState.UNKNOWN, update_source="missing")
