"""Causal task-point belief updates for V1 diagnostics and V2 extension."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import inf, sqrt

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
    visible: bool = False
    occluded_since: float | None = None
    occlusion_duration: float = 0.0
    observation_count: int = 0
    error_sum: float = 0.0
    max_error: float = 0.0
    identity_switches: int = 0
    reappeared_at: float | None = None
    recovery_time: float | None = None

    @property
    def mean_error(self) -> float | None:
        return self.error_sum / self.observation_count if self.observation_count else None


def update_belief(
    belief: TaskBelief,
    observation: Point | None,
    *,
    timestamp: float,
    hold_last_observation: bool = True,
    ground_truth: XYZ | None = None,
    matched_point_id: str | None = None,
) -> TaskBelief:
    """Update only from current or past observations; never uses future data."""

    if observation is not None and observation.valid and observation.observed:
        error = 0.0
        count = belief.observation_count
        error_sum = belief.error_sum
        max_error = belief.max_error
        if ground_truth is not None:
            error = sqrt(sum((a - b) ** 2 for a, b in zip(observation.xyz, ground_truth)))
            count += 1
            error_sum += error
            max_error = max(max_error, error)
        reappeared_at = timestamp if belief.occluded_since is not None else belief.reappeared_at
        recovery_time = (
            max(0.0, timestamp - belief.occluded_since)
            if belief.occluded_since is not None
            else belief.recovery_time
        )
        return replace(
            belief,
            mean=observation.xyz,
            variance=(0.0, 0.0, 0.0),
            last_seen=timestamp,
            age=0.0,
            initialized=True,
            state=ObservationState.OBSERVED,
            update_source="current_observation",
            visible=True,
            occluded_since=None,
            occlusion_duration=0.0,
            observation_count=count,
            error_sum=error_sum,
            max_error=max_error,
            identity_switches=belief.identity_switches
            + int(matched_point_id is not None and matched_point_id != belief.point_id),
            reappeared_at=reappeared_at,
            recovery_time=recovery_time,
        )
    if hold_last_observation and belief.initialized:
        age = inf if belief.last_seen is None else max(0.0, timestamp - belief.last_seen)
        occluded_since = belief.occluded_since if belief.occluded_since is not None else timestamp
        return replace(
            belief,
            age=age,
            state=ObservationState.PREDICTED,
            update_source="causal_hold",
            visible=False,
            occluded_since=occluded_since,
            occlusion_duration=max(0.0, timestamp - occluded_since),
        )
    occluded_since = belief.occluded_since if belief.occluded_since is not None else timestamp
    return replace(
        belief,
        age=inf,
        state=ObservationState.UNKNOWN,
        update_source="missing",
        visible=False,
        occluded_since=occluded_since,
        occlusion_duration=max(0.0, timestamp - occluded_since),
    )
