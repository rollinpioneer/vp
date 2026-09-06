"""Simple, deterministic context selectors frozen for the V1 baseline matrix."""

from __future__ import annotations

from math import dist
from random import Random

from vico_point.core.types import Point


def random_points(points: tuple[Point, ...], count: int, seed: int) -> tuple[Point, ...]:
    if count <= 0:
        return ()
    return tuple(Random(seed).sample(list(points), min(count, len(points))))


def farthest_points(points: tuple[Point, ...], count: int) -> tuple[Point, ...]:
    if count <= 0 or not points:
        return ()
    chosen = [points[0]]
    while len(chosen) < min(count, len(points)):
        candidate = max(
            (point for point in points if point not in chosen),
            key=lambda point: min(dist(point.xyz, selected.xyz) for selected in chosen),
        )
        chosen.append(candidate)
    return tuple(chosen)


def local_crop(
    points: tuple[Point, ...], anchors: tuple[tuple[float, float, float], ...], count: int
) -> tuple[Point, ...]:
    if count <= 0 or not points:
        return ()
    if not anchors:
        return farthest_points(points, count)
    ranked = sorted(points, key=lambda point: min(dist(point.xyz, anchor) for anchor in anchors))
    return tuple(ranked[: min(count, len(ranked))])
