"""Policy boundary: V0 can smoke-test this without bundling Point Bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from vico_point.core.types import ActionProtocol, PointFrame


@dataclass(frozen=True)
class Action:
    values: tuple[float, ...]
    protocol: ActionProtocol


class PointPolicy(Protocol):
    def predict(self, frame: PointFrame) -> Action: ...


class PointBridgeAdapter(Protocol):
    """Implement this adapter when the pinned upstream checkout is available."""

    def train(self, *, config_path: str, seed: int) -> str: ...

    def evaluate(self, *, checkpoint: str, scenario_ids: list[str]) -> list[dict[str, object]]: ...


class InterfaceSmokePolicy:
    """A no-learning policy used only for frame/action contract checks."""

    def __init__(self, protocol: ActionProtocol | None = None) -> None:
        self.protocol = protocol or ActionProtocol()

    def predict(self, frame: PointFrame) -> Action:
        values = (0.0,) * self.protocol.action_dim
        return Action(values=values, protocol=self.protocol)
