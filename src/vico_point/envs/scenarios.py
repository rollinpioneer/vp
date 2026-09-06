"""V1 four-condition scenario registry and deterministic split generation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from random import Random


CONDITIONS = ("E00", "E10", "E01", "E11")
TASKS = ("bowl_on_plate", "mug_on_plate")


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    source_episode_group_id: str
    task: str
    layout_family: str
    condition: str
    occlusion: bool
    constraint_context: bool
    occlusion_duration_s: float
    seed: int
    split: str


def generate_scenarios(
    *,
    scenarios_per_condition: int = 50,
    tasks: tuple[str, ...] = TASKS,
    seed: int = 0,
) -> list[Scenario]:
    if scenarios_per_condition <= 0:
        raise ValueError("scenarios_per_condition must be positive")
    rng = Random(seed)
    scenarios: list[Scenario] = []
    for task_index, task in enumerate(tasks):
        for condition in CONDITIONS:
            for index in range(scenarios_per_condition):
                scenario_seed = rng.randrange(2**31)
                # Split by source group before any observation variants are created.
                split = "test" if index >= scenarios_per_condition * 0.8 else "validation"
                if index < scenarios_per_condition * 0.6:
                    split = "train"
                layout_family = f"{task}_layout_{index % 5:02d}"
                occlusion = condition in {"E10", "E11"}
                constraint = condition in {"E01", "E11"}
                duration = (0.25, 0.5, 1.0)[index % 3] if occlusion else 0.0
                scenarios.append(
                    Scenario(
                        scenario_id=f"{task_index:02d}_{condition}_{index:03d}",
                        source_episode_group_id=f"{task}_episode_{index:03d}",
                        task=task,
                        layout_family=layout_family,
                        condition=condition,
                        occlusion=occlusion,
                        constraint_context=constraint,
                        occlusion_duration_s=duration,
                        seed=scenario_seed,
                        split=split,
                    )
                )
    return scenarios


def write_registry(scenarios: list[Scenario], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(Scenario.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: getattr(item, field) for field in fields} for item in scenarios)
