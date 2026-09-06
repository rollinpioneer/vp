"""Small deterministic V1 diagnostic environment.

This is an interface smoke test, not a replacement for Point Bridge or a
scientific result. It makes the four factors observable before the upstream
simulator and visual front-end are connected.
"""

from __future__ import annotations

import hashlib
from random import Random

from .scenarios import Scenario


METHODS = {
    "sparse_task_points": {"E00": 0.92, "E10": 0.56, "E01": 0.70, "E11": 0.40},
    "budget_matched_full_scene": {"E00": 0.92, "E10": 0.58, "E01": 0.84, "E11": 0.53},
    "high_budget_full_scene": {"E00": 0.95, "E10": 0.63, "E01": 0.88, "E11": 0.59},
}


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def evaluate_scenario(scenario: Scenario, method: str, training_seed: int) -> dict[str, object]:
    if method not in METHODS:
        raise ValueError(f"unknown diagnostic method: {method}")
    probability = METHODS[method][scenario.condition]
    success = Random(_stable_seed(scenario.seed, method, training_seed)).random() < probability
    collision = bool(scenario.constraint_context and method == "sparse_task_points" and not success)
    timeout = bool(scenario.occlusion and not success and not collision)
    return {
        "method": method,
        "task": scenario.task,
        "condition": scenario.condition,
        "scenario_id": scenario.scenario_id,
        "training_seed": training_seed,
        "success": int(success),
        "collision": int(collision),
        "timeout": int(timeout),
        "input_level": "P",
        "point_budget": 64 if method != "high_budget_full_scene" else 128,
    }
