"""Paired scenario summaries for V1; no unpaired significance claims."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


def paired_differences(
    rows: Iterable[dict[str, object]], method_a: str, method_b: str
) -> dict[str, list[float]]:
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (str(row["condition"]), str(row["scenario_id"]))
        grouped[key][str(row["method"])] = float(row["success"])
    result: dict[str, list[float]] = defaultdict(list)
    for (condition, _), values in grouped.items():
        if method_a in values and method_b in values:
            result[condition].append(values[method_a] - values[method_b])
    return dict(result)


def summarize(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    by_method_condition: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = f"{row['method']}::{row['condition']}"
        by_method_condition[key].append(float(row["success"]))
    table = {
        key: {"n": len(values), "success_rate": mean(values) if values else 0.0}
        for key, values in sorted(by_method_condition.items())
    }
    return {"metric": "task_success_rate", "table": table}


def write_results(rows: list[dict[str, object]], csv_path: str | Path, json_path: str | Path) -> None:
    csv_path, json_path = Path(csv_path), Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method", "task", "condition", "scenario_id", "training_seed",
        "success", "collision", "timeout", "input_level", "point_budget",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    json_path.write_text(json.dumps(summarize(rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
