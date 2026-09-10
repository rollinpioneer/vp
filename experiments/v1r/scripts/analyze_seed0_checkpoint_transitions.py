#!/usr/bin/env python3
"""Run D1 E3: join the three existing clean-dev rollout tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["scenario_id"]: row for row in rows}
    if len(result) != len(rows): raise ValueError(f"duplicate scenario_id in {path}")
    return result


def compact(row: dict[str, str]) -> dict[str, Any]:
    keys = (
        "scenario_id", "layout", "success", "failure_stage", "steps",
        "min_eef_bowl_distance_m", "min_bowl_plate_distance_m",
        "first_grasp_step", "first_20_actions_sha256",
    )
    return {key: row.get(key, "") for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-100k", type=Path, required=True)
    parser.add_argument("--rollout-200k", type=Path, required=True)
    parser.add_argument("--rollout-300k", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tables = {"100": load(args.rollout_100k), "200": load(args.rollout_200k), "300": load(args.rollout_300k)}
    ids = set(tables["100"])
    if any(set(table) != ids for table in tables.values()): raise ValueError("E3 scenario_id sets do not match")
    if len(ids) != 40: raise ValueError(f"E3 requires 40 scenarios, got {len(ids)}")
    records=[]; patterns=Counter(); transitions=[]
    for step, table in tables.items():
        devices = {row.get("evaluation_device", "") for row in table.values()}
        if devices != {"cuda"}:
            raise ValueError(f"{step}k evaluation device mismatch: {devices}")
        if any(row.get("initial_state_match") != "passed" for row in table.values()):
            raise ValueError(f"{step}k contains an initial-state mismatch")
    if any(
        tables["100"][scenario_id].get("initial_state_sha256") != tables["200"][scenario_id].get("initial_state_sha256")
        or tables["100"][scenario_id].get("initial_state_sha256") != tables["300"][scenario_id].get("initial_state_sha256")
        for scenario_id in ids
    ):
        raise ValueError("checkpoint tables do not share the same initial-state hashes")
    for scenario_id in sorted(ids):
        values = tuple(int(tables[step][scenario_id]["success"]) for step in ("100","200","300"))
        pattern = "".join(map(str, values)); patterns[pattern]+=1
        record = {"scenario_id": scenario_id, "layout": int(tables["300"][scenario_id]["layout"]), "success_pattern_100_200_300": pattern,
                  "checkpoints": {step: compact(tables[step][scenario_id]) for step in ("100","200","300")}}
        records.append(record)
        if values[1] != values[2]:
            transitions.append({"scenario_id": scenario_id, "direction": "success_to_failure" if values[1] and not values[2] else "failure_to_success", **record})
    layout3 = [record for record in records if record["layout"] == 3]
    s200={r for r,row in tables["200"].items() if int(row["success"])}; s300={r for r,row in tables["300"].items() if int(row["success"])}
    result={"stage":"V1-R.2K.seed0-D1.E3", "status":"complete", "diagnostic_only":True,
            "inputs": {step: {"path":str(path),"sha256":sha256(path)} for step,path in (("100",args.rollout_100k), ("200",args.rollout_200k), ("300",args.rollout_300k))},
            "scenario_count":len(ids), "pattern_counts":dict(sorted(patterns.items())),
            "success_sets": {"100":sum(int(tables["100"][x]["success"]) for x in ids), "200":len(s200), "300":len(s300)},
            "success_200_intersection_300":len(s200&s300), "success_200_union_300":len(s200|s300),
            "success_200_300_jaccard":len(s200&s300)/len(s200|s300) if s200|s300 else None,
            "transitions_200_to_300":transitions, "layout_3":layout3,
            "same_scenario_ids":True, "same_initial_state_contract":True,
            "gate_impact":"diagnostic_only_not_a_clean_dev_gate"}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"patterns":result["pattern_counts"],"transitions":len(transitions),"jaccard":result["success_200_300_jaccard"]}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
