#!/usr/bin/env python3
"""Materialize the audited successful episode index without copying PKL data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for layout in range(1, 5):
        audit_path = args.audit_dir / f"v0_success_audit_shard{layout}.json"
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        for record in payload["records"]:
            if not (record["saved_final_success"] or record["saved_final_action_success"]):
                continue
            demo_index = int(str(record["demo_key"]).rsplit("_", 1)[1])
            rows.append(
                {
                    "layout": layout,
                    "task": "bowl_on_plate",
                    "demo_index": demo_index,
                    "demo_key": record["demo_key"],
                    "steps": record["steps"],
                    "success_criterion": "saved_final_success_or_saved_final_action_success",
                    "source_pkl": f"bowl_on_plate_{layout}.pkl",
                }
            )
    rows.sort(key=lambda row: (int(row["layout"]), int(row["demo_index"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} audited successful episodes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
