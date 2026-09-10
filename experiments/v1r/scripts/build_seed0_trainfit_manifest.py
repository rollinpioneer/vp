#!/usr/bin/env python3
"""Build the frozen 20-row D1 training-initial-state manifest."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seed0_d1_common import DATASET_MANIFEST, sha256, validate_preflight  # noqa: E402
from state_utils import array_sha256, raw_array_sha256  # noqa: E402

FIELDS = [
    "scenario_id", "layout", "episode_key", "episode_index", "source_artifact",
    "source_artifact_sha256", "source_pkl", "source_pkl_sha256", "model_xml",
    "model_xml_sha256", "initial_state_sha256", "object_template_sha256",
]


def build_rows(dataset_manifest: Path) -> tuple[list[dict[str, str]], str]:
    payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    source = ROOT / payload["source_manifest"]
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    source_records = {
        (int(row["layout"]), row["demo_key"]): row
        for row in source_payload["records"] if row.get("accepted") and row.get("counts_toward_target", True)
    }
    pkl_by_layout = {int(item["path"].rsplit("_", 1)[-1].split(".")[0]): item for item in payload["pkl_files"]}
    rows: list[dict[str, str]] = []
    mapping_parts: list[str] = []
    by_layout_index = {layout: 0 for layout in range(1, 5)}
    for record in payload["records"]:
        layout = int(record["layout"])
        key = record["demo_key"]
        source_record = source_records[(layout, key)]
        artifact = ROOT / record["artifact"]
        xml = ROOT / source_record["model_xml"]
        pkl_path = ROOT / pkl_by_layout[layout]["path"]
        episode_index = by_layout_index[layout]
        by_layout_index[layout] += 1
        with np.load(artifact, allow_pickle=False) as bundle:
            initial_hash = raw_array_sha256(np.asarray(bundle["initial_state"]))
        data = pickle.load(pkl_path.open("rb"))
        template = data["object_point_templates"][episode_index]
        template_hash = array_sha256(np.concatenate([value.reshape(-1) for _, value in sorted(template.items())]))
        observed = {
            "artifact": sha256(artifact), "xml": sha256(xml), "pkl": sha256(pkl_path),
        }
        if observed["artifact"] != record["artifact_sha256"]:
            raise ValueError(f"artifact hash mismatch: {artifact}")
        if observed["xml"] != source_record["model_xml_sha256"]:
            raise ValueError(f"XML hash mismatch: {xml}")
        if observed["pkl"] != pkl_by_layout[layout]["sha256"]:
            raise ValueError(f"PKL hash mismatch: {pkl_path}")
        if initial_hash != source_record["initial_state_sha256"]:
            raise ValueError(f"initial state hash mismatch: layout {layout} {key}")
        if template_hash != record["object_point_template_sha256"]:
            raise ValueError(f"object template hash mismatch: layout {layout} {key}")
        rows.append({
            "scenario_id": f"v1r_2k_trainfit_l{layout}_{key}", "layout": str(layout),
            "episode_key": key, "episode_index": str(episode_index),
            "source_artifact": record["artifact"], "source_artifact_sha256": observed["artifact"],
            "source_pkl": str(pkl_path.relative_to(ROOT)), "source_pkl_sha256": observed["pkl"],
            "model_xml": source_record["model_xml"], "model_xml_sha256": observed["xml"],
            "initial_state_sha256": initial_hash, "object_template_sha256": template_hash,
        })
        mapping_parts.append(f"{layout}/{key}/{record['artifact_sha256']}")
    mapping_hash = __import__("hashlib").sha256("\n".join(mapping_parts).encode()).hexdigest()
    return rows, mapping_hash


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, default=DATASET_MANIFEST)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/v1r/manifests/b1_2k_20_trainfit.csv")
    args = parser.parse_args()
    frozen = validate_preflight()
    rows, mapping_hash = build_rows(args.dataset_manifest.resolve())
    if len(rows) != 20 or any(sum(int(row["layout"]) == layout for row in rows) != 5 for layout in range(1, 5)):
        raise ValueError("trainfit manifest must contain five unique episodes per layout")
    if len({(row["layout"], row["episode_key"]) for row in rows}) != 20:
        raise ValueError("duplicate layout/episode key in trainfit manifest")
    expected = frozen["manifest"]["ordered_episode_mapping_sha256"]
    if mapping_hash != expected:
        raise ValueError(f"ordered mapping mismatch: expected {expected}, got {mapping_hash}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"rows": 20, "per_layout": 5, "ordered_episode_mapping_sha256": mapping_hash, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

