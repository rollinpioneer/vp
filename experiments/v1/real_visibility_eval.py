#!/usr/bin/env python3
"""Evaluate paired E00/E10/oracle observations from a real simulator capture."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vico_point.envs.visibility import OcclusionWindow  # noqa: E402
from vico_point.policy.pointbridge_adapter import CausalPointBridgeAdapter  # noqa: E402


def load_capture(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"rgb", "depth_m", "gt_points", "intrinsic", "target_to_camera"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"capture missing arrays: {sorted(missing)}")
        capture = {key: np.asarray(data[key]) for key in required}
    steps = len(capture["rgb"])
    if capture["depth_m"].shape[0] != steps or capture["gt_points"].shape[0] != steps:
        raise ValueError("rgb, depth_m and gt_points must have the same step count")
    return capture


def matrix_at(array: np.ndarray, step: int) -> np.ndarray:
    return array if array.ndim == 2 else array[step]


def evaluate(capture: dict[str, np.ndarray], start: int, duration: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    point_count = capture["gt_points"].shape[1]
    point_ids = tuple(f"task_{index:04d}" for index in range(point_count))
    branches = (
        ("E00", False),
        ("E10", False),
        ("E10_ORACLE_HIDDEN_GT", True),
    )
    window = OcclusionWindow(start, duration)
    rows: list[dict[str, object]] = []
    branch_summary: dict[str, object] = {}
    for branch, oracle in branches:
        adapter = CausalPointBridgeAdapter(point_ids)
        visible_fractions: list[float] = []
        held_fractions: list[float] = []
        unknown_fractions: list[float] = []
        for step in range(len(capture["rgb"])):
            condition = "E00" if branch == "E00" else "E10"
            adapted = adapter.adapt(
                capture["gt_points"][step],
                capture["rgb"][step],
                capture["depth_m"][step],
                matrix_at(capture["intrinsic"], step),
                matrix_at(capture["target_to_camera"], step),
                step=step,
                condition=condition,
                window=window,
                oracle_hidden_truth=oracle,
            )
            if not oracle:
                adapter.audit_no_hidden_truth(adapted)
            visible = float(np.mean(adapted.visible_mask))
            held = adapted.source.count("last_reliable_hold") / point_count
            unknown = adapted.source.count("unknown") / point_count
            visible_fractions.append(visible)
            held_fractions.append(held)
            unknown_fractions.append(unknown)
            rows.append(
                {
                    "branch": branch,
                    "step": step,
                    "occlusion_active": int(adapted.visibility.occlusion_active),
                    "visible_fraction": visible,
                    "last_reliable_hold_fraction": held,
                    "unknown_fraction": unknown,
                    "max_age_steps": float(np.max(adapted.age_steps[np.isfinite(adapted.age_steps)]))
                    if np.isfinite(adapted.age_steps).any()
                    else "",
                    "hidden_truth_reads": sum(source == "oracle_hidden_gt" for source in adapted.source),
                }
            )
        branch_summary[branch] = {
            "steps": len(visible_fractions),
            "mean_visible_fraction": float(np.mean(visible_fractions)),
            "mean_last_reliable_hold_fraction": float(np.mean(held_fractions)),
            "mean_unknown_fraction": float(np.mean(unknown_fractions)),
            "policy_hidden_truth_reads": 0 if not oracle else None,
            "diagnostic_oracle": oracle,
        }
    summary = {
        "stage": "V1",
        "status": "real_capture_visibility_gate_complete",
        "scientific_scope": "observation_gate_only_not_policy_success",
        "paired_branches": [branch for branch, _ in branches],
        "point_count": point_count,
        "occlusion": {
            "start_step": start,
            "duration_steps": duration,
            "control_hz": 20,
            "duration_seconds": duration / 20.0,
        },
        "branches": branch_summary,
        "invariants": {
            "same_capture_and_gt_point_identity": True,
            "rgb_depth_visible_points_synchronized": True,
            "non_oracle_hidden_truth_reads": 0,
            "oracle_is_diagnostic_upper_bound_only": True,
        },
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path, help="NPZ produced from the pinned Point Bridge simulator")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "v1" / "real_visibility")
    parser.add_argument("--occlusion-start-step", type=int, default=20)
    parser.add_argument("--occlusion-duration-steps", type=int, choices=(5, 10), default=10)
    args = parser.parse_args()
    capture = load_capture(args.capture)
    rows, summary = evaluate(capture, args.occlusion_start_step, args.occlusion_duration_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
