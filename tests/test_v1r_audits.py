import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.v1r.scripts.evaluate_clean_baseline import evaluate
from experiments.v1r.scripts.audit_runner_parity import path_success_summary
from experiments.v1r.scripts.state_utils import (
    load_state_bundle,
    load_state_index,
    raw_array_sha256,
)
from vico_point.data.layout_balanced_sampler import layout_balanced_weights


class V1RAuditTests(unittest.TestCase):
    def test_runner_path_success_summary_reports_each_runner(self):
        records = [
            {
                "paths": {
                    "legacy": {"success": 1},
                    "new": {"success": 1},
                    "official": {"success": 1},
                }
            },
            {
                "paths": {
                    "legacy": {"success": 0},
                    "new": {"success": 0},
                    "official": {"success": 0},
                }
            },
        ]
        summary = path_success_summary(records)
        self.assertEqual(set(summary), {"legacy", "new", "official"})
        self.assertTrue(all(value["success_rate"] == 0.5 for value in summary.values()))

    def test_layout_balanced_weights_give_each_layout_equal_mass(self):
        layouts = [1, 1, 1, 2, 3, 3]
        weights = layout_balanced_weights(layouts)
        totals = {}
        for layout, weight in zip(layouts, weights):
            totals[layout] = totals.get(layout, 0.0) + weight
        self.assertEqual(set(totals), {1, 2, 3})
        for value in totals.values():
            self.assertAlmostEqual(value, 1 / 3)

    def test_state_bundle_rejects_index_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "state.npz"
            state = np.arange(8, dtype=np.float64)
            np.savez_compressed(
                state_path,
                sim_state=state,
                object_point_keys=np.asarray([], dtype="U"),
            )
            index_path = root / "index.csv"
            fields = [
                "scenario_id",
                "layout",
                "simulator_seed",
                "initial_state_key",
                "state_path",
                "state_sha256",
                "restored_state_sha256",
            ]
            with index_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "scenario_id": "s0",
                        "layout": 1,
                        "simulator_seed": 1,
                        "initial_state_key": "s0_state",
                        "state_path": str(state_path),
                        "state_sha256": "wrong",
                        "restored_state_sha256": raw_array_sha256(state),
                    }
                )
            row = load_state_index(index_path)["s0"]
            with self.assertRaises(ValueError):
                load_state_bundle(root, row)

    def test_clean_gate_requires_expected_actual_state_equality(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            rollout = root / "rollout.csv"
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"x")
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["training_seed", "scenario_id", "condition"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "training_seed": 0,
                        "scenario_id": "s0",
                        "condition": "E00_CLEAN",
                    }
                )
            fields = [
                "training_seed",
                "scenario_id",
                "success",
                "simulator_exception",
                "action_decode_error",
                "expected_initial_state_sha256",
                "actual_initial_state_sha256",
                "initial_state_match",
            ]
            with rollout.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "training_seed": 0,
                        "scenario_id": "s0",
                        "success": 1,
                        "simulator_exception": 0,
                        "action_decode_error": 0,
                        "expected_initial_state_sha256": "a",
                        "actual_initial_state_sha256": "b",
                        "initial_state_match": "passed",
                    }
                )
            result = evaluate(manifest, {0: checkpoint}, [rollout], [0])
            self.assertEqual(result["paired_initial_state_check"], "failed_or_unresolved")

    def test_dev_precheck_keeps_predeclared_half_success_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            rollout = root / "rollout.csv"
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"x")
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["training_seed", "scenario_id", "condition", "split"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "training_seed": 0,
                        "scenario_id": "s0",
                        "condition": "E00_CLEAN",
                        "split": "dev",
                    }
                )
            with rollout.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "training_seed",
                        "scenario_id",
                        "success",
                        "simulator_exception",
                        "action_decode_error",
                        "expected_initial_state_sha256",
                        "actual_initial_state_sha256",
                        "initial_state_match",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "training_seed": 0,
                        "scenario_id": "s0",
                        "success": 0,
                        "simulator_exception": 0,
                        "action_decode_error": 0,
                        "expected_initial_state_sha256": "a",
                        "actual_initial_state_sha256": "a",
                        "initial_state_match": "passed",
                    }
                )
            audits = {
                "runner_parity": {"status": "passed"},
                "cpu_cuda_parity": {
                    "status": "blocked_unavailable_cuda",
                    "deployment_device_protocol_frozen": True,
                },
                "expert_replay": {"status": "passed"},
            }
            result = evaluate(manifest, {0: checkpoint}, [rollout], [0], audits)
            self.assertEqual(result["evaluation_stage"], "dev_precheck")
            self.assertEqual(result["status"], "blocked_clean_baseline")
            self.assertEqual(
                result["thresholds"]["pooled_success_rate_min"], 0.50
            )
            self.assertTrue(
                any("below the predeclared 0.50" in item for item in result["blockers"])
            )

    def test_per_seed_rate_ignores_rollout_rows_outside_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            rollout = root / "rollout.csv"
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"x")
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["training_seed", "scenario_id", "condition", "split"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "training_seed": 0,
                        "scenario_id": "selected",
                        "condition": "E00_CLEAN",
                        "split": "dev",
                    }
                )
            fields = [
                "training_seed",
                "scenario_id",
                "success",
                "simulator_exception",
                "action_decode_error",
                "expected_initial_state_sha256",
                "actual_initial_state_sha256",
                "initial_state_match",
            ]
            with rollout.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for scenario_id, success in (("selected", 1), ("extra", 0)):
                    writer.writerow(
                        {
                            "training_seed": 0,
                            "scenario_id": scenario_id,
                            "success": success,
                            "simulator_exception": 0,
                            "action_decode_error": 0,
                            "expected_initial_state_sha256": "a",
                            "actual_initial_state_sha256": "a",
                            "initial_state_match": "passed",
                        }
                    )
            audits = {
                "runner_parity": {"status": "passed"},
                "cpu_cuda_parity": {
                    "status": "blocked_unavailable_cuda",
                    "deployment_device_protocol_frozen": True,
                },
                "expert_replay": {"status": "passed"},
            }
            result = evaluate(manifest, {0: checkpoint}, [rollout], [0], audits)
            self.assertEqual(result["per_seed"]["0"]["rows"], 1)
            self.assertEqual(result["per_seed"]["0"]["success_rate"], 1.0)
            self.assertEqual(result["status"], "dev_precheck_passed")


if __name__ == "__main__":
    unittest.main()
