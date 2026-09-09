import csv
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.v1r.scripts.evaluate_clean_baseline import evaluate
from experiments.v1r.scripts.audit_runner_parity import path_success_summary
from experiments.v1r.scripts.diagnose_expert_replay import (
    attach_first_explanatory_difference,
    reset_gripper_cache,
)
from experiments.v1r.scripts.state_utils import (
    load_state_bundle,
    load_state_index,
    raw_array_sha256,
)


class V1RAuditTests(unittest.TestCase):
    def test_targeted_diagnostic_localizes_first_delta_execution_difference(self):
        transitions = []
        for index in range(3):
            action = [0.1 * (index + 1), 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
            if index == 2:
                action[0] = 1.0
            transitions.append(
                {
                    "action_index": index,
                    "action": action,
                    "actual_vs_saved_controller_target_errors": {
                        "position_l2_m": 0.0 if index == 0 else 0.001 * index,
                        "orientation_angle_rad": 0.0,
                    },
                    "actual_vs_reference_errors": {
                        "eef_position_l2_m": 0.001 * (index + 1),
                        "eef_orientation_angle_rad": 0.0,
                        "eef_linear_velocity_l2_m_per_s": 0.0,
                        "eef_angular_velocity_l2_rad_per_s": 0.0,
                        "gripper_qpos_l2_m": 0.0,
                        "bowl_position_l2_m": 0.0,
                    },
                    "controller_target_tracking_errors": {
                        "position_l2_m": 0.01,
                    },
                }
            )
        replay = {"controller_use_delta": True, "transitions": transitions}
        attach_first_explanatory_difference(replay, 3)
        evidence = replay["first_explanatory_difference"]
        self.assertTrue(evidence["candidate_supported"])
        self.assertEqual(
            evidence["first_translation_saturation"]["action_index"], 2
        )
        self.assertTrue(evidence["delta_target_error_propagation"]["supported"])

    def test_diagnostic_reset_clears_stateful_gripper_cache(self):
        class Gripper:
            dof = 1

            def __init__(self):
                self.current_action = np.array([0.75])

        class Robot:
            gripper = Gripper()

        class Env:
            robots = [Robot()]

        env = Env()
        reset_gripper_cache(env)
        np.testing.assert_array_equal(env.robots[0].gripper.current_action, [0.0])
        reset_gripper_cache(env, [-0.25])
        np.testing.assert_array_equal(env.robots[0].gripper.current_action, [-0.25])

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

    def test_pointbridge_dataset_keeps_task_layouts_as_sampling_groups(self):
        upstream = Path("/home/xushijie/vico-point/third_party/pointbridge")
        if not upstream.is_dir():
            self.skipTest("Point Bridge checkout is not available")
        sys.path.insert(0, str(upstream))
        try:
            from point_bridge.read_data.mimiclabs import BCDataset
        except ImportError as exc:
            self.skipTest(f"Point Bridge runtime is not installed: {exc}")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task in ("bowl_on_plate_2", "bowl_on_plate_1"):
                observation = {
                    "eef_states": np.tile(np.array([[0, 0, 0, 1, 0, 0, 0]], dtype=np.float64), (2, 1)),
                    "gripper_states": np.array([-1.0, 1.0]),
                    "robot_tracks_3d": np.zeros((2, 1, 3), dtype=np.float32),
                    "object_tracks_1_3d": np.zeros((2, 1, 1, 3), dtype=np.float32),
                }
                payload = {
                    "observations": [observation],
                    "actions": [np.zeros((2, 7), dtype=np.float32)],
                    "task_emb": np.zeros(384, dtype=np.float32),
                }
                (root / f"{task}.pkl").write_bytes(pickle.dumps(payload))
            dataset = BCDataset(
                path=str(root), suffix=None, num_demos_per_task=1,
                history_len=1, action_chunking=False, num_queries=1,
                img_size=[8, 8], num_robot_points=1, num_points_per_obj=1,
                robot_points_key="robot_tracks", object_points_key="object_tracks",
                pixel_keys=["unused"], act_subsample=1, obs_subsample=1,
                obs_type=["points"], action_mode="delta_pose",
            )
            self.assertEqual(dataset.tasks, ["bowl_on_plate_1", "bowl_on_plate_2"])
            self.assertEqual(sorted(dataset._episodes), [0, 1])
            self.assertEqual(len(dataset._sample_episode(env_idx=0)["action"]), 2)

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
