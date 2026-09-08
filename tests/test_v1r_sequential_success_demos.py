import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.v1r.scripts.capture_sequential_success_demos import (
    capture_candidate,
    candidate_indices,
    make_env,
)
from experiments.v1r.scripts.verify_sequential_success_demos import (
    build_s0_labels,
    build_s1_labels,
    classify_verification,
)


def rotation_6d(matrix):
    value = np.asarray(matrix, dtype=np.float64)
    return value[..., :2, :].reshape(value.shape[:-2] + (6,))


class V1RSequentialSuccessDemoTests(unittest.TestCase):
    def test_capture_has_one_reset_and_no_midtrajectory_state_injection(self):
        source = inspect.getsource(capture_candidate)
        self.assertEqual(source.count("env.reset_to("), 1)
        self.assertNotIn("set_state_from_flattened", source)

    def test_environment_factory_keeps_explicit_delta_mode(self):
        source = inspect.getsource(make_env)
        self.assertIn("control_delta", source)
        self.assertIn("if control_delta is not None", source)

    def test_candidate_manifests_are_unionable(self):
        class Handle:
            data = {"demo_0": {}, "demo_1": {}, "demo_2": {}}

            def __getitem__(self, key):
                if key != "data":
                    raise KeyError(key)
                return self.data

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text(
                json.dumps(
                    {
                        "records": [
                            {"shard": 1, "demo_key": "demo_0", "saved_final_success": True},
                            {"shard": 2, "demo_key": "demo_2", "saved_final_success": True},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(
                    {
                        "records": [
                            {"shard": 1, "demo_key": "demo_1", "saved_final_action_success": True}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                candidate_indices(Handle(), 0, None, [first, second], layout=1),
                [0, 1],
            )

    def test_s0_matches_next_pose_terminal_repeat_contract(self):
        eef = np.zeros((3, 7), dtype=np.float64)
        eef[:, 3] = 1.0
        eef[:, 0] = [0.0, 1.0, 2.0]
        gripper = np.array([-1.0, 0.5, 1.0])
        labels = build_s0_labels(eef, gripper, rotation_6d)
        self.assertEqual(labels.shape, (3, 10))
        np.testing.assert_array_equal(labels[:, 0], [1.0, 2.0, 2.0])
        np.testing.assert_array_equal(labels[:, -1], [0.5, 1.0, 1.0])

    def test_s1_keeps_same_step_recorded_gripper_command(self):
        targets = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
        targets[:, 0, 3] = [3.0, 4.0]
        issued = np.zeros((2, 7), dtype=np.float64)
        issued[:, -1] = [-1.0, 1.0]
        labels = build_s1_labels(targets, issued)
        self.assertEqual(labels.shape, (2, 10))
        np.testing.assert_array_equal(labels[:, 0], [3.0, 4.0])
        np.testing.assert_array_equal(labels[:, -1], [-1.0, 1.0])

    def test_verification_keeps_three_gates_separate(self):
        result = classify_verification(20, 20, 0, 20)
        self.assertEqual(result["actual_command_replay_gate"], "passed")
        self.assertEqual(result["s0_label_replay_gate"], "passed")
        self.assertEqual(result["s1_label_replay_gate"], "failed")
        self.assertEqual(result["decision"], "s0_original_pointbridge_label_contract_passed")
        self.assertFalse(result["training_authorized"])

    def test_no_absolute_label_contract_does_not_authorize_training(self):
        result = classify_verification(20, 3, 4, 20)
        self.assertEqual(
            result["decision"], "actual_commands_pass_but_no_absolute_label_contract_passes"
        )
        self.assertFalse(result["training_authorized"])


if __name__ == "__main__":
    unittest.main()
