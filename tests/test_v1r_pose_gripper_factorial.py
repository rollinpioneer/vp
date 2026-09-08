import unittest

import numpy as np

from experiments.v1r.scripts.run_pointbridge_pose_gripper_factorial import (
    build_raw_factorial_actions,
    classify_factorial_outcome,
    extract_switch_window,
    first_closing_switch,
    first_stable_true_run,
    longest_true_run,
    pointbridge_pose_to_world_targets,
    saved_targets_to_pointbridge_pose,
)


def matrix_to_rotation_6d(matrix):
    matrix = np.asarray(matrix)
    return matrix[..., :2, :].reshape(matrix.shape[:-2] + (6,))


def rotation_6d_to_matrix(value):
    value = np.asarray(value).reshape(2, 3)
    first = value[0] / np.linalg.norm(value[0])
    second = value[1] - np.dot(first, value[1]) * first
    second = second / np.linalg.norm(second)
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=0)


class V1RPoseGripperFactorialTests(unittest.TestCase):
    def test_first_closing_switch_detects_open_to_close_transition(self):
        self.assertEqual(first_closing_switch([-1, -1, 1, 1]), 2)
        self.assertIsNone(first_closing_switch([-1, -1, -1]))

    def test_stable_grasp_requires_consecutive_steps(self):
        values = [False, True, True, False, True, True, True, True, True]
        self.assertEqual(first_stable_true_run(values, minimum_steps=5), 4)
        self.assertEqual(longest_true_run(values), 5)

    def test_saved_target_pointbridge_transform_roundtrips(self):
        robot_base = np.eye(4)
        robot_base[:3, 3] = [-0.66, 0.0, 0.912]
        gripper = np.eye(4)
        gripper[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        targets = np.repeat(np.eye(4)[None], 2, axis=0)
        targets[0, :3, 3] = [-0.2, 0.1, 1.1]
        targets[1, :3, 3] = [-0.1, -0.2, 1.0]
        poses = saved_targets_to_pointbridge_pose(
            targets, robot_base, gripper, matrix_to_rotation_6d
        )
        actual = pointbridge_pose_to_world_targets(
            poses, robot_base, gripper, rotation_6d_to_matrix
        )
        np.testing.assert_allclose(actual, targets, atol=1e-12)

    def test_factorial_changes_only_requested_pose_and_gripper_slices(self):
        baseline = np.arange(40, dtype=np.float64).reshape(4, 10)
        baseline[:, -1] = [-1, 1, 1, -1]
        raw_gripper = np.array([-1, -1, 1, 1], dtype=np.float64)
        targets = np.repeat(np.eye(4)[None], 4, axis=0)
        targets[:, 0, 3] = np.arange(4) + 100
        groups = build_raw_factorial_actions(
            baseline,
            raw_gripper,
            targets,
            np.eye(4),
            np.eye(4),
            matrix_to_rotation_6d,
        )
        np.testing.assert_array_equal(groups["A"], baseline)
        np.testing.assert_array_equal(groups["A"][:, :9], groups["B"][:, :9])
        np.testing.assert_array_equal(groups["C"][:, :9], groups["D"][:, :9])
        np.testing.assert_array_equal(groups["A"][:, -1], groups["C"][:, -1])
        np.testing.assert_array_equal(groups["B"][:, -1], groups["D"][:, -1])
        np.testing.assert_array_equal(groups["B"][:, -1], raw_gripper)

    def test_switch_window_keeps_five_steps_on_each_side(self):
        rows = [{"action_index": index} for index in range(20)]
        window = extract_switch_window(rows, 10, radius=5)
        self.assertEqual([row["action_index"] for row in window], list(range(5, 16)))
        self.assertEqual(window[0]["relative_to_applied_close"], -5)
        self.assertEqual(window[-1]["relative_to_applied_close"], 5)

    def test_formal_gate_does_not_drop_below_twenty_of_twenty(self):
        outcome = classify_factorial_outcome(
            {"A": 13, "B": 19, "C": 13, "D": 19}, True
        )
        self.assertEqual(outcome["result"], "result_4_no_group_reaches_20_of_20")
        self.assertEqual(outcome["formal_gate"], "failed")

    def test_each_predeclared_passing_outcome_routes_without_training(self):
        cases = [
            ({"A": 13, "B": 20, "C": 13, "D": 20}, "B", "p0_g1"),
            ({"A": 13, "B": 13, "C": 20, "D": 20}, "C", "p1_g0"),
            ({"A": 13, "B": 13, "C": 13, "D": 20}, "D", "p1_g1"),
        ]
        for totals, group, contract in cases:
            with self.subTest(group=group):
                outcome = classify_factorial_outcome(totals, True)
                self.assertEqual(outcome["winning_group"], group)
                self.assertEqual(outcome["formal_gate"], "passed")
                self.assertIn(contract, outcome["next_stage"])
                self.assertIn("without_training", outcome["next_stage"])

    def test_invalid_baseline_blocks_interpretation(self):
        outcome = classify_factorial_outcome(
            {"A": 12, "B": 20, "C": 20, "D": 20}, False
        )
        self.assertEqual(outcome["result"], "invalid_baseline_replication")
        self.assertEqual(outcome["formal_gate"], "failed")


if __name__ == "__main__":
    unittest.main()
