import unittest

import numpy as np

from experiments.v1r.scripts.validate_pointbridge_absolute_pose import (
    classify_gate,
    coordinate_frame_contract,
    control_boundary_evidence,
    gripper_contract,
    native_label_alignment,
    normalize_and_postprocess,
    pointbridge_postprocess_actions,
)


class V1RPointBridgeAbsolutePoseTests(unittest.TestCase):
    def test_native_label_alignment_requires_next_pose_and_terminal_repeat(self):
        eef = np.arange(40, dtype=np.float64).reshape(4, 10)
        actions = np.concatenate([eef[1:], eef[-1:]], axis=0)
        result = native_label_alignment(
            {"action": actions, "observation": {"eef_states": eef}}
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["act_subsample"], 1)

    def test_pointbridge_postprocess_matches_pose_affine_inverse(self):
        stats = {
            "actions": {
                "min": np.array([-2.0, 1.0]),
                "max": np.array([2.0, 5.0]),
            }
        }
        actual = pointbridge_postprocess_actions(np.array([[0.25, 0.75]]), stats)
        np.testing.assert_allclose(actual, [[-1.0, 4.0]])

    def test_training_normalization_casts_to_float32_before_postprocess(self):
        class Dataset:
            pass

        dataset = Dataset()
        dataset.stats = {
            "actions": {
                "min": np.array([-2.0, 1.0], dtype=np.float64),
                "max": np.array([2.0, 5.0], dtype=np.float64),
            }
        }
        dataset.preprocess = {
            "actions": lambda value: (value - dataset.stats["actions"]["min"])
            / (
                dataset.stats["actions"]["max"]
                - dataset.stats["actions"]["min"]
                + 1e-5
            )
        }

        normalized, deployment = normalize_and_postprocess(
            dataset, np.array([[0.0, 3.0]], dtype=np.float64)
        )
        self.assertEqual(normalized.dtype, np.float32)
        self.assertEqual(deployment.dtype, np.float64)

    def test_coordinate_frame_contract_requires_matching_robot_base(self):
        result = coordinate_frame_contract(np.eye(4), np.eye(4))
        self.assertTrue(result["passed"])
        self.assertTrue(result["exact_match"])

    def test_gripper_contract_preserves_scalar_open_close_sign(self):
        raw = np.zeros((3, 10), dtype=np.float64)
        raw[:, -1] = [-1.0, 1.0, -1.0]
        deployment = raw.copy()
        deployment[:, -1] = [-0.99999, 0.99999, -0.99999]
        result = gripper_contract(raw, deployment)
        self.assertTrue(result["passed"])
        self.assertEqual(result["channels"], 1)

    def test_unconfigured_absolute_osc_limits_are_not_reported_as_hits(self):
        result = control_boundary_evidence([0.4, 0.1, 1.2], None)
        self.assertFalse(result["configured"])
        self.assertFalse(result["hit"])

    def test_configured_boundary_reports_axes(self):
        limits = np.array([[0.0, -1.0], [1.0, 1.0]])
        result = control_boundary_evidence([0.0, 0.25], limits)
        self.assertTrue(result["configured"])
        self.assertTrue(result["hit"])
        self.assertEqual(result["lower_axes"], [0])

    def test_case_b_blocks_training_when_layout_1_or_2_fails(self):
        per_layout = {
            str(layout): {"checked": 5, "passed": 4 if layout == 2 else 5}
            for layout in range(1, 5)
        }
        decision = classify_gate(per_layout, 0.05)
        self.assertEqual(decision["case"], "B")
        self.assertEqual(decision["gate"], "failed")
        self.assertFalse(decision["b0_b1_training_authorized"])
        self.assertIn("without_retraining", decision["next_stage"])

    def test_case_c_localizes_policy_problem_after_all_experts_pass(self):
        per_layout = {
            str(layout): {"checked": 5, "passed": 5} for layout in range(1, 5)
        }
        decision = classify_gate(per_layout, 0.05)
        self.assertEqual(decision["case"], "C")
        self.assertEqual(decision["gate"], "passed")
        self.assertTrue(decision["b0_b1_training_authorized"])
        self.assertIn("data_construction", decision["next_stage"])


if __name__ == "__main__":
    unittest.main()
