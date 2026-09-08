import inspect
import unittest

import numpy as np

from experiments.v1r.scripts.run_executable_absolute_label_contracts import (
    append_final_target_hold,
    build_s0_transition_labels,
    classify_contract_decision,
    first_divergence_index,
    replay_absolute_actions,
    select_tail_plan,
    world_actions_to_targets,
)
from experiments.v1r.scripts.verify_delta_pose_contract import (
    normalize_delta_actions,
    shared_action_stats,
)
from experiments.v1r.scripts.verify_sequential_success_demos import (
    world_targets_to_actions,
)


class V1RExecutableAbsoluteLabelContractTests(unittest.TestCase):
    def test_delta_pose_shared_normalization_roundtrip_preserves_shape_and_sign(self):
        first = np.array(
            [[-0.02, 0.01, 0.0, 0.03, -0.04, 0.05, -1.0]], dtype=np.float64
        )
        second = np.array(
            [[0.04, -0.02, 0.01, -0.01, 0.02, -0.03, 1.0]], dtype=np.float64
        )
        stats = shared_action_stats([first, second])
        normalized, deployment, error = normalize_delta_actions(
            np.concatenate([first, second]), stats
        )
        self.assertEqual(normalized.dtype, np.float32)
        self.assertEqual(deployment.shape, (2, 7))
        np.testing.assert_array_equal(
            np.sign(deployment[:, -1]), np.sign(np.concatenate([first, second])[:, -1])
        )
        self.assertLessEqual(error, 1e-6)

    def test_absolute_replay_does_not_force_controller_updates_between_steps(self):
        source = inspect.getsource(replay_absolute_actions)
        self.assertNotIn("controller.update", source)

    def test_s0_transition_uses_every_post_step_pose_and_same_step_gripper(self):
        post_step = np.zeros((3, 7), dtype=np.float64)
        post_step[:, :3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        post_step[:, 3] = 1.0
        issued = np.zeros((3, 7), dtype=np.float64)
        issued[:, -1] = [-1.0, 0.25, 1.0]

        labels = build_s0_transition_labels(post_step, issued)

        self.assertEqual(labels.shape, (3, 10))
        np.testing.assert_array_equal(labels[:, :3], post_step[:, :3])
        np.testing.assert_array_equal(labels[:, -1], issued[:, -1])

    def test_s1_world_bypass_is_a_direct_world_target_conversion(self):
        from scipy.spatial.transform import Rotation as rotation

        targets = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
        targets[:, :3, 3] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        targets[:, :3, :3] = rotation.from_rotvec(
            [[0.01, -0.02, 0.03], [-0.04, 0.05, -0.06]]
        ).as_matrix()
        gripper = np.array([-1.0, 1.0])

        actions = world_targets_to_actions(targets, gripper)
        reconstructed = world_actions_to_targets(actions)

        np.testing.assert_allclose(reconstructed, targets, atol=1e-12)
        np.testing.assert_array_equal(actions[:, -1], gripper)

    def test_tail_plan_prefers_ten_real_actions_when_available(self):
        plan = select_tail_plan(source_action_count=25, captured_action_count=15)
        self.assertEqual(plan["strategy"], "remaining_hdf5_delta_actions")
        self.assertEqual(plan["steps"], 10)
        self.assertEqual(plan["source_start_action_index"], 15)
        self.assertEqual(plan["source_stop_action_index_exclusive"], 25)

    def test_tail_plan_uses_five_holds_only_when_real_tail_is_short(self):
        plan = select_tail_plan(source_action_count=19, captured_action_count=15)
        self.assertEqual(plan["strategy"], "final_absolute_target_hold")
        self.assertEqual(plan["steps"], 5)
        self.assertEqual(plan["remaining_source_actions"], 4)

    def test_final_target_hold_preserves_prefix_and_repeats_last_action(self):
        actions = np.arange(21, dtype=np.float64).reshape(3, 7)
        extended = append_final_target_hold(actions, steps=5)
        np.testing.assert_array_equal(extended[:3], actions)
        np.testing.assert_array_equal(extended[3:], np.repeat(actions[-1:], 5, axis=0))

    def test_first_divergence_uses_predeclared_tolerance(self):
        reference = np.zeros((3, 4), dtype=np.float64)
        actual = reference.copy()
        actual[1, 2] = 2e-9
        self.assertEqual(first_divergence_index(actual, reference), 1)
        actual[1, 2] = 0.5e-9
        self.assertIsNone(first_divergence_index(actual, reference))

    def test_case_b_freezes_transition_contract_before_other_candidates(self):
        decision = classify_contract_decision(
            {"s0_old": 3, "s0_transition": 20, "s1_pb": 16, "s1_world": 20},
            {name: 20 for name in ("s0_old", "s0_transition", "s1_pb", "s1_world")},
            checked=20,
            actual_delta_passed=20,
        )
        self.assertEqual(decision["case"], "B")
        self.assertEqual(decision["selected_label_contract"], "s0_transition")
        self.assertTrue(decision["seed0_training_authorized"])
        self.assertFalse(decision["b0_b1_training_authorized"])

    def test_case_a_does_not_freeze_world_only_contract(self):
        decision = classify_contract_decision(
            {"s0_old": 3, "s0_transition": 18, "s1_pb": 16, "s1_world": 20},
            {name: 20 for name in ("s0_old", "s0_transition", "s1_pb", "s1_world")},
            checked=20,
            actual_delta_passed=20,
        )
        self.assertEqual(decision["case"], "A")
        self.assertIsNone(decision["selected_label_contract"])
        self.assertFalse(decision["seed0_training_authorized"])

    def test_case_c_freezes_uniform_tail_only_after_strict_tail_gate(self):
        decision = classify_contract_decision(
            {name: 19 for name in ("s0_old", "s0_transition", "s1_pb", "s1_world")},
            {"s0_old": 19, "s0_transition": 20, "s1_pb": 18, "s1_world": 18},
            checked=20,
            actual_delta_passed=20,
        )
        self.assertEqual(decision["case"], "C")
        self.assertEqual(
            decision["selected_label_contract"], "s0_transition_with_uniform_tail"
        )
        self.assertTrue(decision["uniform_tail_required"])

    def test_case_d_keeps_training_blocked_and_selects_delta_followup(self):
        decision = classify_contract_decision(
            {name: 19 for name in ("s0_old", "s0_transition", "s1_pb", "s1_world")},
            {name: 19 for name in ("s0_old", "s0_transition", "s1_pb", "s1_world")},
            checked=20,
            actual_delta_passed=20,
        )
        self.assertEqual(decision["case"], "D")
        self.assertIsNone(decision["selected_label_contract"])
        self.assertFalse(decision["seed0_training_authorized"])
        self.assertIn("delta_pose", decision["next_action"])


if __name__ == "__main__":
    unittest.main()
