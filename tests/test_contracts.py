import unittest

import numpy as np

from vico_point.belief.state import TaskBelief, update_belief
from vico_point.context.selection import farthest_points, local_crop
from vico_point.core.types import InputLevel, PointBudget, ObservationState
from vico_point.core.validation import validate_frame
from vico_point.envs.scenarios import CONDITIONS, generate_scenarios
from vico_point.envs.visibility import OcclusionWindow, visible_points_from_depth
from vico_point.envs.mimiclabs_compat import migrate_saved_model_xml
from vico_point.perception.observation import build_frame, make_point
from vico_point.policy.pointbridge_adapter import (
    CausalPointBridgeAdapter,
    fill_unknown_from_visible_group_centroids,
)


class ContractTests(unittest.TestCase):
    def test_frame_budget_and_masks(self):
        budget = PointBudget()
        frame = build_frame(
            "f0",
            0.0,
            InputLevel.P,
            task_points=(make_point("task-0", (1.0, 0.0, 0.0), "task", timestamp=0.0),),
        )
        validate_frame(frame, budget)
        self.assertEqual(frame.count_by_group()["total"], 1)

    def test_occluded_belief_is_causal_and_unknown_before_initialization(self):
        belief = TaskBelief("task-0", "task")
        missing = update_belief(belief, None, timestamp=0.0)
        self.assertEqual(missing.state, ObservationState.UNKNOWN)
        observed = update_belief(
            missing,
            make_point("task-0", (1.0, 2.0, 3.0), "task", timestamp=1.0),
            timestamp=1.0,
        )
        predicted = update_belief(observed, None, timestamp=1.5)
        self.assertEqual(predicted.state, ObservationState.PREDICTED)
        self.assertEqual(predicted.mean, (1.0, 2.0, 3.0))
        self.assertAlmostEqual(predicted.age, 0.5)

    def test_scenario_split_keeps_source_group_together(self):
        scenarios = generate_scenarios(scenarios_per_condition=10)
        self.assertEqual({item.condition for item in scenarios}, set(CONDITIONS))
        groups = {}
        for item in scenarios:
            groups.setdefault(item.source_episode_group_id, set()).add(item.split)
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        paired = {}
        for item in scenarios:
            paired.setdefault(item.source_episode_group_id, set()).add(item.seed)
        self.assertTrue(all(len(seeds) == 1 for seeds in paired.values()))

    def test_context_selectors_are_bounded(self):
        points = tuple(make_point(f"p{i}", (float(i), 0.0, 0.0), "context", timestamp=0.0) for i in range(5))
        self.assertEqual(len(farthest_points(points, 3)), 3)
        self.assertEqual(len(local_crop(points, ((0.0, 0.0, 0.0),), 2)), 2)

    def test_rgb_depth_and_visibility_are_synchronously_occluded(self):
        rgb = np.full((10, 10, 3), 255, dtype=np.uint8)
        depth = np.ones((10, 10), dtype=np.float32)
        point = np.array([[0.0, 0.0, 1.0]])
        intrinsic = np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]])
        window = OcclusionWindow(2, 3, (0.4, 0.4, 0.7, 0.7), depth_m=0.25)
        result = visible_points_from_depth(point, rgb, depth, intrinsic, np.eye(4), window=window, step=2)
        self.assertTrue(result.occlusion_active)
        self.assertFalse(result.visible_mask[0])
        self.assertEqual(tuple(result.rgb[5, 5]), window.rgb)
        self.assertAlmostEqual(float(result.depth_m[5, 5]), 0.25)

    def test_non_oracle_adapter_holds_past_point_without_hidden_truth(self):
        adapter = CausalPointBridgeAdapter(("p0",))
        rgb = np.full((10, 10, 3), 255, dtype=np.uint8)
        depth = np.ones((10, 10), dtype=np.float32)
        intrinsic = np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]])
        gt = np.array([[0.0, 0.0, 1.0]])
        first = adapter.adapt(gt, rgb, depth, intrinsic, np.eye(4), step=0, condition="E00")
        adapter.audit_no_hidden_truth(first)
        moved_hidden = np.array([[0.1, 0.0, 1.0]])
        window = OcclusionWindow(1, 1, (0.0, 0.0, 1.0, 1.0), depth_m=0.25)
        hidden = adapter.adapt(
            moved_hidden, rgb, depth, intrinsic, np.eye(4), step=1, condition="E10", window=window
        )
        adapter.audit_no_hidden_truth(hidden)
        np.testing.assert_allclose(hidden.policy_points, gt)
        self.assertEqual(hidden.source, ("causal_hold",))

    def test_unknown_policy_points_use_visible_group_centroid(self):
        points = np.array(
            [
                [1.0, 2.0, 3.0],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
            ]
        )
        filled, count = fill_unknown_from_visible_group_centroids(
            points, np.array([True, False, False, False]), group_size=2
        )
        np.testing.assert_allclose(filled[1], points[0])
        np.testing.assert_allclose(filled[2:], 0.0)
        self.assertEqual(count, 3)

    def test_oracle_only_reveals_synthetic_occluder_hidden_points(self):
        adapter = CausalPointBridgeAdapter(("p0",))
        rgb = np.full((10, 10, 3), 255, dtype=np.uint8)
        depth = np.ones((10, 10), dtype=np.float32)
        intrinsic = np.array(
            [[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]]
        )
        gt = np.array([[0.0, 0.0, 1.0]])
        window = OcclusionWindow(1, 1, (0.0, 0.0, 1.0, 1.0), depth_m=0.25)
        before = adapter.adapt(
            gt,
            rgb,
            depth,
            intrinsic,
            np.eye(4),
            step=0,
            condition="E10",
            window=window,
            oracle_hidden_truth=True,
        )
        self.assertEqual(before.source, ("current_visible_depth",))
        hidden = adapter.adapt(
            gt,
            rgb,
            depth,
            intrinsic,
            np.eye(4),
            step=1,
            condition="E10",
            window=window,
            oracle_hidden_truth=True,
        )
        self.assertEqual(hidden.source, ("oracle_gt_hidden",))

    def test_belief_diagnostics_track_error_switch_and_recovery(self):
        belief = TaskBelief("task-0", "task")
        belief = update_belief(
            belief,
            make_point("detector-7", (1.1, 2.0, 3.0), "task", timestamp=0.0),
            timestamp=0.0,
            ground_truth=(1.0, 2.0, 3.0),
            matched_point_id="detector-7",
        )
        belief = update_belief(belief, None, timestamp=0.5)
        belief = update_belief(
            belief,
            make_point("task-0", (1.0, 2.0, 3.0), "task", timestamp=1.0),
            timestamp=1.0,
            ground_truth=(1.0, 2.0, 3.0),
            matched_point_id="task-0",
        )
        self.assertEqual(belief.identity_switches, 1)
        self.assertAlmostEqual(belief.mean_error or 0.0, 0.05)
        self.assertAlmostEqual(belief.recovery_time or 0.0, 0.5)

    def test_saved_mimiclabs_xml_migration_is_metadata_only_and_idempotent(self):
        expected_bbox_sizes = {
            "bowl_0": "0.0649999688 0.0649981696 0.0184044146",
            "bowl_1": "0.059999415 0.0599991702 0.026286042",
            "bowl_2": "0.057499530225 0.05749959405 0.02421663445",
            "bowl_4": "0.05 0.04999958 0.0248779995",
        }
        for variant, bbox_size in expected_bbox_sizes.items():
            with self.subTest(variant=variant):
                xml = f"""<mujoco><asset>
                <texture file='/assets/objaverse/bowl/{variant}/visual/image0.png'/>
                </asset><worldbody>
                <body name='plate_main'><geom name='plate_g0' type='box' size='1 1 1'/></body>
                <body name='bowl_main'><geom name='bowl_g0' type='box' size='1 1 1'/></body>
                </worldbody></mujoco>"""
                migrated = migrate_saved_model_xml(xml)
                migrated_twice = migrate_saved_model_xml(migrated)
                self.assertEqual(migrated, migrated_twice)
                self.assertIn("plate_reg_bbox", migrated)
                self.assertIn("bowl_reg_int", migrated)
                self.assertIn(bbox_size, migrated)
                self.assertIn("plate_horizontal_radius_site", migrated)
                self.assertEqual(migrated.count("plate_g0"), 1)


if __name__ == "__main__":
    unittest.main()
