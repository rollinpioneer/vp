import unittest

from vico_point.belief.state import TaskBelief, update_belief
from vico_point.context.selection import farthest_points, local_crop
from vico_point.core.types import InputLevel, PointBudget, ObservationState
from vico_point.core.validation import validate_frame
from vico_point.envs.scenarios import CONDITIONS, generate_scenarios
from vico_point.perception.observation import build_frame, make_point


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

    def test_context_selectors_are_bounded(self):
        points = tuple(make_point(f"p{i}", (float(i), 0.0, 0.0), "context", timestamp=0.0) for i in range(5))
        self.assertEqual(len(farthest_points(points, 3)), 3)
        self.assertEqual(len(local_crop(points, ((0.0, 0.0, 0.0),), 2)), 2)


if __name__ == "__main__":
    unittest.main()
