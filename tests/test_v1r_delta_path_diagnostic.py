import inspect
import json
import unittest
from pathlib import Path

import numpy as np

from experiments.v1r.scripts.run_delta_pose_path_diagnostic import (
    ACTION_SOURCE_BY_CONDITION,
    float32_labels_to_float64_controller,
    replay_condition,
)
from experiments.v1r.scripts.verify_delta_numeric_contract import select_records


class V1RDeltaPathDiagnosticTests(unittest.TestCase):
    def test_action_sources_cover_float64_float32_and_normalized_paths(self):
        self.assertEqual(
            set(ACTION_SOURCE_BY_CONDITION),
            {
                "A_capture_raw",
                "B_capture_normalized",
                "C_pointbridge_raw",
                "D_pointbridge_normalized",
                "E_capture_float32",
                "F_pointbridge_float32",
                "G_pointbridge_float32_label_float64_controller",
            },
        )
        self.assertEqual(
            ACTION_SOURCE_BY_CONDITION["E_capture_float32"],
            "raw_issued_actions_float32",
        )
        self.assertEqual(
            ACTION_SOURCE_BY_CONDITION["B_capture_normalized"],
            "shared_minmax_float32_inverse_float64",
        )

    def test_replay_preserves_input_action_dtype(self):
        source = inspect.getsource(replay_condition)
        self.assertIn("enumerate(np.asarray(actions))", source)
        self.assertNotIn("enumerate(np.asarray(actions, dtype=np.float64))", source)

    def test_candidate_rounds_in_float32_then_promotes_for_controller(self):
        raw = np.array([[0.123456789, 0, 0, 0, 0, 0, -1]], dtype=np.float64)
        labels, controller, error = float32_labels_to_float64_controller(raw)
        self.assertEqual(labels.dtype, np.float32)
        self.assertEqual(controller.dtype, np.float64)
        np.testing.assert_array_equal(controller, labels.astype(np.float64))
        self.assertGreater(error, 0.0)

    def test_selected_scope_does_not_expand_to_all_records(self):
        records = [
            {"accepted": True, "layout": layout, "demo_key": demo}
            for layout, demo in ((1, "demo_18"), (2, "demo_27"), (3, "demo_10"), (4, "demo_2"))
        ]
        records.append({"accepted": True, "layout": 1, "demo_key": "demo_extra"})
        selected = select_records({"records": records}, "selected")
        self.assertEqual(len(selected), 4)

    def test_candidate_report_blocks_all_twenty_expansion_after_three_of_four(self):
        report = json.loads(
            Path("experiments/v1r/reports/delta_numeric_contract_2j_n.json").read_text()
        )
        self.assertEqual(report["gate"], {"checked": 4, "passed": 3, "required_passed": 4, "status": "failed"})
        self.assertFalse(report["decision"]["all_20_expansion_authorized"])
        self.assertFalse(report["decision"]["all_20_expansion_run"])
        self.assertIsNone(report["decision"]["selected_label_contract"])

    def test_factorial_report_uses_unambiguous_necessity_observations(self):
        report = json.loads(
            Path("experiments/v1r/reports/delta_pose_path_diagnostic_2j_f.json").read_text()
        )
        conclusion = report["conclusion"]
        self.assertTrue(conclusion["observed_minmax_path_failed_all_three_selected_failure_records"])
        self.assertFalse(conclusion["minmax_is_necessary_for_demo18_failure"])
        self.assertNotIn("minmax_float32_inverse_is_sufficient_for_all_three_failures", conclusion)


if __name__ == "__main__":
    unittest.main()
