import inspect
import unittest

from experiments.v1r.scripts.run_delta_pose_path_diagnostic import (
    ACTION_SOURCE_BY_CONDITION,
    replay_condition,
)


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


if __name__ == "__main__":
    unittest.main()
