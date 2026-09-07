import unittest

import numpy as np

from vico_point.envs.visibility import OcclusionWindow
from vico_point.policy.pointbridge_adapter import CausalPointBridgeAdapter


class NoOracleLeakageTests(unittest.TestCase):
    def test_non_oracle_hidden_truth_raises_if_forced(self):
        adapter = CausalPointBridgeAdapter(("p0",))
        rgb = np.full((10, 10, 3), 255, dtype=np.uint8)
        depth = np.ones((10, 10), dtype=np.float32)
        intrinsic = np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]])
        window = OcclusionWindow(0, 1, (0.0, 0.0, 1.0, 1.0), depth_m=0.25)
        observation = adapter.adapt(
            np.array([[0.0, 0.0, 1.0]]),
            rgb,
            depth,
            intrinsic,
            np.eye(4),
            step=0,
            condition="E10",
            window=window,
            oracle_hidden_truth=True,
        )
        with self.assertRaises(RuntimeError):
            adapter.audit_no_hidden_truth(observation)


if __name__ == "__main__":
    unittest.main()
