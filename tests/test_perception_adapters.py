import unittest

import numpy as np

from vico_point.perception.cotracker_adapter import triangulate_points
from vico_point.perception.evidence import PointSource
from vico_point.perception.segment_depth_adapter import extract_segment_depth_evidence


class PerceptionAdapterTests(unittest.TestCase):
    def test_segment_depth_marks_previous_fallback(self):
        ids = ("p0", "p1")
        intrinsic = np.eye(3)
        previous = extract_segment_depth_evidence(
            np.array([[1, 1], [0, 0]], dtype=bool),
            np.ones((2, 2)),
            intrinsic,
            np.eye(4),
            point_ids=ids,
            timestamp_seconds=0.0,
        )
        current = extract_segment_depth_evidence(
            np.zeros((2, 2), dtype=bool),
            np.ones((2, 2)),
            intrinsic,
            np.eye(4),
            point_ids=ids,
            timestamp_seconds=0.5,
            previous=previous,
        )
        self.assertTrue(all(item == PointSource.PREVIOUS_FALLBACK for item in current.source))
        self.assertTrue(np.all(current.age_seconds > 0))

    def test_cotracker_requires_two_visible_cameras_for_metric_point(self):
        ids = ("p0",)
        tracks = {"a": np.array([[0.0, 0.0]]), "b": np.array([[0.1, 0.0]])}
        visibility = {"a": np.array([1.0]), "b": np.array([1.0])}
        evidence = triangulate_points(
            tracks,
            visibility,
            {"a": np.eye(3), "b": np.eye(3)},
            {"a": np.eye(4), "b": np.array([[1, 0, 0, 0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])},
            point_ids=ids,
            timestamp_seconds=0.0,
        )
        self.assertEqual(evidence.source, (PointSource.TRACK_CURRENT,))
        self.assertTrue(evidence.visible[0])


if __name__ == "__main__":
    unittest.main()
