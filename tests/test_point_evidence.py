import unittest

import numpy as np

from vico_point.perception.evidence import PointEvidence, PointSource
from vico_point.policy.pointbridge_adapter import (
    fill_unknown_from_visible_group_centroids_with_sources,
)


class PointEvidenceTests(unittest.TestCase):
    def test_negative_age_is_rejected(self):
        evidence = PointEvidence(
            xyz=np.zeros((1, 3)),
            visible=np.array([True]),
            confidence=np.array([1.0]),
            age_seconds=np.array([-0.1]),
            source=(PointSource.DEPTH_CURRENT,),
            timestamp_seconds=0.0,
            camera_mask=np.ones((1, 1), dtype=bool),
            point_ids=("p0",),
        )
        with self.assertRaises(ValueError):
            evidence.validate()

    def test_source_count_must_match_points(self):
        evidence = PointEvidence(
            xyz=np.zeros((2, 3)),
            visible=np.array([True, False]),
            confidence=np.array([1.0, np.nan]),
            age_seconds=np.array([0.0, 1.0]),
            source=(PointSource.DEPTH_CURRENT,),
            timestamp_seconds=0.0,
            camera_mask=np.ones((2, 1), dtype=bool),
            point_ids=("p0", "p1"),
        )
        with self.assertRaises(ValueError):
            evidence.validate()

    def test_centroid_fill_has_distinct_source(self):
        points = np.array([[1.0, 0.0, 0.0], [np.nan, np.nan, np.nan]])
        filled, count, sources = fill_unknown_from_visible_group_centroids_with_sources(
            points, np.array([True, False]), group_size=2
        )
        self.assertEqual(count, 1)
        self.assertEqual(sources[1], "group_centroid_fill")
        np.testing.assert_allclose(filled[1], filled[0])


if __name__ == "__main__":
    unittest.main()
