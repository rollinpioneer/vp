import csv
import unittest
from pathlib import Path

from experiments.v1r.scripts.check_manifest_balance import check


class V1RManifestTests(unittest.TestCase):
    def test_factorial_manifest_is_balanced(self):
        path = Path("experiments/v1r/manifests/visibility_factorial.csv")
        self.assertTrue(path.exists())
        summary = check(path)
        self.assertEqual(summary["base_scenarios"], 240)
        self.assertEqual(summary["cell_count"], 20)

    def test_clean_manifests_have_disjoint_splits(self):
        paths = [
            Path("experiments/v1r/manifests/clean_baseline_dev.csv"),
            Path("experiments/v1r/manifests/clean_baseline_confirm.csv"),
        ]
        rows = []
        for path in paths:
            self.assertTrue(path.exists())
            with path.open(newline="", encoding="utf-8") as handle:
                rows.extend(csv.DictReader(handle))
        dev = {row["scenario_id"] for row in rows if row["split"] == "dev"}
        confirm = {row["scenario_id"] for row in rows if row["split"] == "confirm"}
        self.assertTrue(dev.isdisjoint(confirm))
        self.assertEqual(len(dev), 40)
        self.assertEqual(len(confirm), 100)


if __name__ == "__main__":
    unittest.main()
