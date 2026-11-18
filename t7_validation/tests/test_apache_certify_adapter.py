"""Independent behavioral tests of the Apache Certify adapter."""
import importlib.util
import math
import tempfile
import unittest
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "t7_apache_certify_adapter",
    ROOT / "t7_validation" / "adapters" / "apache" / "certify.py",
)
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


class TestApacheCertifyAdapter(unittest.TestCase):
    def setUp(self):
        self.csv = ROOT / "data" / "raw" / "Apache_AllMeasurements.csv"

    def test_original_groups_are_complete_and_numerically_correct(self):
        from remember import load_configurations
        rows = load_configurations(self.csv)
        import statistics
        for setting, answer in zip(("keepAlive", "accessLog"), ADAPTER.answers(self.csv)):
            groups = answer["groups"]
            self.assertEqual(set(groups), {"0", "1"})
            self.assertEqual(sum(g["n"] for g in groups.values()), len(rows))
            for value in (0, 1):
                expected = [r["Performance-"] for r in rows if r[setting] == value]
                reported = groups[str(value)]
                self.assertEqual(reported["n"], len(expected))
                self.assertAlmostEqual(reported["mean"], statistics.mean(expected))
                self.assertAlmostEqual(reported["median"], statistics.median(expected))
                self.assertTrue(math.isfinite(reported["mean"]))
            self.assertEqual(len(answer["comparisons"]), 1)
            self.assertIn(answer["comparisons"][0]["verdict"], ("same", "different"))

    def test_expected_observational_pattern(self):
        keepalive, accesslog = ADAPTER.answers(self.csv)
        self.assertEqual(keepalive["comparisons"][0]["verdict"], "different")
        self.assertEqual(accesslog["comparisons"][0]["verdict"], "same")
        self.assertGreater(keepalive["groups"]["1"]["mean"], keepalive["groups"]["0"]["mean"])
        # 'same' here means inconclusive under our conservative rule.

    def test_row_shuffle_preserves_group_statistics(self):
        import random
        original = ADAPTER.answers(self.csv)
        with self.csv.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            names = reader.fieldnames
            rows = list(reader)
        random.Random(43).shuffle(rows)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reordered.csv"
            with path.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=names)
                writer.writeheader()
                writer.writerows(rows)
            altered = ADAPTER.answers(path)
        for before, after in zip(original, altered):
            self.assertEqual(before["groups"], after["groups"])
            self.assertEqual(before["comparisons"], after["comparisons"])


if __name__ == "__main__":
    unittest.main()
