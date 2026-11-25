"""Behavioral checks for the Apache T7 Flow common-harness adapter."""
import csv
import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "t7_apache_flow_adapter",
    ROOT / "t7_validation" / "adapters" / "apache" / "flow.py",
)
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


class TestApacheFlowAdapter(unittest.TestCase):
    def setUp(self):
        self.csv = ROOT / "data" / "raw" / "Apache_AllMeasurements.csv"

    def read_csv(self):
        with self.csv.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            return reader.fieldnames, list(reader)

    def temporary_csv(self, names, rows):
        folder = tempfile.TemporaryDirectory()
        path = Path(folder.name) / "modified_apache.csv"
        with path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=names)
            writer.writeheader()
            writer.writerows(rows)
        self.addCleanup(folder.cleanup)
        return path

    def test_shared_arrival_order_and_checkpoints(self):
        alerts, summary = ADAPTER.answers(self.csv)
        self.assertEqual(len(alerts["assessments"]), 38)
        self.assertEqual([x["row"] for x in alerts["assessments"]], list(range(155, 193)))
        cps = summary["checkpoints"]
        self.assertEqual([p["after"] for p in cps], [0, 19, 38])
        self.assertEqual([p["summary"]["Performance-"]["n"] for p in cps], [154, 173, 192])
        for p in cps:
            self.assertEqual(len(p["shortlist"]), 5)
            self.assertTrue(all(1 <= r <= 154 + p["after"] for r in p["shortlist"]))

    def test_future_targets_do_not_change_past_alerts(self):
        original_alerts, original_summary = ADAPTER.answers(self.csv)
        names, rows = self.read_csv()
        for row in rows[-19:]:
            row["Performance-"] = "0"
        changed_alerts, changed_summary = ADAPTER.answers(self.temporary_csv(names, rows))
        self.assertEqual(
            [a["alert"] for a in original_alerts["assessments"][:19]],
            [a["alert"] for a in changed_alerts["assessments"][:19]],
        )
        self.assertEqual(original_summary["checkpoints"][:2], changed_summary["checkpoints"][:2])
        self.assertNotEqual(original_summary["checkpoints"][-1], changed_summary["checkpoints"][-1])

    def test_shuffled_input_recomputes_from_new_file_order(self):
        names, rows = self.read_csv()
        random.Random(20261117).shuffle(rows)
        alerts, summary = ADAPTER.answers(self.temporary_csv(names, rows))
        self.assertEqual(len(alerts["assessments"]), 38)
        self.assertEqual([p["after"] for p in summary["checkpoints"]], [0, 19, 38])
        self.assertEqual(summary["checkpoints"][-1]["summary"]["Performance-"]["n"], 192)


if __name__ == "__main__":
    unittest.main()
