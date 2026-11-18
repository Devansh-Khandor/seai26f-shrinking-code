"""Unit tests for the T7 Remember adapter's actual behaviors."""
import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
module_path = ROOT / "t7_validation" / "adapters" / "apache" / "remember.py"
spec = importlib.util.spec_from_file_location("t7_remember_adapter", module_path)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
answers = adapter.answers


class TestAdapter(unittest.TestCase):
    def setUp(self):
        self.csv_path = ROOT / "data" / "raw" / "Apache_AllMeasurements.csv"

    def test_summary_and_favorable(self):
        summary, favorable = answers(self.csv_path)
        performance = summary["Performance-"]
        self.assertEqual(performance["best"], 840)
        self.assertEqual(performance["worst"], 2640)
        self.assertEqual(len(favorable["top"]), 10)
        self.assertGreaterEqual(len(favorable["settings"]), 2)

    def test_row_order_changes_reported_row_ids(self):
        original_summary, original_favorable = answers(self.csv_path)
        with tempfile.TemporaryDirectory() as folder:
            changed = Path(folder) / "Apache.csv"
            with self.csv_path.open(newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)
            with changed.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(list(reversed(rows)))
            summary, favorable = answers(changed)
        self.assertEqual(summary, original_summary)
        self.assertNotEqual(favorable["top"], original_favorable["top"])


if __name__ == "__main__":
    unittest.main()
