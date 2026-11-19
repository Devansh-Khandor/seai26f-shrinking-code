"""Behavioral tests for the NASA Remember adapter."""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "t7_validation/adapters/nasa/remember.py"
spec = importlib.util.spec_from_file_location("nasa_remember_adapter", SRC)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class TestNASAAdapterRemember(unittest.TestCase):
    def setUp(self):
        self.dataset = ROOT / "data/raw/nasa93dem.csv"

    def test_contract_and_goal_directions(self):
        summary, top = adapter.answers(self.dataset)
        self.assertEqual(set(summary), {"Kloc+", "Effort-", "Defects-"})
        self.assertEqual(len(top["top"]), 10)
        self.assertEqual(len(set(top["top"])), 10)
        self.assertEqual(summary["Kloc+"]["best"], summary["Kloc+"]["max"])
        self.assertEqual(summary["Kloc+"]["worst"], summary["Kloc+"]["min"])
        for goal in ("Effort-", "Defects-"):
            self.assertEqual(summary[goal]["best"], summary[goal]["min"])
            self.assertEqual(summary[goal]["worst"], summary[goal]["max"])

    def test_row_reordering_preserves_statistics_not_row_identifiers(self):
        summary, top = adapter.answers(self.dataset)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reordered.csv"
            with self.dataset.open(newline="", encoding="utf-8") as stream:
                reader = csv.reader(stream)
                header, *rows = list(reader)
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerows(list(reversed(rows)))
            new_summary, new_top = adapter.answers(path)
        self.assertEqual(summary, new_summary)
        self.assertNotEqual(top["top"], new_top["top"])

    def test_input_data_changes_change_summary(self):
        orig_summary, _ = adapter.answers(self.dataset)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "changed.csv"
            with self.dataset.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
            j = rows[0].index("Effort-")
            rows[1][j] = str(float(rows[1][j]) + 100000)
            with path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerows(rows)
            new_summary, _ = adapter.answers(path)
        self.assertNotEqual(orig_summary["Effort-"]["mean"], new_summary["Effort-"]["mean"])


if __name__ == "__main__":
    unittest.main()
