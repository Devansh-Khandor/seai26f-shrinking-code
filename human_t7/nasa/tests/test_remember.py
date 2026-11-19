"""Regression tests for the original NASA T7 Remember implementation."""
import csv
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "human_t7" / "nasa" / "remember.py"
spec = importlib.util.spec_from_file_location("nasa_original_remember", SRC)
remember = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remember)


class TestRemember(unittest.TestCase):
    @staticmethod
    def projects():
        return [
            {"row_number": 1, "Kloc+": 10.0, "Effort-": 30.0, "Defects-": 20.0},
            {"row_number": 2, "Kloc+": 30.0, "Effort-": 10.0, "Defects-": 5.0},
            {"row_number": 3, "Kloc+": 20.0, "Effort-": 20.0, "Defects-": 10.0},
        ]

    def test_summary_statistics(self):
        s = remember.summarize(self.projects())
        self.assertEqual(s["Kloc+"], {"minimum": 10, "median": 20, "mean": 20, "maximum": 30})
        self.assertEqual(s["Effort-"]["median"], 20)
        self.assertEqual(s["Defects-"]["mean"], 35 / 3)

    def test_goal_directions_and_ranking(self):
        ranked = remember.rank_projects(self.projects())
        self.assertEqual([p["row_number"] for p in ranked], [2, 3, 1])
        self.assertAlmostEqual(ranked[0]["d2h"], 0)
        self.assertAlmostEqual(ranked[-1]["d2h"], 1)

    def test_constant_goal_has_zero_distance_contribution(self):
        projects = self.projects()
        for row in projects:
            row["Defects-"] = 7.0
        bounds = remember.goal_bounds(projects)
        self.assertTrue(math.isfinite(remember.distance_to_heaven(projects[0], bounds)))
        self.assertAlmostEqual(remember.distance_to_heaven(projects[1], bounds), 0)

    def test_tied_scores_break_by_row_number(self):
        p = self.projects()
        clone = dict(p[1], row_number=4)
        ranked = remember.rank_projects(p + [clone])
        self.assertEqual([r["row_number"] for r in ranked[:2]], [2, 4])

    def test_read_projects_and_csv_row_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            csvpath = Path(tmp) / "tiny.csv"
            with csvpath.open("w", newline="", encoding="utf-8") as out:
                writer = csv.writer(out)
                writer.writerow(["Attribute", "Kloc+", "Effort-", "Defects-"])
                writer.writerow([" vh ", "3", "5", "7"])
                writer.writerow(["n", "8", "2", "1"])
            rows = remember.load_projects(csvpath)
            self.assertEqual([r["row_number"] for r in rows], [1, 2])
            self.assertEqual(rows[0]["Attribute"], "vh")
            self.assertEqual(rows[0]["Kloc+"], 3.0)

    def test_missing_goal_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            csvpath = Path(tmp) / "bad.csv"
            csvpath.write_text("Kloc+,Effort-\n1,2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing goal columns"):
                remember.load_projects(csvpath)

    def test_empty_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            csvpath = Path(tmp) / "empty.csv"
            csvpath.write_text("Kloc+,Effort-,Defects-\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no projects"):
                remember.load_projects(csvpath)

    def test_nasa_csv_has_93_rows(self):
        rows = remember.load_projects()
        self.assertEqual(len(rows), 93)
        self.assertEqual(set(remember.summarize(rows)), set(remember.GOALS))


if __name__ == "__main__":
    unittest.main()
