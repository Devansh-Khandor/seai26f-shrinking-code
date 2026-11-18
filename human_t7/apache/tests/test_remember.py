"""Behavioral tests for Apache T7 Remember."""

import sys
import unittest
from pathlib import Path

APACHE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APACHE_DIR))

from remember import (
    GOAL,
    adjustable_settings,
    favorable_configurations,
    load_configurations,
    rank_configurations,
    setting_differences,
    summarize,
)


def make_row(number, performance, keep_alive):
    return {
        "row_number": number,
        "base": 1,
        "keepAlive": keep_alive,
        GOAL: float(performance),
    }


class TestRemember(unittest.TestCase):

    def setUp(self):
        self.rows = [
            make_row(i, value, int(i <= 4))
            for i, value in enumerate(
                [10, 20, 30, 40, 50, 60, 70, 80],
                start=1,
            )
        ]

    def test_summary(self):
        stats = summarize(self.rows)
        self.assertEqual(stats["count"], 8)
        self.assertEqual(stats["minimum"], 10)
        self.assertEqual(stats["maximum"], 80)
        self.assertEqual(stats["median"], 45)
        self.assertEqual(stats["mean"], 45)

    def test_ranking_minimizes_performance(self):
        reversed_rows = list(reversed(self.rows))
        ranked = rank_configurations(reversed_rows)

        self.assertEqual(ranked[0]["row_number"], 1)
        self.assertEqual(ranked[-1]["row_number"], 8)

    def test_favorable_selection(self):
        favorable, remaining, cutoff = (
            favorable_configurations(self.rows)
        )

        self.assertEqual(cutoff, 20)
        self.assertEqual(len(favorable), 2)
        self.assertEqual(len(remaining), 6)
        self.assertEqual(
            [row["row_number"] for row in favorable],
            [1, 2],
        )

    def test_favorable_ties(self):
        rows = [
            make_row(i, value, 0)
            for i, value in enumerate(
                [10, 20, 20, 40, 50, 60, 70, 80],
                start=1,
            )
        ]

        favorable, remaining, cutoff = (
            favorable_configurations(rows)
        )

        self.assertEqual(cutoff, 20)
        self.assertEqual(len(favorable), 3)
        self.assertEqual(len(remaining), 5)

    def test_constant_excluded(self):
        self.assertEqual(
            adjustable_settings(self.rows),
            ["keepAlive"],
        )

    def test_setting_difference(self):
        favorable, remaining, _ = (
            favorable_configurations(self.rows)
        )

        result = setting_differences(
            favorable, remaining, ["keepAlive"]
        )[0]

        self.assertAlmostEqual(
            result["favorable_rate"], 1.0
        )
        self.assertAlmostEqual(
            result["remaining_rate"], 2 / 6
        )
        self.assertAlmostEqual(
            result["difference"], 2 / 3
        )

    def test_real_dataset(self):
        rows = load_configurations()
        self.assertEqual(len(rows), 192)
        self.assertEqual(len(adjustable_settings(rows)), 8)
        self.assertEqual({row["base"] for row in rows}, {1})

        stats = summarize(rows)
        self.assertLessEqual(
            stats["minimum"], stats["median"]
        )
        self.assertLessEqual(
            stats["median"], stats["maximum"]
        )


if __name__ == "__main__":
    unittest.main()
