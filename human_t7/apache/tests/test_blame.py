"""Behavioral tests for Apache T7 Blame."""

import sys
import unittest
from pathlib import Path

APACHE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APACHE_DIR))

from blame import (
    analyze_individual_settings,
    analyze_setting_pairs,
    prevalence,
    split_poor_performers,
)
from remember import (
    GOAL,
    adjustable_settings,
    load_configurations,
)


def make_row(number, performance, a, b):
    return {
        "row_number": number,
        "base": 1,
        "a": a,
        "b": b,
        GOAL: float(performance),
    }


class TestBlame(unittest.TestCase):

    def setUp(self):
        self.rows = [
            make_row(1, 10, 0, 0),
            make_row(2, 20, 0, 1),
            make_row(3, 30, 0, 0),
            make_row(4, 40, 0, 1),
            make_row(5, 50, 1, 0),
            make_row(6, 60, 1, 1),
            make_row(7, 70, 1, 1),
            make_row(8, 80, 1, 1),
        ]

    def test_poor_group_selects_high_values(self):
        poor, other, cutoff = split_poor_performers(self.rows)

        self.assertEqual(cutoff, 70)
        self.assertEqual(len(poor), 2)
        self.assertEqual(len(other), 6)
        self.assertEqual(
            {r["row_number"] for r in poor}, {7, 8}
        )

    def test_ties_included(self):
        rows = [
            make_row(i, value, 0, 0)
            for i, value in enumerate(
                [10, 20, 30, 40, 50, 60, 70, 70],
                start=1,
            )
        ]

        poor, other, cutoff = split_poor_performers(rows)

        self.assertEqual(cutoff, 70)
        self.assertEqual(len(poor), 2)
        self.assertEqual(len(other), 6)

    def test_prevalence(self):
        self.assertEqual(
            prevalence(self.rows, (("a", 1),)), 0.5
        )
        self.assertEqual(
            prevalence(
                self.rows, (("a", 1), ("b", 1))
            ),
            3 / 8,
        )

    def test_individual_setting_enrichment(self):
        poor, other, _ = split_poor_performers(self.rows)

        results = analyze_individual_settings(
            poor, other, ["a", "b"]
        )

        top = results[0]

        self.assertEqual(top["setting"], "a")
        self.assertEqual(top["value"], 1)
        self.assertAlmostEqual(top["poor_rate"], 1)
        self.assertAlmostEqual(top["other_rate"], 2 / 6)
        self.assertAlmostEqual(top["difference"], 2 / 3)

    def test_pair_enrichment(self):
        poor, other, _ = split_poor_performers(self.rows)

        results = analyze_setting_pairs(
            poor, other, ["a", "b"], min_support=2
        )

        self.assertEqual(len(results), 1)

        pair = results[0]

        self.assertEqual(pair["first_setting"], "a")
        self.assertEqual(pair["first_value"], 1)
        self.assertEqual(pair["second_setting"], "b")
        self.assertEqual(pair["second_value"], 1)
        self.assertEqual(pair["poor_count"], 2)
        self.assertEqual(pair["other_count"], 1)
        self.assertAlmostEqual(
            pair["difference"], 1 - 1 / 6
        )

    def test_pair_minimum_support(self):
        poor, other, _ = split_poor_performers(self.rows)

        self.assertEqual(
            analyze_setting_pairs(
                poor, other, ["a", "b"], min_support=3
            ),
            [],
        )

    def test_invalid_fraction(self):
        with self.assertRaises(ValueError):
            split_poor_performers(self.rows, fraction=0)

        with self.assertRaises(ValueError):
            split_poor_performers(self.rows, fraction=1)

    def test_empty_prevalence_rejected(self):
        with self.assertRaises(ValueError):
            prevalence([], (("a", 1),))

    def test_constant_feature_excluded(self):
        self.assertNotIn(
            "base", adjustable_settings(self.rows)
        )

    def test_real_dataset(self):
        rows = load_configurations()
        settings = adjustable_settings(rows)

        poor, other, cutoff = split_poor_performers(rows)

        self.assertEqual(len(poor) + len(other), 192)
        self.assertGreater(len(poor), 0)
        self.assertGreater(len(other), 0)
        self.assertTrue(
            all(row[GOAL] >= cutoff for row in poor)
        )

        individual = analyze_individual_settings(
            poor, other, settings
        )
        pairs = analyze_setting_pairs(
            poor, other, settings
        )

        self.assertTrue(all(
            item["difference"] > 0
            for item in individual + pairs
        ))
        self.assertTrue(all(
            item["poor_count"] >= 5
            for item in pairs
        ))


if __name__ == "__main__":
    unittest.main()
