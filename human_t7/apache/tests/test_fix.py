"""Behavioral tests for Apache T7 Fix."""

import math
import sys
import unittest
from pathlib import Path

APACHE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APACHE_DIR))

from fix import (
    find_minimum_repair,
    predict_without_self,
    recommend_small_repair,
    valid_alternatives,
)
from guess import differing_settings, train_test_split
from remember import GOAL, adjustable_settings, load_configurations


def make_row(number, a, b, c, performance):
    return {
        "row_number": number,
        "base": 1,
        "a": a,
        "b": b,
        "c": c,
        GOAL: float(performance),
    }


class TestFix(unittest.TestCase):

    def setUp(self):
        self.settings = ["a", "b", "c"]
        self.history = [
            make_row(1, 0, 0, 0, 100),
            make_row(2, 0, 0, 1, 200),
            make_row(3, 0, 1, 0, 300),
            make_row(4, 0, 1, 1, 400),
            make_row(5, 1, 0, 0, 500),
            make_row(6, 1, 0, 1, 600),
            make_row(7, 1, 1, 0, 700),
            make_row(8, 1, 1, 1, 800),
        ]
        self.source = make_row(99, 1, 1, 1, 9999)
        self.candidates = self.history + [self.source]

    def test_self_exclusion(self):
        row = self.history[0]
        outcome = predict_without_self(
            row, self.history, self.settings, k=2
        )

        # Row 1 must not be among its own neighbors.
        self.assertNotIn(1, outcome["neighbor_rows"])

    def test_source_target_not_used_for_prediction(self):
        modified = self.source.copy()
        modified[GOAL] = -999999

        first = predict_without_self(
            self.source, self.history, self.settings
        )
        second = predict_without_self(
            modified, self.history, self.settings
        )

        self.assertEqual(first["prediction"],
                         second["prediction"])

    def test_only_observed_alternatives(self):
        _, alternatives = valid_alternatives(
            self.source, self.candidates,
            self.history, self.settings
        )

        ids = {row["row_number"] for row in self.candidates}
        self.assertEqual(len(alternatives), 7)
        self.assertNotIn(
            8,
            {item["row_number"] for item in alternatives},
        )
        self.assertTrue(all(
            item["row_number"] in ids
            for item in alternatives
        ))
        self.assertNotIn(
            99, {item["row_number"] for item in alternatives}
        )

    def test_change_count_matches_hamming_distance(self):
        _, alternatives = valid_alternatives(
            self.source, self.candidates,
            self.history, self.settings
        )

        by_id = {
            row["row_number"]: row for row in self.candidates
        }

        for item in alternatives:
            candidate = by_id[item["row_number"]]
            self.assertEqual(
                item["changes"],
                len(differing_settings(
                    self.source, candidate, self.settings
                )),
            )
            self.assertEqual(
                item["changes"],
                len(item["setting_changes"]),
            )

    def test_small_repair_has_improvement(self):
        result = recommend_small_repair(
            self.source, self.candidates,
            self.history, self.settings,
            max_changes=2
        )

        recommendation = result["recommendation"]
        self.assertIsNotNone(recommendation)
        self.assertLessEqual(recommendation["changes"], 2)
        self.assertGreater(
            recommendation["predicted_improvement"], 0
        )

    def test_small_repair_minimizes_edits(self):
        _, alternatives = valid_alternatives(
            self.source, self.candidates,
            self.history, self.settings
        )
        result = recommend_small_repair(
            self.source, self.candidates,
            self.history, self.settings,
            max_changes=2
        )

        eligible = [
            item for item in alternatives
            if item["changes"] <= 2
            and item["predicted_improvement"] > 1e-9
        ]

        self.assertEqual(
            result["recommendation"]["changes"],
            min(item["changes"] for item in eligible),
        )

    def test_minimum_repair_meets_target(self):
        _, alternatives = valid_alternatives(
            self.source, self.candidates,
            self.history, self.settings
        )
        target = min(
            item["prediction"] for item in alternatives
        ) + 1

        result = find_minimum_repair(
            self.source, self.candidates,
            self.history, self.settings,
            target=target
        )

        recommendation = result["recommendation"]
        self.assertIsNotNone(recommendation)
        self.assertLessEqual(
            recommendation["prediction"], target
        )

        eligible = [
            item for item in alternatives
            if item["prediction"] <= target
        ]
        self.assertEqual(
            recommendation["changes"],
            min(item["changes"] for item in eligible),
        )

    def test_unreachable_target_returns_none(self):
        result = find_minimum_repair(
            self.source, self.candidates,
            self.history, self.settings,
            target=-1
        )
        self.assertIsNone(result["recommendation"])

    def test_invalid_target_rejected(self):
        with self.assertRaises(ValueError):
            find_minimum_repair(
                self.source, self.candidates,
                self.history, self.settings,
                target=math.nan
            )

    def test_invalid_edit_limit_rejected(self):
        with self.assertRaises(ValueError):
            recommend_small_repair(
                self.source, self.candidates,
                self.history, self.settings,
                max_changes=0
            )

    def test_constant_setting_cannot_change(self):
        invalid = make_row(100, 0, 0, 0, 1)
        invalid["base"] = 0

        _, alternatives = valid_alternatives(
            self.source,
            self.candidates + [invalid],
            self.history,
            self.settings
        )

        self.assertNotIn(
            100, {item["row_number"] for item in alternatives}
        )

    def test_real_dataset_and_target_leakage(self):
        rows = load_configurations()
        settings = adjustable_settings(rows)
        train, test = train_test_split(rows)

        source = max(
            test,
            key=lambda row: (row[GOAL], -row["row_number"])
        )

        result = recommend_small_repair(
            source, rows, train, settings
        )
        _, alternatives = valid_alternatives(
            source, rows, train, settings
        )

        self.assertGreater(len(alternatives), 0)

        recommendation = result["recommendation"]
        if recommendation is not None:
            self.assertIn(
                recommendation["row_number"],
                {row["row_number"] for row in rows},
            )
            self.assertLessEqual(
                recommendation["changes"], 2
            )

        # Alter held-out outcomes without altering settings.
        # This must not change the selected repair for a fixed source.
        test_ids = {row["row_number"] for row in test}
        modified_rows = [row.copy() for row in rows]

        for row in modified_rows:
            if row["row_number"] in test_ids:
                row[GOAL] = -123456

        revised = recommend_small_repair(
            source, modified_rows, train, settings
        )

        self.assertEqual(
            result["recommendation"],
            revised["recommendation"],
        )


if __name__ == "__main__":
    unittest.main()
