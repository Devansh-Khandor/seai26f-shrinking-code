"""Behavioral tests for Apache T7 Guess."""

import math
import sys
import unittest
from pathlib import Path

APACHE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APACHE_DIR))

from remember import GOAL, load_configurations, adjustable_settings
from guess import (
    configuration_distance,
    differing_settings,
    estimate_pair_difference,
    evaluate_predictions,
    mean_absolute_error,
    nearest_neighbors,
    one_setting_pairs,
    predict_performance,
    train_test_split,
)


def make_row(number, a, b, performance):
    return {
        "row_number": number,
        "base": 1,
        "a": a,
        "b": b,
        GOAL: float(performance),
    }


class TestGuess(unittest.TestCase):

    def setUp(self):
        self.settings = ["a", "b"]
        self.history = [
            make_row(1, 0, 0, 10),
            make_row(2, 0, 1, 20),
            make_row(3, 1, 0, 30),
            make_row(4, 1, 1, 40),
        ]

    def test_distance(self):
        self.assertEqual(
            configuration_distance(
                self.history[0], self.history[0],
                self.settings
            ),
            0.0,
        )
        self.assertEqual(
            configuration_distance(
                self.history[0], self.history[3],
                self.settings
            ),
            1.0,
        )
        self.assertEqual(
            configuration_distance(
                self.history[0], self.history[1],
                self.settings
            ),
            0.5,
        )

    def test_known_prediction(self):
        query = make_row(99, 0, 0, 9999)
        result = predict_performance(
            query, self.history, self.settings, k=2
        )
        self.assertEqual(result["prediction"], 15)
        self.assertEqual(result["neighbor_rows"], [1, 2])

    def test_no_target_leakage(self):
        first = make_row(99, 0, 0, -1000)
        second = make_row(99, 0, 0, 999999)

        a = predict_performance(
            first, self.history, self.settings, k=2
        )
        b = predict_performance(
            second, self.history, self.settings, k=2
        )

        self.assertEqual(a["prediction"], b["prediction"])

    def test_mae(self):
        self.assertEqual(
            mean_absolute_error([10, 20], [8, 24]),
            3.0,
        )

    def test_split_deterministic_and_disjoint(self):
        rows = [
            make_row(i, i % 2, (i // 2) % 2, i)
            for i in range(192)
        ]

        train1, test1 = train_test_split(rows)
        train2, test2 = train_test_split(rows)

        self.assertEqual(len(train1), 154)
        self.assertEqual(len(test1), 38)

        ids1 = {r["row_number"] for r in train1}
        ids2 = {r["row_number"] for r in test1}

        self.assertFalse(ids1 & ids2)
        self.assertEqual(len(ids1 | ids2), 192)
        self.assertEqual(
            [r["row_number"] for r in train1],
            [r["row_number"] for r in train2],
        )
        self.assertEqual(
            [r["row_number"] for r in test1],
            [r["row_number"] for r in test2],
        )

    def test_one_setting_pairs(self):
        pairs = one_setting_pairs(
            self.history, self.settings
        )
        self.assertEqual(len(pairs), 4)
        self.assertTrue(all(
            len(differing_settings(a, b, self.settings)) == 1
            for a, b, _ in pairs
        ))

    def test_invalid_pair_rejected(self):
        with self.assertRaises(ValueError):
            estimate_pair_difference(
                self.history[0],
                self.history[3],
                self.history,
                self.settings,
                k=2,
            )

    def test_valid_pair(self):
        result = estimate_pair_difference(
            self.history[0],
            self.history[1],
            self.history,
            self.settings,
            k=2,
        )
        self.assertEqual(result["changed_setting"], "b")
        self.assertTrue(
            math.isfinite(result["predicted_difference"])
        )

    def test_baseline_calculation(self):
        train = self.history[:3]
        test = [self.history[3]]

        result = evaluate_predictions(
            train, test, self.settings, k=2
        )

        self.assertEqual(result["baseline_prediction"], 20)
        self.assertEqual(result["baseline_mae"], 20)

    def test_real_dataset_prediction(self):
        rows = load_configurations()
        settings = adjustable_settings(rows)
        train, test = train_test_split(rows)

        result = evaluate_predictions(train, test, settings)

        self.assertEqual(len(train), 154)
        self.assertEqual(len(test), 38)
        self.assertEqual(len(result["predicted"]), 38)
        self.assertTrue(math.isfinite(result["mae"]))
        self.assertTrue(
            math.isfinite(result["baseline_mae"])
        )

    def test_real_pair_uses_held_out_records(self):
        rows = load_configurations()
        settings = adjustable_settings(rows)
        train, test = train_test_split(rows)

        train_ids = {r["row_number"] for r in train}
        pairs = one_setting_pairs(test, settings)

        self.assertGreater(len(pairs), 0)

        a, b, changed = pairs[0]

        self.assertNotIn(a["row_number"], train_ids)
        self.assertNotIn(b["row_number"], train_ids)
        self.assertEqual(
            differing_settings(a, b, settings),
            [changed],
        )

        result = estimate_pair_difference(
            a, b, train, settings
        )
        self.assertTrue(
            math.isfinite(result["predicted_difference"])
        )


if __name__ == "__main__":
    unittest.main()
