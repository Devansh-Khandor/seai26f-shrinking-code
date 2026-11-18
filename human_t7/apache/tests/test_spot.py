"""Behavioral tests for Apache T7 Spot."""

import math
import sys
import unittest
from pathlib import Path

APACHE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APACHE_DIR))

from spot import (
    analyze_anomalies,
    configuration_anomaly,
    outcome_anomaly,
    training_thresholds,
)
from guess import train_test_split
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


class TestSpot(unittest.TestCase):

    def setUp(self):
        self.settings = ["a", "b", "c"]
        self.history = [
            make_row(1, 0, 0, 0, 10),
            make_row(2, 0, 0, 1, 20),
            make_row(3, 0, 1, 0, 30),
            make_row(4, 1, 0, 0, 40),
            make_row(5, 1, 1, 1, 50),
            make_row(6, 1, 1, 0, 60),
        ]
        self.query = make_row(99, 0, 0, 0, 100)

    def test_known_outcome_residual(self):
        result = outcome_anomaly(
            self.query, self.history, self.settings, k=2
        )

        self.assertEqual(result["predicted"], 15)
        self.assertEqual(result["residual"], 85)
        self.assertEqual(result["absolute_residual"], 85)
        self.assertEqual(result["neighbor_rows"], [1, 2])

    def test_negative_residual(self):
        query = make_row(99, 0, 0, 0, 5)
        result = outcome_anomaly(
            query, self.history, self.settings, k=2
        )

        self.assertEqual(result["residual"], -10)
        self.assertEqual(result["absolute_residual"], 10)

    def test_configuration_distance_calculation(self):
        result = configuration_anomaly(
            self.query, self.history, self.settings, k=2
        )

        self.assertAlmostEqual(
            result["isolation_score"], 1 / 6
        )
        self.assertEqual(result["nearest_distance"], 0)
        self.assertEqual(result["neighbor_rows"], [1, 2])

    def test_configuration_score_ignores_target(self):
        changed = self.query.copy()
        changed[GOAL] = 1_000_000

        first = configuration_anomaly(
            self.query, self.history, self.settings, k=3
        )
        second = configuration_anomaly(
            changed, self.history, self.settings, k=3
        )

        self.assertEqual(first, second)

    def test_outcome_prediction_does_not_use_query_target(self):
        changed = self.query.copy()
        changed[GOAL] = -1_000_000

        first = outcome_anomaly(
            self.query, self.history, self.settings, k=2
        )
        second = outcome_anomaly(
            changed, self.history, self.settings, k=2
        )

        self.assertEqual(
            first["predicted"], second["predicted"]
        )
        self.assertNotEqual(
            first["absolute_residual"],
            second["absolute_residual"],
        )

    def test_self_neighbor_rejected(self):
        for detector in (outcome_anomaly, configuration_anomaly):
            with self.assertRaises(ValueError):
                detector(
                    self.history[0],
                    self.history,
                    self.settings,
                    k=2,
                )

    def test_training_thresholds_from_history(self):
        thresholds = training_thresholds(
            self.history, self.settings, k=2
        )

        self.assertEqual(
            len(thresholds["training_residuals"]), 6
        )
        self.assertEqual(
            len(thresholds["training_isolation_scores"]), 6
        )
        self.assertGreaterEqual(
            thresholds["residual_threshold"], 0
        )
        self.assertGreaterEqual(
            thresholds["isolation_threshold"], 0
        )

    def test_threshold_not_affected_by_test_outcomes(self):
        test = [self.query]
        modified = [self.query.copy()]
        modified[0][GOAL] = 999_999

        first = analyze_anomalies(
            test, self.history, self.settings, k=2
        )
        second = analyze_anomalies(
            modified, self.history, self.settings, k=2
        )

        self.assertEqual(
            first["thresholds"],
            second["thresholds"],
        )
        self.assertEqual(
            first["configuration_results"],
            second["configuration_results"],
        )

    def test_extreme_performance_is_flagged(self):
        history = [
            make_row(i, 0, 0, 0, 10)
            for i in range(1, 7)
        ]
        extreme = make_row(99, 0, 0, 0, 1000)

        result = analyze_anomalies(
            [extreme], history, self.settings, k=2
        )

        self.assertEqual(
            result["thresholds"]["residual_threshold"], 0
        )
        self.assertTrue(
            result["outcome_results"][0]["flagged"]
        )
        self.assertFalse(
            result["configuration_results"][0]["flagged"]
        )

    def test_unusual_settings_can_be_flagged_independently(self):
        history = [
            make_row(i, 0, 0, 0, 10)
            for i in range(1, 7)
        ]
        unusual = make_row(99, 1, 1, 1, 10)

        result = analyze_anomalies(
            [unusual], history, self.settings, k=2
        )

        self.assertTrue(
            result["configuration_results"][0]["flagged"]
        )
        self.assertFalse(
            result["outcome_results"][0]["flagged"]
        )

    def test_results_sorted_by_anomaly_strength(self):
        test = [
            make_row(99, 0, 0, 0, 100),
            make_row(100, 1, 1, 1, 1000),
            make_row(101, 0, 1, 1, 10),
        ]

        result = analyze_anomalies(
            test, self.history, self.settings, k=2
        )

        residuals = [
            item["absolute_residual"]
            for item in result["outcome_results"]
        ]
        scores = [
            item["isolation_score"]
            for item in result["configuration_results"]
        ]

        self.assertEqual(residuals, sorted(residuals, reverse=True))
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_real_dataset(self):
        rows = load_configurations()
        settings = adjustable_settings(rows)
        train, test = train_test_split(rows)

        result = analyze_anomalies(test, train, settings)

        self.assertEqual(len(train), 154)
        self.assertEqual(len(test), 38)
        self.assertEqual(len(result["outcome_results"]), 38)
        self.assertEqual(len(result["configuration_results"]), 38)

        train_ids = {row["row_number"] for row in train}

        for item in result["outcome_results"]:
            self.assertTrue(math.isfinite(item["absolute_residual"]))
            self.assertTrue(
                set(item["neighbor_rows"]).issubset(train_ids)
            )

        for item in result["configuration_results"]:
            self.assertGreaterEqual(item["isolation_score"], 0)
            self.assertLessEqual(item["isolation_score"], 1)
            self.assertTrue(
                set(item["neighbor_rows"]).issubset(train_ids)
            )


if __name__ == "__main__":
    unittest.main()
