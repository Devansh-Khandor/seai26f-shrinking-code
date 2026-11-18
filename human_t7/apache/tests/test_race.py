"""Behavioral tests for Apache T7 Race."""

import math
import sys
import unittest
from pathlib import Path

APACHE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APACHE_DIR))

from race import (
    bootstrap_mean_interval,
    compare_selection,
    evaluate_prediction_race,
    paired_error_summary,
    random_inspection_trials,
)
from choose import plan_inspections, oracle_best
from guess import train_test_split
from remember import GOAL, adjustable_settings, load_configurations


def make_row(number, a, b, performance):
    return {
        "row_number": number,
        "base": 1,
        "a": a,
        "b": b,
        GOAL: float(performance),
    }


class TestRace(unittest.TestCase):

    def setUp(self):
        self.settings = ["a", "b"]

        self.history = [
            make_row(1, 0, 0, 10),
            make_row(2, 0, 1, 20),
            make_row(3, 1, 0, 70),
            make_row(4, 1, 1, 80),
        ]

        self.candidates = [
            make_row(5, 0, 0, 50),
            make_row(6, 1, 1, 5),
            make_row(7, 0, 1, 30),
        ]

        self.plan = plan_inspections(
            self.candidates,
            self.history,
            self.settings,
            k=2,
        )

    def test_random_trial_count(self):
        results = random_inspection_trials(
            self.candidates, budgets=(1, 2, 3),
            trials=30,
        )

        for budget in (1, 2, 3):
            self.assertEqual(len(results[budget]), 30)

    def test_random_trials_reproducible(self):
        first = random_inspection_trials(
            self.candidates, budgets=(1, 2, 3),
            trials=30, seed=42,
        )
        second = random_inspection_trials(
            self.candidates, budgets=(1, 2, 3),
            trials=30, seed=42,
        )

        self.assertEqual(first, second)

    def test_random_budget_monotonicity(self):
        results = random_inspection_trials(
            self.candidates, budgets=(1, 2, 3),
            trials=30,
        )

        for trial in range(30):
            self.assertGreaterEqual(
                results[1][trial], results[2][trial]
            )
            self.assertGreaterEqual(
                results[2][trial], results[3][trial]
            )

    def test_full_budget_reaches_pool_optimum(self):
        results = random_inspection_trials(
            self.candidates, budgets=(3,), trials=30
        )

        self.assertEqual(results[3], [5] * 30)

    def test_invalid_random_budget(self):
        with self.assertRaises(ValueError):
            random_inspection_trials(
                self.candidates, budgets=(4,)
            )

        with self.assertRaises(ValueError):
            random_inspection_trials(
                self.candidates, budgets=(0,)
            )

    def test_random_duplicate_ids_rejected(self):
        duplicated = self.candidates + [
            self.candidates[0].copy()
        ]

        with self.assertRaises(ValueError):
            random_inspection_trials(duplicated)

    def test_selection_comparison_accounting(self):
        result = compare_selection(
            self.plan, self.candidates,
            budgets=(1, 2, 3),
            trials=30,
        )

        self.assertEqual(len(result), 3)

        for item in result:
            self.assertEqual(
                item["guided_wins"]
                + item["ties"]
                + item["random_wins"],
                30,
            )
            self.assertEqual(len(item["random_values"]), 30)

    def test_guided_full_budget_equals_oracle(self):
        result = compare_selection(
            self.plan, self.candidates,
            budgets=(3,), trials=30,
        )[0]

        self.assertEqual(
            result["guided_best"],
            oracle_best(self.candidates)["performance"],
        )
        self.assertEqual(result["ties"], 30)

    def test_known_paired_error_comparison(self):
        result = paired_error_summary(
            actual=[10, 20, 30],
            model_pred=[12, 18, 28],
            baseline_pred=[20, 20, 20],
        )

        self.assertAlmostEqual(result["model_mae"], 2)
        self.assertAlmostEqual(
            result["baseline_mae"], 20 / 3
        )
        self.assertAlmostEqual(
            result["mae_improvement"], 14 / 3
        )
        self.assertEqual(result["model_better"], 2)
        self.assertEqual(result["baseline_better"], 1)
        self.assertEqual(result["equal"], 0)
        self.assertEqual(
            result["paired_improvements"], [8, -2, 8]
        )

    def test_identical_predictors_tie(self):
        result = paired_error_summary(
            actual=[10, 20],
            model_pred=[12, 18],
            baseline_pred=[12, 18],
        )

        self.assertEqual(result["mae_improvement"], 0)
        self.assertEqual(result["equal"], 2)

    def test_misaligned_predictions_rejected(self):
        with self.assertRaises(ValueError):
            paired_error_summary(
                actual=[1, 2],
                model_pred=[1],
                baseline_pred=[1, 2],
            )

    def test_bootstrap_constant_improvement(self):
        interval = bootstrap_mean_interval(
            [5.0] * 20, repeats=200, seed=42
        )

        self.assertEqual(interval, (5.0, 5.0))

    def test_bootstrap_reproducibility(self):
        values = [-3.0, 1.0, 4.0, 9.0]

        first = bootstrap_mean_interval(
            values, repeats=200, seed=42
        )
        second = bootstrap_mean_interval(
            values, repeats=200, seed=42
        )

        self.assertEqual(first, second)
        self.assertLessEqual(first[0], first[1])

    def test_real_dataset_selection(self):
        rows = load_configurations()
        settings = adjustable_settings(rows)
        train, test = train_test_split(rows)

        plan = plan_inspections(test, train, settings)
        comparisons = compare_selection(
            plan, test, budgets=(5, 10, 20)
        )

        self.assertEqual(len(train), 154)
        self.assertEqual(len(test), 38)
        self.assertEqual(len(comparisons), 3)

        oracle = oracle_best(test)["performance"]

        for result in comparisons:
            self.assertGreaterEqual(
                result["guided_best"], oracle
            )
            self.assertTrue(all(
                value >= oracle
                for value in result["random_values"]
            ))
            self.assertEqual(
                result["guided_wins"]
                + result["ties"]
                + result["random_wins"],
                30,
            )

    def test_real_dataset_prediction(self):
        rows = load_configurations()
        settings = adjustable_settings(rows)
        train, test = train_test_split(rows)

        result = evaluate_prediction_race(
            train, test, settings
        )

        self.assertEqual(result["count"], 38)
        self.assertEqual(
            result["model_better"]
            + result["baseline_better"]
            + result["equal"],
            38,
        )
        self.assertTrue(math.isfinite(result["model_mae"]))
        self.assertTrue(
            math.isfinite(result["baseline_mae"])
        )

        # Reconcile with the previously recorded Guess results.
        self.assertAlmostEqual(
            result["model_mae"], 174.79, places=2
        )
        self.assertAlmostEqual(
            result["baseline_mae"], 484.05, places=2
        )


if __name__ == "__main__":
    unittest.main()
