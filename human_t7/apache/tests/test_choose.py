"""Behavioral tests for Apache T7 Choose."""

import sys
import unittest
from pathlib import Path

APACHE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APACHE_DIR))

from choose import (
    evaluate_budgets,
    first_discovery,
    oracle_best,
    plan_inspections,
    simulate_inspections,
)
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


class TestChoose(unittest.TestCase):

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

    def test_plan_ranks_by_prediction(self):
        plan = plan_inspections(
            self.candidates, self.history, self.settings, k=2
        )

        self.assertEqual(
            [item["row_number"] for item in plan],
            [5, 7, 6],
        )
        self.assertEqual(
            [item["priority"] for item in plan],
            [1, 2, 3],
        )

    def test_plan_does_not_use_candidate_targets(self):
        original = plan_inspections(
            self.candidates, self.history, self.settings, k=2
        )

        modified = [row.copy() for row in self.candidates]
        modified[0][GOAL] = -1000000
        modified[1][GOAL] = 1000000
        modified[2][GOAL] = -500000

        revised = plan_inspections(
            modified, self.history, self.settings, k=2
        )

        self.assertEqual(original, revised)

    def test_plan_is_deterministic(self):
        original = plan_inspections(
            self.candidates, self.history, self.settings, k=2
        )
        reordered = plan_inspections(
            list(reversed(self.candidates)),
            self.history,
            self.settings,
            k=2,
        )

        self.assertEqual(original, reordered)

    def test_budget_one_reveals_one_result(self):
        plan = plan_inspections(
            self.candidates, self.history, self.settings, k=2
        )
        outcome = simulate_inspections(
            plan, self.candidates, budget=1
        )

        self.assertEqual(len(outcome["inspected"]), 1)
        self.assertEqual(outcome["best_row"], 5)
        self.assertEqual(outcome["best_performance"], 50)

    def test_best_so_far_improves_monotonically(self):
        plan = plan_inspections(
            self.candidates, self.history, self.settings, k=2
        )
        outcome = simulate_inspections(
            plan, self.candidates, budget=3
        )

        best_values = [
            item["best_so_far"] for item in outcome["inspected"]
        ]

        self.assertEqual(best_values, [50, 30, 5])

    def test_budget_results(self):
        plan = plan_inspections(
            self.candidates, self.history, self.settings, k=2
        )
        results = evaluate_budgets(
            plan, self.candidates, budgets=(1, 2, 3)
        )

        self.assertEqual(
            [r["best_performance"] for r in results],
            [50, 30, 5],
        )

    def test_oracle_is_best_measured(self):
        result = oracle_best(self.candidates)

        self.assertEqual(result["row_number"], 6)
        self.assertEqual(result["performance"], 5)

    def test_first_discovery(self):
        plan = plan_inspections(
            self.candidates, self.history, self.settings, k=2
        )

        discovery = first_discovery(
            plan, self.candidates, target=5
        )

        self.assertEqual(discovery["inspection"], 3)
        self.assertEqual(discovery["row_number"], 6)

    def test_unreachable_target(self):
        plan = plan_inspections(
            self.candidates, self.history, self.settings, k=2
        )

        self.assertIsNone(
            first_discovery(plan, self.candidates, target=1)
        )

    def test_invalid_budget_rejected(self):
        plan = plan_inspections(
            self.candidates, self.history, self.settings, k=2
        )

        with self.assertRaises(ValueError):
            simulate_inspections(plan, self.candidates, budget=0)

        with self.assertRaises(ValueError):
            simulate_inspections(plan, self.candidates, budget=4)

    def test_training_overlap_rejected(self):
        with self.assertRaises(ValueError):
            plan_inspections(
                self.history[:1],
                self.history,
                self.settings,
                k=2,
            )

    def test_real_dataset_budgets(self):
        rows = load_configurations()
        settings = adjustable_settings(rows)
        train, test = train_test_split(rows)

        plan = plan_inspections(test, train, settings)

        self.assertEqual(len(plan), 38)
        self.assertEqual(
            sorted(p["row_number"] for p in plan),
            sorted(r["row_number"] for r in test),
        )

        results = evaluate_budgets(
            plan, test, budgets=(5, 10, 20)
        )

        self.assertEqual(
            [r["budget"] for r in results],
            [5, 10, 20],
        )

        best = [r["best_performance"] for r in results]

        self.assertTrue(
            all(
                best[i + 1] <= best[i]
                for i in range(len(best) - 1)
            )
        )

        full_pool_best = oracle_best(test)["performance"]

        self.assertTrue(
            all(value >= full_pool_best for value in best)
        )


if __name__ == "__main__":
    unittest.main()
