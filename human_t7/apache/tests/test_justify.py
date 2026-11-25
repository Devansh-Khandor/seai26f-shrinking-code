"""Behavioral tests for Apache T7 Justify."""

import math
import sys
import unittest
from pathlib import Path

APACHE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APACHE_DIR))

from justify import (
    compare_configurations,
    explain_prediction,
    recommend_configuration,
    validate_explanation,
)
from guess import (
    one_setting_pairs,
    predict_performance,
    train_test_split,
)
from remember import GOAL, adjustable_settings, load_configurations


def make_row(number, a, b, performance):
    return {
        "row_number": number,
        "base": 1,
        "a": a,
        "b": b,
        GOAL: float(performance),
    }


class TestJustify(unittest.TestCase):

    def setUp(self):
        self.settings = ["a", "b"]

        self.history = [
            make_row(1, 0, 0, 10),
            make_row(2, 0, 1, 20),
            make_row(3, 1, 0, 70),
            make_row(4, 1, 1, 80),
        ]

        self.candidates = [
            make_row(5, 0, 0, 9999),
            make_row(6, 1, 1, -9999),
        ]

    def test_explanation_uses_historical_neighbors(self):
        result = explain_prediction(
            self.candidates[0],
            self.history,
            self.settings,
            k=2,
        )

        self.assertEqual(result["prediction"], 15)
        self.assertEqual(
            [r["row_number"] for r in result["evidence"]],
            [1, 2],
        )

    def test_explanation_settings_are_correct(self):
        result = explain_prediction(
            self.candidates[0],
            self.history,
            self.settings,
            k=2,
        )

        self.assertEqual(result["settings"], {"a": 0, "b": 0})
        self.assertEqual(
            result["evidence"][0]["differing_settings"],
            [],
        )
        self.assertEqual(
            result["evidence"][1]["differing_settings"],
            ["b"],
        )

    def test_distances_are_correct(self):
        result = explain_prediction(
            self.candidates[0],
            self.history,
            self.settings,
            k=2,
        )

        self.assertEqual(
            [r["distance"] for r in result["evidence"]],
            [0.0, 0.5],
        )

    def test_recommendation_uses_predictions_not_targets(self):
        selected, explanation = recommend_configuration(
            self.candidates,
            self.history,
            self.settings,
            k=2,
        )

        self.assertEqual(selected["row_number"], 5)
        self.assertEqual(explanation["prediction"], 15)

        # Candidate 6 has an artificially excellent recorded value,
        # but that outcome must not influence the selection.
        self.assertEqual(self.candidates[1][GOAL], -9999)

    def test_recommendation_invariant_to_targets(self):
        changed = [row.copy() for row in self.candidates]
        changed[0][GOAL] = -1_000_000
        changed[1][GOAL] = 1_000_000

        original, _ = recommend_configuration(
            self.candidates, self.history, self.settings, k=2
        )
        revised, _ = recommend_configuration(
            changed, self.history, self.settings, k=2
        )

        self.assertEqual(
            original["row_number"],
            revised["row_number"],
        )

    def test_empty_candidates_rejected(self):
        with self.assertRaises(ValueError):
            recommend_configuration(
                [], self.history, self.settings
            )

    def test_comparison_lists_exact_changes(self):
        result = compare_configurations(
            self.candidates[0],
            self.candidates[1],
            self.history,
            self.settings,
            k=2,
        )

        self.assertEqual(
            [r["setting"] for r in result["differences"]],
            ["a", "b"],
        )
        self.assertEqual(
            result["predicted_difference"], 35
        )

    def test_same_record_comparison_rejected(self):
        with self.assertRaises(ValueError):
            compare_configurations(
                self.candidates[0],
                self.candidates[0],
                self.history,
                self.settings,
            )

    def test_evidence_validates(self):
        result = explain_prediction(
            self.candidates[0],
            self.history,
            self.settings,
            k=2,
        )

        self.assertTrue(
            validate_explanation(
                result, self.history, self.settings, k=2
            )
        )

    def test_fabricated_evidence_rejected(self):
        result = explain_prediction(
            self.candidates[0],
            self.history,
            self.settings,
            k=2,
        )

        result["evidence"][0]["performance"] = 123456

        self.assertFalse(
            validate_explanation(
                result, self.history, self.settings, k=2
            )
        )

    def test_real_dataset_recommendation(self):
        rows = load_configurations()
        settings = adjustable_settings(rows)
        history, candidates = train_test_split(rows)

        selected, explanation = recommend_configuration(
            candidates, history, settings
        )

        self.assertEqual(len(history), 154)
        self.assertEqual(len(candidates), 38)

        self.assertIn(selected, candidates)
        self.assertNotIn(
            selected["row_number"],
            {row["row_number"] for row in history},
        )
        self.assertEqual(len(explanation["evidence"]), 5)
        self.assertTrue(
            validate_explanation(explanation, history, settings)
        )

        predicted_values = [
            predict_performance(
                row, history, settings
            )["prediction"]
            for row in candidates
        ]

        self.assertTrue(
            math.isclose(
                explanation["prediction"],
                min(predicted_values),
            )
        )

    def test_real_pair_and_explanations(self):
        rows = load_configurations()
        settings = adjustable_settings(rows)
        history, candidates = train_test_split(rows)
        pairs = one_setting_pairs(candidates, settings)

        self.assertGreater(len(pairs), 0)

        a, b, changed = pairs[0]
        result = compare_configurations(
            a, b, history, settings
        )

        self.assertEqual(len(result["differences"]), 1)
        self.assertEqual(
            result["differences"][0]["setting"], changed
        )
        self.assertTrue(
            validate_explanation(
                result["prediction_a"], history, settings
            )
        )
        self.assertTrue(
            validate_explanation(
                result["prediction_b"], history, settings
            )
        )


if __name__ == "__main__":
    unittest.main()
