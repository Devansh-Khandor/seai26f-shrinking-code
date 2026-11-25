"""Regression tests for NASA T7's original Guess implementation, no source edits."""
import importlib.util
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NASA = ROOT / "human_t7" / "nasa"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


original_remember = load("original_nasa_guess_remember", NASA / "remember.py")
prior = sys.modules.get("remember")
try:
    sys.modules["remember"] = original_remember
    original_guess = load("original_nasa_guess", NASA / "guess.py")
finally:
    if prior is None:
        sys.modules.pop("remember", None)
    else:
        sys.modules["remember"] = prior


class TestOriginalNASAGuess(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"row_number": 1, "a": "x", "b": "m", "ignoredX": "A", "Kloc+": 1., "Effort-": 10., "Defects-": 3.},
            {"row_number": 2, "a": "x", "b": "n", "ignoredX": "B", "Kloc+": 2., "Effort-": 20., "Defects-": 7.},
            {"row_number": 3, "a": "y", "b": "n", "ignoredX": "C", "Kloc+": 3., "Effort-": 30., "Defects-": 11.},
        ]
        self.features = ["a", "b"]

    def test_features_exclude_goals_ignored_and_row_numbers(self):
        self.assertEqual(original_guess.get_features(self.rows), self.features)

    def test_distance_is_fraction_of_categorical_mismatches(self):
        self.assertEqual(original_guess.distance(self.rows[0], self.rows[0], self.features), 0.)
        self.assertEqual(original_guess.distance(self.rows[0], self.rows[1], self.features), .5)
        self.assertEqual(original_guess.distance(self.rows[0], self.rows[2], self.features), 1.)

    def test_neighbors_sorted_by_distance_then_row_id(self):
        candidate = dict(self.rows[1], row_number=9)
        neighbors = original_guess.nearest_neighbors(candidate, list(reversed(self.rows)), self.features, k=3)
        self.assertEqual([r["row_number"] for r in neighbors], [2, 1, 3])

    def test_predict_returns_mean_and_observed_neighbor_range(self):
        prediction = original_guess.predict(self.rows[0], self.rows[1:], self.features, "Effort-", k=2)
        self.assertEqual(prediction["estimate"], 25)
        self.assertEqual(prediction["low"], 20)
        self.assertEqual(prediction["high"], 30)

    def test_predictions_do_not_use_proposals_goals(self):
        proposal = self.rows[0].copy()
        before = original_guess.predict(proposal, self.rows[1:], self.features, "Defects-")
        proposal["Defects-"] = 10**9
        proposal["Effort-"] = -10**9
        after = original_guess.predict(proposal, self.rows[1:], self.features, "Defects-")
        self.assertEqual(before["estimate"], after["estimate"])
        self.assertEqual((before["low"], before["high"]), (after["low"], after["high"]))

    def test_training_mean_baseline_and_knn_mae(self):
        actual = [dict(self.rows[0], row_number=4)]
        model_mae, baseline_mae = original_guess.evaluate(self.rows[1:], actual, self.features, "Effort-")
        self.assertEqual(model_mae, 15)
        self.assertEqual(baseline_mae, 15)

    def test_full_nasa_feature_count_and_finite_predictions(self):
        projects = original_remember.load_projects()
        features = original_guess.get_features(projects)
        self.assertEqual(len(projects), 93)
        self.assertEqual(len(features), 22)
        pred = original_guess.predict(projects[-1], projects[:-1], features, "Effort-")
        self.assertTrue(math.isfinite(pred["estimate"]))
        self.assertLessEqual(pred["low"], pred["estimate"])
        self.assertLessEqual(pred["estimate"], pred["high"])


if __name__ == "__main__":
    unittest.main()
