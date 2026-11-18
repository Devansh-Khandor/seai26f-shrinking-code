"""Behavioral tests for Apache T7 Certify."""

import math
import sys
import unittest
from pathlib import Path

APACHE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APACHE_DIR))

from certify import (
    split_by_setting,
    cohens_d,
    cliffs_delta,
    ks_statistic,
    permutation_pvalue,
    compare_groups,
)
from remember import GOAL, load_configurations


class TestCertify(unittest.TestCase):

    def setUp(self):
        self.rows = [
            {"keepAlive": 0, "accessLog": 0, GOAL: 10},
            {"keepAlive": 0, "accessLog": 1, GOAL: 20},
            {"keepAlive": 1, "accessLog": 0, GOAL: 30},
            {"keepAlive": 1, "accessLog": 1, GOAL: 40},
        ]

    def test_correct_groups(self):
        disabled, enabled = split_by_setting(
            self.rows, "keepAlive"
        )
        self.assertEqual(disabled, [10, 20])
        self.assertEqual(enabled, [30, 40])

    def test_mean_difference(self):
        result = compare_groups(self.rows, "keepAlive")
        self.assertEqual(result["mean_difference"], 20)
        self.assertEqual(result["n0"], 2)
        self.assertEqual(result["n1"], 2)

    def test_cliffs_delta(self):
        self.assertEqual(
            cliffs_delta([1, 2], [3, 4]), 1.0
        )
        self.assertEqual(
            cliffs_delta([3, 4], [1, 2]), -1.0
        )
        self.assertEqual(
            cliffs_delta([1, 2], [1, 2]), 0.0
        )

    def test_ks(self):
        self.assertEqual(
            ks_statistic([1, 2], [3, 4]), 1.0
        )
        self.assertEqual(
            ks_statistic([1, 2], [1, 2]), 0.0
        )

    def test_cohens_d(self):
        self.assertGreater(
            cohens_d([1, 2, 3], [8, 9, 10]), 0
        )

    def test_permutation_reproducible(self):
        a = [1, 2, 3]
        b = [8, 9, 10]
        self.assertEqual(
            permutation_pvalue(a, b),
            permutation_pvalue(a, b),
        )

    def test_identical_groups_not_significant(self):
        self.assertEqual(
            permutation_pvalue([1, 2, 3], [1, 2, 3]),
            1.0,
        )

    def test_missing_group_rejected(self):
        with self.assertRaises(ValueError):
            split_by_setting(
                self.rows[:2], "keepAlive"
            )

    def test_real_dataset(self):
        rows = load_configurations()

        for setting in ("keepAlive", "accessLog"):
            result = compare_groups(rows, setting)

            self.assertEqual(
                result["n0"] + result["n1"], 192
            )
            self.assertTrue(
                math.isfinite(result["mean_difference"])
            )
            self.assertGreaterEqual(result["permutation_p"], 0)
            self.assertLessEqual(result["permutation_p"], 1)


if __name__ == "__main__":
    unittest.main()
