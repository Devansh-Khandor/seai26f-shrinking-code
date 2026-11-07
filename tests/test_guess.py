import io
import math
from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys
import unittest

from shrinking_code.data import Row, Table
from shrinking_code.distance import XDistance
from shrinking_code.evaluation import evaluate, mae, train_test_split
from shrinking_code.prediction import KNNRegressor

ROOT = Path(__file__).resolve().parents[1]


def table(text):
    return Table.from_csv(io.StringIO(text))


class DistanceTests(unittest.TestCase):
    def test_numeric_distance_and_training_scope(self):
        t = table('Size,Y+\n10,0\n30,100\n')
        distance = XDistance.fit(t)
        self.assertEqual(distance(t.rows[0], Row(3, (15, None))), 0.25)
        self.assertEqual(distance(t.rows[0], Row(3, (50, None))), 2)
        self.assertEqual(distance.summaries[0].maximum, 30)

    def test_categorical_distance(self):
        t = table('origin,Y+\n1,10\n2,20\n')
        distance = XDistance.fit(t)
        self.assertEqual(distance(t.rows[0], t.rows[0]), 0)
        self.assertEqual(distance(t.rows[0], t.rows[1]), 1)

    def test_mixed_distance_and_p(self):
        t = table('Size,kind,Y+\n0,a,0\n10,b,1\n')
        query = Row(3, (5, 'b', None))
        self.assertAlmostEqual(XDistance.fit(t)(t.rows[0], query), math.sqrt(0.625))
        self.assertEqual(XDistance.fit(t, p=1)(t.rows[0], query), 0.75)
        for p in [0, -1, 0.5, True, '2', float('inf'), float('nan')]:
            with self.subTest(p=p), self.assertRaises(ValueError):
                XDistance.fit(t, p=p)

    def test_goals_ignored_columns_and_identity_not_in_distance(self):
        t = table('Size,kind,Y+,Cost-,SecretX\n0,a,10,20,1\n10,b,30,40,2\n')
        query = Row(300, (0, 'a', -99999, 99999, 99999))
        self.assertEqual(XDistance.fit(t)(t.rows[0], query), 0)

    def test_missing_inputs(self):
        t = table('Size,kind,Y+\n0,a,1\n10,b,2\n')
        distance = XDistance.fit(t)
        self.assertEqual(distance(t.rows[0], Row(3, (None, 'b', None))), 1)
        self.assertEqual(distance(t.rows[0], Row(3, (5, None, None))), 0.5)
        self.assertTrue(math.isinf(distance(t.rows[0], Row(3, (None, None, 1)))))
        empty = table('Size,kind,Y+\n?,a,1\n?,b,2\n')
        self.assertEqual(XDistance.fit(empty)(Row(3, (0, 'a', 0)), Row(4, (10, 'a', 0))), 0)

    def test_constant_numeric_inputs(self):
        t = table('Size,kind,Y+\n7,a,1\n7,b,2\n')
        self.assertAlmostEqual(XDistance.fit(t)(t.rows[0], t.rows[1]), math.sqrt(0.5))
        # A constant training feature has no learned scale, even for a novel value.
        self.assertEqual(XDistance.fit(t)(t.rows[0], Row(3, (99, 'a', 0))), 0)

    def test_no_independent_inputs(self):
        t = table('Y+,IgnoredX\n1,2\n')
        self.assertTrue(math.isinf(XDistance.fit(t)(t.rows[0], Row(2, (3, 4)))))


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.training = table('Size,Y+,Cost-\n0,10,100\n10,30,200\n20,90,400\n')
        self.query = Row(4, (5, None, None))

    def test_known_predictions_for_two_targets_and_uncertainty(self):
        model = KNNRegressor.fit(self.training, k=2)
        prediction = model.predict(self.query, 'Y+')
        self.assertEqual([r.number for r in prediction.neighbors], [1, 2])
        self.assertEqual((prediction.value, prediction.minimum, prediction.maximum,
                          prediction.stddev), (20, 10, 30, 10))
        self.assertEqual(model.predict(self.query, 'Cost-').value, 150)

    def test_deterministic_ties_use_original_identity(self):
        reversed_rows = Table(self.training.columns, tuple(reversed(self.training.rows)))
        prediction = KNNRegressor.fit(reversed_rows, k=1).predict(self.query, 'Y+')
        self.assertEqual(prediction.value, 10)
        self.assertEqual(prediction.stddev, 0)

    def test_roundoff_ties_follow_original_identity(self):
        t = table('Size,Y+\n2,10\n4,30\n0,90\n12,90\n')
        # Both first neighbors are 1/12 away; subtraction after normalization
        # can represent those equal distances with different final bits.
        prediction = KNNRegressor.fit(t, k=1).predict(Row(5, (3, None)), 'Y+')
        self.assertEqual(prediction.neighbors[0].number, 1)
        self.assertEqual(prediction.value, 10)

    def test_fewer_than_k_and_missing_target(self):
        t = table('Size,Y+\n0,?\n10,30\n20,90\n')
        prediction = KNNRegressor.fit(t).predict(Row(4, (0, None)), 'Y+')
        self.assertEqual([r.number for r in prediction.neighbors], [2, 3])
        self.assertEqual((prediction.value, prediction.stddev), (60, 30))

    def test_invalid_k_and_targets(self):
        for k in [0, -1, 1.5, True, None, '5']:
            with self.subTest(k=k), self.assertRaises(ValueError):
                KNNRegressor.fit(self.training, k=k)
        model = KNNRegressor.fit(self.training)
        for target in ['Size', 'absent']:
            with self.assertRaisesRegex(ValueError, 'Unavailable'):
                model.predict(self.query, target)
        with self.assertRaises(ValueError):
            KNNRegressor.fit(Table(self.training.columns, ()))

    def test_unavailable_neighbors(self):
        t = table('Size,Y+\n0,?\n1,?\n')
        with self.assertRaisesRegex(ValueError, 'No eligible'):
            KNNRegressor.fit(t).predict(Row(3, (0, None)), 'Y+')
        with self.assertRaisesRegex(ValueError, 'No eligible'):
            KNNRegressor.fit(self.training).predict(Row(4, (None, 0, 0)), 'Y+')

    def test_reject_training_identity_even_with_masked_target(self):
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            KNNRegressor.fit(self.training).predict(Row(1, (0, None, None)), 'Y+')

    def test_query_outcomes_cannot_change_prediction(self):
        model = KNNRegressor.fit(self.training)
        self.assertEqual(model.predict(self.query, 'Y+'),
                         model.predict(Row(4, (5, -999, 999)), 'Y+'))

    def test_model_and_training_statistics_stay_fixed(self):
        model = KNNRegressor.fit(self.training)
        before = model.distance.summaries
        model.predict(Row(4, (1000, None, None)), 'Y+')
        self.assertEqual(model.distance.summaries, before)
        with self.assertRaises(FrozenInstanceError):
            model.k = 10


class EvaluationTests(unittest.TestCase):
    def test_split_deterministic_disjoint_complete_and_goal_independent(self):
        t = Table.load(ROOT / 'data/raw/auto93.csv')
        training, test = train_test_split(t)
        self.assertEqual((len(training.rows), len(test.rows)), (318, 80))
        self.assertFalse({r.number for r in training.rows} & {r.number for r in test.rows})
        self.assertEqual(set(training.rows) | set(test.rows), set(t.rows))
        self.assertEqual((training, test), train_test_split(t))
        self.assertNotEqual((training, test), train_test_split(t, seed=43))
        altered = Table(t.columns, tuple(Row(r.number, tuple(None if c.role == 'y' else
                         r.cells[c.index] for c in t.columns)) for r in t.rows))
        self.assertEqual([r.number for r in train_test_split(altered)[1].rows],
                         [r.number for r in test.rows])

    def test_known_mae_and_training_mean_baseline(self):
        full = table('Size,Y+\n0,10\n10,30\n1,14\n9,26\n5,?\n')
        model = KNNRegressor.fit(Table(full.columns, full.rows[:2]), k=1)
        result = evaluate(model, Table(full.columns, full.rows[2:]), 'Y+')
        self.assertEqual((result.training_count, result.test_count), (2, 2))
        self.assertEqual((result.mae, result.baseline, result.baseline_mae), (4, 20, 6))
        self.assertEqual(mae([1, 4, 9], [3, 4, 6]), 5 / 3)
        for actual, predicted in [([], []), ([1], []), ([1], [1, 2])]:
            with self.assertRaises(ValueError):
                mae(actual, predicted)

    def test_evaluation_leakage_guards_and_no_observed_targets(self):
        t = table('Size,Y+\n0,10\n10,30\n')
        model = KNNRegressor.fit(t)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            evaluate(model, t, 'Y+')
        with self.assertRaisesRegex(ValueError, 'schemas'):
            evaluate(model, table('Other,Y+\n1,2\n'), 'Y+')
        with self.assertRaises(ValueError):
            evaluate(model, Table(t.columns, (Row(3, (0, None)),)), 'Y+')

    def test_extreme_heldout_values_cannot_change_training_scale_or_baseline(self):
        t = table('Size,Y+\n0,10\n10,30\n')
        model = KNNRegressor.fit(t, k=1)
        test = Table(t.columns, (Row(3, (1000, 100000)),))
        result = evaluate(model, test, 'Y+')
        self.assertEqual(model.distance.summaries[0].maximum, 10)
        self.assertEqual(result.baseline, 20)
        self.assertEqual(result.predictions[0][1].value, 30)
        self.assertEqual(result.mae, 99970)

    def test_missing_training_targets_do_not_enter_baseline(self):
        t = table('Size,Y+\n0,?\n10,30\n')
        model = KNNRegressor.fit(t)
        result = evaluate(model, Table(t.columns, (Row(3, (0, 40)),)), 'Y+')
        self.assertEqual((result.baseline, result.baseline_mae, result.mae), (30, 10, 10))

    def test_guess_cli_output_is_reproducible_and_matches_evaluation(self):
        command = [sys.executable, '-m', 'shrinking_code.demo', '--domain', 'cars', '--skill', 'guess']
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, subprocess.check_output(command, cwd=ROOT, text=True))
        for text in ['CAR-03', 'CAR-04', 'train=318; test=80', 'not calibrated prediction intervals']:
            self.assertIn(text, result.stdout)
        training, test = train_test_split(Table.load(ROOT / 'data/raw/auto93.csv'))
        model = KNNRegressor.fit(training)
        for target in ['Mpg+', 'Lbs-']:
            evaluation = evaluate(model, test, target)
            self.assertIn(f'MAE={evaluation.mae:.6f}', result.stdout)
            self.assertIn(f'baseline_MAE={evaluation.baseline_mae:.6f}', result.stdout)
            self.assertEqual(evaluation.test_count, 80)
            train_ids = {r.number for r in training.rows}
            for row, prediction in evaluation.predictions:
                self.assertNotIn(row.number, train_ids)
                self.assertTrue({r.number for r in prediction.neighbors} <= train_ids)
