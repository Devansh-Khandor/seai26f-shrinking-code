import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from shrinking_code.anomalies import input_isolation, outcome_residuals
from shrinking_code.data import Row, Table
from shrinking_code.distance import XDistance
from shrinking_code.evaluation import evaluate, train_test_split
from shrinking_code.prediction import KNNRegressor

ROOT = Path(__file__).resolve().parents[1]


def table(text):
    return Table.from_csv(io.StringIO(text))


def split(text, n=2):
    t = table(text)
    return Table(t.columns, t.rows[:n]), Table(t.columns, t.rows[n:])


class SpotTests(unittest.TestCase):
    def test_independent_isolation_scores_and_self_exclusion(self):
        t = table('Size,Goal+\n0,10\n1,20\n10,30\n')
        results = input_isolation(t, k=2, top=2)
        self.assertEqual([r.row.number for r in results], [3, 1, 2])
        for result, score in zip(results, [.95, .55, .5]):
            self.assertAlmostEqual(result.score, score)
            self.assertNotIn(result.row.number, [n for n, _ in result.neighbors])
        self.assertEqual([r.flagged for r in results], [True, True, False])
        self.assertEqual(results[0].neighbors, ((2, .9), (1, 1)))

    def test_mixed_distance_numeric_codes_and_active_columns(self):
        t = table('Size,origin,SecretX,Goal+\n0,1,999,100\n5,1,0,0\n0,9,888,5\n10,1,77,3\n')
        results = {r.row.number: r for r in input_isolation(t, k=1)}
        self.assertAlmostEqual(results[1].score, math.sqrt(.125))
        self.assertAlmostEqual(results[3].score, math.sqrt(.5))
        self.assertEqual(results[3].neighbors[0][0], 1)
        self.assertEqual(t.rows[2].cells[1], '9')
        distance = XDistance.fit(t)
        self.assertEqual([c.name for c in distance.columns], ['Size', 'origin'])
        for result in results.values():
            n, value = result.neighbors[0]
            self.assertEqual(value, distance(result.row, t.rows[n-1]))

    def test_y_and_ignored_changes_do_not_change_isolation(self):
        t = table('Size,SecretX,Goal+\n0,1,100\n1,2,200\n10,3,300\n')
        altered = Table(t.columns, tuple(Row(r.number, (r.cells[0], None, -999*r.number)) for r in t.rows))
        evidence = lambda data: [(r.row.number, r.score, r.neighbors, r.flagged) for r in input_isolation(data)]
        self.assertEqual(evidence(t), evidence(altered))

    def test_duplicate_x_constant_columns_and_ties(self):
        t = table('Size,kind,Goal+\n1,a,1\n1,a,2\n1,a,3\n')
        results = input_isolation(t, k=1)
        self.assertEqual([r.row.number for r in results], [1, 2, 3])
        self.assertEqual([r.score for r in results], [0, 0, 0])
        self.assertEqual([r.neighbors[0][0] for r in results], [2, 1, 1])
        self.assertEqual(results, input_isolation(Table(t.columns, tuple(reversed(t.rows))), k=1))

    def test_missing_x_and_no_comparable_neighbors(self):
        t = table('Size,kind,Goal+\n0,a,1\n?,a,2\n?,?,3\n')
        results = input_isolation(t)
        self.assertEqual([r.score for r in results], [0, 0, None])
        self.assertFalse(results[-1].flagged)
        self.assertEqual(results[-1].neighbors, ())
        self.assertIn('unscored', results[-1].reason)

    def test_small_singleton_empty_and_no_x(self):
        t = table('Size,Goal+\n0,1\n10,2\n')
        self.assertEqual([r.score for r in input_isolation(t)], [1, 1])
        self.assertEqual([len(r.neighbors) for r in input_isolation(t)], [1, 1])
        self.assertIsNone(input_isolation(Table(t.columns, t.rows[:1]))[0].score)
        self.assertEqual(input_isolation(Table(t.columns, ())), ())
        self.assertTrue(all(r.score is None for r in input_isolation(table('Goal+\n1\n2\n'))))

    def test_invalid_input_configuration(self):
        t = table('Size,Goal+\n0,1\n')
        for value in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError): input_isolation(t, k=value)
            with self.assertRaises(ValueError): input_isolation(t, top=value)
        with self.assertRaises(ValueError): input_isolation(Table(t.columns, t.rows*2))

    def test_known_residuals_signs_dispersion_and_order(self):
        a, b = split('Size,Reward+\n0,10\n10,30\n1,25\n9,15\n5,20\n')
        results = outcome_residuals(a, b, 'Reward+', k=2, top=2)
        self.assertEqual([r.row.number for r in results], [3, 4, 5])
        self.assertEqual([r.signed for r in results], [5, -5, 0])
        self.assertEqual([r.score for r in results], [5, 5, 0])
        self.assertEqual([r.flagged for r in results], [True, True, False])
        for r in results:
            self.assertEqual((r.prediction.value, r.prediction.minimum, r.prediction.maximum, r.prediction.stddev), (20, 10, 30, 10))

    def test_missing_test_and_training_targets(self):
        a, b = split('Size,Reward+\n0,?\n10,30\n1,25\n9,?\n')
        results = outcome_residuals(a, b, 'Reward+')
        self.assertEqual(results[0].prediction.value, 30)
        self.assertEqual(results[0].score, 5)
        self.assertIsNone(results[1].prediction)
        self.assertIsNone(results[1].score)
        self.assertFalse(results[1].flagged)
        all_missing = Table(b.columns, (b.rows[1],))
        self.assertFalse(outcome_residuals(a, all_missing, 'Reward+')[0].flagged)
        self.assertEqual(outcome_residuals(a, Table(b.columns, ()), 'Reward+'), ())

    def test_invalid_residual_requests(self):
        a, b = split('Size,Reward+\n0,10\n10,30\n1,25\n')
        for target in ['Size', 'absent']:
            with self.assertRaises(ValueError): outcome_residuals(a, b, target)
        with self.assertRaisesRegex(ValueError, 'overlap'): outcome_residuals(a, a, 'Reward+')
        with self.assertRaisesRegex(ValueError, 'unique'): outcome_residuals(a, Table(b.columns, b.rows*2), 'Reward+')
        with self.assertRaisesRegex(ValueError, 'schemas'): outcome_residuals(a, Table((), ()), 'Reward+')
        a, b = split('Size,Reward+\n0,?\n10,?\n1,25\n')
        with self.assertRaisesRegex(ValueError, 'No observed'): outcome_residuals(a, b, 'Reward+')
        a, b = split('Size,Reward+\n0,10\n10,30\n?,25\n')
        with self.assertRaisesRegex(ValueError, 'No eligible'): outcome_residuals(a, b, 'Reward+')

    def test_adversarial_heldout_labels_change_residual_not_prediction(self):
        a, b = split('Size,Reward+\n0,10\n10,30\n1,25\n9,15\n')
        altered = Table(b.columns, tuple(Row(r.number, (r.cells[0], 999)) for r in b.rows))
        first = {r.row.number: r for r in outcome_residuals(a, b, 'Reward+')}
        second = {r.row.number: r for r in outcome_residuals(a, altered, 'Reward+')}
        for number in first:
            self.assertEqual(first[number].prediction, second[number].prediction)
            self.assertNotEqual(first[number].score, second[number].score)

    def test_prediction_boundary_masks_goals_and_ignored_fields(self):
        a, b = split('Size,SecretX,Reward+\n0,99,10\n10,88,30\n100,77,999\n')
        original = KNNRegressor.predict
        def guarded(model, query, target):
            self.assertEqual(model.training, a)
            self.assertEqual(model.distance.summaries[0].maximum, 10)
            self.assertEqual(query.cells, (100, None, None))
            self.assertNotIn(query.number, [r.number for r in model.training.rows])
            return original(model, query, target)
        with patch.object(KNNRegressor, 'predict', guarded):
            self.assertEqual(outcome_residuals(a, b, 'Reward+')[0].prediction.value, 20)

    def test_targets_rank_separately(self):
        a, b = split('Size,Reward+,Cost-\n0,0,100\n10,10,200\n1,50,150\n9,5,999\n')
        reward = outcome_residuals(a, b, 'Reward+', k=2, top=1)
        cost = outcome_residuals(a, b, 'Cost-', k=2, top=1)
        self.assertEqual((reward[0].row.number, reward[0].score), (3, 45))
        self.assertEqual((cost[0].row.number, cost[0].score), (4, 849))

    def test_top_five_deterministic_even_zero_scores(self):
        t = table('Size,Reward+\n' + '1,5\n'*8)
        results = input_isolation(t)
        self.assertEqual([r.row.number for r in results if r.flagged], [1,2,3,4,5])
        a, b = Table(t.columns, t.rows[:1]), Table(t.columns, t.rows[1:])
        results = outcome_residuals(a, b, 'Reward+')
        self.assertEqual([r.row.number for r in results if r.flagged], [2,3,4,5,6])
        self.assertEqual(results, outcome_residuals(a, Table(b.columns, tuple(reversed(b.rows))), 'Reward+'))

    def test_cars_agree_with_guess_evaluation(self):
        a, b = train_test_split(Table.load(ROOT/'data/raw/auto93.csv'))
        self.assertEqual((len(a.rows),len(b.rows)), (318,80))
        for target in ['Mpg+', 'Lbs-']:
            old = evaluate(KNNRegressor.fit(a), b, target)
            results = outcome_residuals(a, b, target)
            predictions = {r.number: p for r,p in old.predictions}
            for r in results:
                self.assertEqual(r.prediction, predictions[r.row.number])
                self.assertEqual(r.score, abs(r.actual - r.prediction.value))
            self.assertAlmostEqual(sum(r.score for r in results)/len(results), old.mae)

    def test_spot_cli(self):
        cmd = [sys.executable, '-m', 'shrinking_code.demo', '--domain', 'cars', '--skill', 'spot']
        out = subprocess.check_output(cmd, cwd=ROOT, text=True)
        self.assertEqual(out, subprocess.check_output(cmd, cwd=ROOT, text=True))
        for value in ['CAR-17', 'CAR-18', 'training=318; held-out=80', 'row=155 isolation=',
                      'Mpg+: largest', 'Lbs-: largest', 'signed_residual=', 'not confirmed errors']:
            self.assertIn(value, out)
        self.assertEqual(sum(line.startswith('row=') for line in out.splitlines()), 15)

    def test_all_eight_previous_outputs(self):
        fixture = json.loads((ROOT/'tests/fixtures/pre_spot_demo_hashes.json').read_text())
        for skill, digest in fixture['sha256'].items():
            out = subprocess.check_output([sys.executable, '-m', 'shrinking_code.demo', '--domain', 'cars', '--skill', skill], cwd=ROOT)
            self.assertEqual(hashlib.sha256(out).hexdigest(), digest, skill)
