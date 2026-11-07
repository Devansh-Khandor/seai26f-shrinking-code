import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

from shrinking_code.data import Row, Table
from shrinking_code.evaluation import train_test_split
from shrinking_code.prediction import KNNRegressor
from shrinking_code.selection import HistoricalOracle, select_candidates

ROOT = Path(__file__).resolve().parents[1]


def split(text, n=2):
    t = Table.from_csv(io.StringIO(text))
    return Table(t.columns, t.rows[:n]), Table(t.columns, t.rows[n:])


def run(training, hidden, budget=40, k=1):
    oracle = HistoricalOracle(hidden, budget)
    result = select_candidates(training, oracle.visible, oracle.inspect, budget, k)
    return result, oracle


class ChooseTests(unittest.TestCase):
    def setUp(self):
        self.training, self.hidden = split('Size,Cost-,Gain+\n0,0,10\n10,10,0\n1,2,8\n9,8,2\n')

    def test_fixed_partition(self):
        t = Table.load(ROOT / 'data/raw/auto93.csv')
        a, b = train_test_split(t)
        ids = list(range(1, 399))
        random.Random(42).shuffle(ids)
        self.assertEqual([r.number for r in a.rows], sorted(ids[:318]))
        self.assertEqual([r.number for r in b.rows], sorted(ids[318:]))
        self.assertFalse({r.number for r in a.rows} & {r.number for r in b.rows})
        self.assertEqual((a, b), train_test_split(t))

    def test_oracle_masks_goals_and_ignored(self):
        a, b = split('Size,SecretX,Reward+\n0,999,0\n10,888,10\n1,777,5\n')
        oracle = HistoricalOracle(b, 1)
        self.assertEqual(oracle.visible.rows[0].cells, (1.0, None, None))
        self.assertEqual(oracle.inspected, ())
        self.assertEqual(oracle.inspect(3), (5.0,))
        self.assertEqual(oracle.inspected, (3,))
        self.assertEqual(oracle.visible.rows[0].cells, (1.0, None, None))

    def test_oracle_rejects_duplicate_unknown_and_over_budget(self):
        oracle = HistoricalOracle(self.hidden, 1)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            oracle.inspect(99)
        oracle.inspect(3)
        for number, message in [(3, 'already'), (4, 'exhausted')]:
            with self.assertRaisesRegex(ValueError, message):
                oracle.inspect(number)
        self.assertEqual(oracle.inspected, (3,))

    def test_unmasked_input_rejected_before_inspection(self):
        with self.assertRaisesRegex(ValueError, 'only X'):
            select_candidates(self.training, self.hidden, lambda _: self.fail('leak'))

    def test_overlap_duplicate_and_schema_rejected(self):
        oracle = HistoricalOracle(self.hidden, 2)
        row = oracle.visible.rows[0]
        for visible, message in [
            (Table(self.training.columns, (Row(1, row.cells),)), 'overlap'),
            (Table(self.training.columns, (row, row)), 'unique'),
            (Table((), ()), 'schema'),
            (Table(self.training.columns, (Row(3, (1,)),)), 'width'),
        ]:
            with self.assertRaisesRegex(ValueError, message):
                select_candidates(self.training, visible, lambda _: self.fail('inspection'))
        with self.assertRaisesRegex(ValueError, 'unique'):
            HistoricalOracle(Table(self.hidden.columns, self.hidden.rows * 2), 2)

    def test_prediction_directions_and_independent_scores(self):
        result, oracle = run(self.training, self.hidden)
        first, second = result.inspections
        self.assertEqual(oracle.inspected, (3, 4))
        self.assertEqual(first.predicted_goals, (0, 10))
        self.assertEqual(first.predicted_score, 0)
        self.assertAlmostEqual(first.observed_score, .2)
        self.assertEqual(second.predicted_goals, (10, 0))
        self.assertEqual(second.predicted_score, 1)
        self.assertAlmostEqual(second.observed_score, .8)
        self.assertEqual(result.prediction_count, 6)  # (2 + 1) candidates * 2 goals

    def test_budget_zero_one_over_pool_and_empty(self):
        for budget, expected in [(0, ()), (1, (3,)), (99, (3, 4))]:
            result, oracle = run(self.training, self.hidden, budget)
            self.assertEqual(oracle.inspected, expected)
            self.assertEqual(len(result.inspections), len(expected))
            self.assertEqual(len(set(oracle.inspected)), len(expected))
        result, oracle = run(self.training, Table(self.hidden.columns, ()))
        self.assertEqual(result.inspections, ())
        self.assertEqual(result.prediction_count, 0)
        self.assertEqual(oracle.inspected, ())

    def test_invalid_budget(self):
        for budget in [-1, 1.5, True, '1']:
            with self.assertRaisesRegex(ValueError, 'Budget'):
                HistoricalOracle(self.hidden, budget)
            with self.assertRaisesRegex(ValueError, 'Budget'):
                select_candidates(self.training, self.hidden, None, budget)

    def test_ties_and_input_order_deterministic(self):
        a, b = split('Size,Reward+\n0,0\n10,10\n9,8\n9,8\n')
        forward, o1 = run(a, b)
        reverse, o2 = run(a, Table(b.columns, tuple(reversed(b.rows))))
        self.assertEqual(o1.inspected, (3, 4))
        self.assertEqual(o1.inspected, o2.inspected)
        self.assertEqual(forward, reverse)
        self.assertEqual(forward.inspections[-1].best_row.number, 3)

    def test_model_update_changes_next_decision(self):
        a, b = split('Size,Reward+\n0,0\n10,10\n9,-10\n8,10\n0,0\n')
        result, oracle = run(a, b)
        # Initially rows 3 and 4 predict 10; after row 3 reveals -10,
        # row 4 predicts -10 and row 5 (prediction 0) becomes preferable.
        self.assertEqual(oracle.inspected, (3, 5, 4))
        self.assertEqual([d.predicted_goals for d in result.inspections], [(10,), (0,), (-10,)])
        self.assertEqual([d.training_count for d in result.inspections], [2, 3, 4])
        self.assertEqual([d.best_score for d in result.inspections], [2, 1, 0])

    def test_fixed_bounds_and_out_of_range_predictions(self):
        a, b = split('Size,Reward+\n0,0\n10,10\n9,30\n8,10\n')
        result, _ = run(a, b)
        first, second = result.inspections
        self.assertEqual(first.observed_score, 2)  # abs(30/10 - 1), no clipping
        self.assertEqual(second.predicted_goals, (30,))
        self.assertEqual(second.predicted_score, 2)  # bounds never expand to 30
        self.assertEqual(second.observed_score, 0)

    def test_constant_goals_even_new_values(self):
        a, b = split('Size,Cost-,Gain+\n0,7,5\n10,7,5\n1,100,-100\n9,7,5\n')
        result, _ = run(a, b)
        self.assertEqual([d.predicted_score for d in result.inspections], [.5, .5])
        self.assertEqual([d.observed_score for d in result.inspections], [.5, .5])

    def test_missing_revealed_goals_cost_one_but_cannot_win(self):
        a, b = split('Size,Cost-,Gain+\n0,0,10\n10,10,0\n1,?,8\n9,8,2\n')
        result, oracle = run(a, b)
        self.assertEqual(oracle.inspected, (3, 4))
        self.assertIsNone(result.inspections[0].observed_score)
        self.assertIsNone(result.inspections[0].best_row)
        self.assertEqual(result.inspections[1].training_count, 3)
        self.assertEqual(result.inspections[1].best_row.number, 4)

    def test_missing_training_goals_and_unpredictable_x(self):
        a, b = split('Size,Cost-,Gain+\n0,?,10\n10,10,?\n1,5,5\n')
        result, _ = run(a, b)
        self.assertEqual(result.inspections[0].predicted_goals, (10, 10))
        a, b = split('Size,Reward+\n0,?\n10,?\n1,5\n')
        with self.assertRaisesRegex(ValueError, 'every goal'):
            run(a, b)
        a, b = split('Size,Reward+\n0,0\n10,10\n?,5\n')
        oracle = HistoricalOracle(b, 1)
        with self.assertRaisesRegex(ValueError, 'No eligible'):
            select_candidates(a, oracle.visible, oracle.inspect, 1)
        self.assertEqual(oracle.inspected, ())

    def test_adversarial_hidden_goals_do_not_change_first_decision(self):
        changed = Table(self.hidden.columns, (
            Row(3, (1, 9999, -9999)), Row(4, (9, -9999, 9999))))
        a, oa = run(self.training, self.hidden, 1)
        b, ob = run(self.training, changed, 1)
        self.assertEqual(oa.visible, ob.visible)
        self.assertEqual(oa.inspected, (3,))
        self.assertEqual(oa.inspected, ob.inspected)
        self.assertEqual(a.inspections[0].predicted_goals, b.inspections[0].predicted_goals)
        self.assertEqual(a.inspections[0].predicted_score, b.inspections[0].predicted_score)

    def test_future_outcomes_cannot_change_earlier_prefix(self):
        a, b = split('Size,Reward+\n0,0\n10,10\n9,-10\n8,10\n0,0\n')
        changed = Table(b.columns, tuple(Row(r.number, (r.cells[0], 999))
                                        if r.number == 4 else r for r in b.rows))
        original, _ = run(a, b)
        altered, _ = run(a, changed)
        self.assertEqual(original.inspections[:2], altered.inspections[:2])
        self.assertEqual(original.inspections[2].predicted_score, altered.inspections[2].predicted_score)

    def test_model_sees_only_previous_observations_and_queries_only_x(self):
        oracle = HistoricalOracle(self.hidden, 2)
        real_fit = KNNRegressor.fit
        training_ids = {r.number for r in self.training.rows}
        def guarded_fit(training, k):
            self.assertEqual({r.number for r in training.rows}, training_ids | set(oracle.inspected))
            model = real_fit(training, k)
            class Guard:
                def predict(inner, row, goal):
                    self.assertNotIn(row.number, oracle.inspected)
                    self.assertTrue(all(row.cells[c.index] is None for c in training.columns if c.role != 'x'))
                    return model.predict(row, goal)
            return Guard()
        with patch('shrinking_code.selection.KNNRegressor.fit', side_effect=guarded_fit):
            select_candidates(self.training, oracle.visible, oracle.inspect, 2, 1)
        self.assertEqual(oracle.inspected, (3, 4))

    def test_monotonic_checkpoints_are_exact_prefix_states(self):
        a, b = split('Size,Reward+\n0,0\n10,10\n9,-10\n8,10\n0,0\n')
        result, _ = run(a, b)
        self.assertIsNone(result.checkpoint(0))
        for cost, expected in [(1, 2), (2, 1), (3, 0)]:
            self.assertIs(result.checkpoint(cost), result.inspections[cost - 1])
            self.assertEqual(result.checkpoint(cost).best_score, expected)
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            result.checkpoint(4)

    def test_malformed_oracle_response_rejected(self):
        oracle = HistoricalOracle(self.hidden, 1)
        for response in [(1,), (float('nan'), 2)]:
            with self.assertRaises(ValueError):
                select_candidates(self.training, oracle.visible, lambda _: response, 1)

    def test_all_previous_demo_outputs_unchanged(self):
        fixture = json.loads((ROOT / 'tests/fixtures/pre_choose_demo_hashes.json').read_text())
        for skill, digest in fixture['sha256'].items():
            output = subprocess.check_output([sys.executable, '-m', 'shrinking_code.demo',
                                              '--domain', 'cars', '--skill', skill], cwd=ROOT)
            self.assertEqual(hashlib.sha256(output).hexdigest(), digest, skill)

    def test_choose_cli_reproducible_checkpoints_and_cost(self):
        command = [sys.executable, '-m', 'shrinking_code.demo', '--domain', 'cars', '--skill', 'choose']
        output = subprocess.check_output(command, cwd=ROOT, text=True)
        self.assertEqual(output, subprocess.check_output(command, cwd=ROOT, text=True))
        for text in ['CAR-13', 'CAR-14', 'Historical training: 318; candidate pool: 80',
                     'Actual inspection cost: 40', 'prediction computations: 7260',
                     'Offline inspection simulation', 'Final recommended inspected car:']:
            self.assertIn(text, output)
        decisions = [line.split() for line in output.splitlines() if line.startswith('inspection=')]
        self.assertEqual(len(decisions), 40)
        ids = [int(line[1].split('=')[1]) for line in decisions]
        self.assertEqual(len(set(ids)), 40)
        self.assertTrue(set(ids) <= {r.number for r in train_test_split(Table.load(ROOT / 'data/raw/auto93.csv'))[1].rows})
        # Independently recover prefix minima from printed actual decisions.
        scores = [float(line[-1].split('=')[1]) for line in decisions]
        for cost in (10, 20, 40):
            best = min(range(cost), key=lambda i: (scores[i], ids[i]))
            self.assertIn(f'budget={cost} best_row={ids[best]} best_d2h={scores[best]:.9f}', output)
