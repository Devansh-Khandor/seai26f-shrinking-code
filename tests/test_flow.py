import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from shrinking_code.data import Column, Row, Table
from shrinking_code.evaluation import train_test_split
from shrinking_code.ranking import rank_rows
from shrinking_code.stats import summarize
from shrinking_code.streaming import Inventory

ROOT = Path(__file__).resolve().parents[1]


def table(text):
    return Table.from_csv(io.StringIO(text))


class FlowTests(unittest.TestCase):
    def test_empty_inventory_and_single_arrival(self):
        inventory = Inventory(table('Size,kind,Cost-\n'))
        self.assertEqual(inventory.shortlist, ())
        self.assertEqual(inventory.summaries[0].count, 0)
        decision = inventory.accept(['2', 'a', '10'])
        self.assertEqual((decision.row.number, decision.previous_count, decision.rank), (1, 0, 1))
        self.assertEqual(decision.score, 0.5)  # Constant single goal uses midpoint.
        self.assertTrue(decision.top_five)
        self.assertEqual(inventory.accepted_count, 1)
        self.assertEqual(inventory.table.rows, (Row(1, (2, 'a', 10)),))

    def test_invalid_initial_states(self):
        t = table('Size,Cost-\n1,2\n')
        invalid = [None, Table((), ()), table('Size\n1\n'),
                   Table(t.columns, (t.rows[0], t.rows[0])),
                   Table(t.columns, (Row(0, (1, 2)),)),
                   Table(t.columns, (Row(1, ('bad', 2)),)),
                   Table((Column(1, 'Cost-'),), ()),
                   Table((Column(0, ''),), ())]
        for state in invalid:
            with self.subTest(state=state), self.assertRaises(ValueError):
                Inventory(state)

    def test_sequential_summary_updates(self):
        inventory = Inventory(table('Size,kind,Cost-\n1,a,10\n'))
        inventory.accept([3, 'b', 20])
        s = inventory.summaries[0]
        self.assertEqual((s.count, s.minimum, s.maximum, s.mean, s.median), (2, 1, 3, 2, 2))
        self.assertEqual(inventory.summaries[1].mode, 'a')
        inventory.accept([8, 'b', 30])
        s = inventory.summaries[0]
        self.assertEqual((s.count, s.minimum, s.maximum, s.mean, s.median), (3, 1, 8, 4, 3))
        self.assertEqual(inventory.summaries[1].frequencies, {'a': 1, 'b': 2})
        self.assertEqual(inventory.summaries[1].mode, 'b')
        self.assertEqual((inventory.accepted_count, len(inventory.table.rows)), (2, 3))

    def test_missing_values_and_categorical_codes(self):
        inventory = Inventory(table('Size,origin,HpX,Cost-,Reward+\n'))
        arrival = inventory.accept(['?', '1', '?', 10, None])
        self.assertIsNone(arrival.rank)
        self.assertIsNone(arrival.score)
        self.assertFalse(arrival.top_five)
        self.assertEqual(inventory.shortlist, ())
        self.assertEqual(inventory.summaries[2].count, 0)
        self.assertEqual(inventory.table.rows[0].cells[1], '1')
        complete = inventory.accept([2, None, 20, 20, 5])
        # Cost range includes the partial row's known 10; Reward is constant.
        self.assertAlmostEqual(complete.score, math.sqrt((1 + 0.25) / 2))
        self.assertEqual(inventory.summaries[1].frequencies, {'1': 1})
        self.assertEqual(inventory.accepted_count, 2)

    def test_top_five_boundary_and_ties(self):
        inventory = Inventory(table('Cost-\n10\n20\n30\n40\n50\n'))
        decision = inventory.accept([50])
        self.assertEqual(decision.rank, 6)
        self.assertFalse(decision.top_five)
        winner = inventory.accept([0])
        self.assertEqual((winner.rank, winner.score, winner.top_five), (1, 0, True))
        self.assertEqual([r.number for r, _ in inventory.shortlist], [7, 1, 2, 3, 4])

    def test_original_id_ties_not_arrival_order(self):
        inventory = Inventory(Table((Column(0, 'Cost-'),), (Row(10, (5,)),)))
        arrival = inventory.accept([5], number=2)
        self.assertEqual(arrival.rank, 1)
        self.assertEqual([r.number for r, _ in inventory.shortlist], [2, 10])
        self.assertEqual(inventory.accept([5]).row.number, 11)

    def test_common_expanded_bounds_rerank_existing_rows(self):
        inventory = Inventory(table('Cost-,Time-\n0,10\n10,0\n5,5\n'))
        self.assertEqual([r.number for r, _ in inventory.shortlist], [3, 1, 2])
        decision = inventory.accept([100, 0])
        self.assertEqual(decision.previous_count, 3)
        self.assertEqual(decision.rank, 4)
        self.assertAlmostEqual(decision.score, math.sqrt(0.5))
        self.assertEqual([r.number for r, _ in inventory.shortlist], [2, 3, 1, 4])
        self.assertAlmostEqual(inventory.shortlist[0][1], math.sqrt(0.01 / 2))
        self.assertAlmostEqual(inventory.shortlist[1][1], math.sqrt((0.0025 + 0.25) / 2))
        self.assertEqual(inventory.summaries[0].maximum, 100)

    def test_validation_failure_does_not_mutate_state(self):
        inventory = Inventory(table('Size,kind,Cost-\n1,a,10\n'))
        before = (inventory.table, inventory.summaries, inventory.shortlist, inventory.accepted_count)
        invalid = [([2, 'b'], 2), (['bad', 'b', 10], 2), ([2, '', 10], 2),
                   ([2, 'b', float('nan')], 2), ([2, 'b', float('inf')], 2),
                   ([2, 'b', 5], 1), ([2, 'b', 5], 0), ([2, 'b', 5], True),
                   ([2, 'b', 5], 2.5), ('2,b,5', 2), (None, 2), ([{}, 'b', 5], 2)]
        for cells, number in invalid:
            with self.subTest(cells=cells), self.assertRaises(ValueError):
                inventory.accept(cells, number)
            self.assertEqual((inventory.table, inventory.summaries, inventory.shortlist,
                              inventory.accepted_count), before)
        self.assertEqual(inventory.accept([2, 'b', 5]).row.number, 2)

    def test_assessment_is_staged_before_acceptance(self):
        inventory = Inventory(table('Cost-\n10\n20\n'))
        original = inventory.table
        def check(candidate):
            self.assertIs(inventory.table, original)
            self.assertEqual(inventory.accepted_count, 0)
            self.assertEqual(len(candidate.rows), 3)
            return rank_rows(candidate)
        with patch('shrinking_code.streaming.rank_rows', side_effect=check):
            decision = inventory.accept([0])
        self.assertEqual(decision.previous_count, 2)
        self.assertEqual(len(inventory.table.rows), 3)

    def test_future_arrivals_cannot_change_saved_decision(self):
        inventory = Inventory(table('Cost-\n0\n10\n'))
        arrival = inventory.accept([5])
        self.assertEqual(arrival.score, 0.5)
        inventory.accept([100])
        self.assertEqual(arrival.score, 0.5)  # Arrival-time snapshot, not retroactive.
        scores = {r.number: s for r, s in inventory.shortlist}
        self.assertEqual(scores[arrival.row.number], 0.05)
        self.assertEqual(arrival.previous_count, 2)


class FlowIntegrationTests(unittest.TestCase):
    def test_final_state_matches_independent_batch(self):
        full = Table.load(ROOT / 'data/raw/auto93.csv')
        initial, incoming = train_test_split(full)
        inventory = Inventory(initial)
        self.assertEqual(len(inventory.table.rows), 318)
        for index, row in enumerate(incoming.rows):
            result = inventory.accept(row.cells, row.number)
            self.assertEqual(result.previous_count, 318 + index)
        self.assertEqual(inventory.accepted_count, 80)
        self.assertEqual(inventory.table, full)
        self.assertEqual(inventory.summaries, tuple(summarize(full, c) for c in full.columns))
        self.assertEqual(inventory.shortlist, tuple(rank_rows(full)[:5]))
        self.assertEqual([r.number for r, _ in inventory.shortlist], [395, 394, 393, 396, 377])

    def check_previous_demo(self, skill):
        fixture = json.loads((ROOT / 'tests/fixtures/pre_flow_demo_hashes.json').read_text())
        output = subprocess.check_output([sys.executable, '-m', 'shrinking_code.demo',
                                          '--domain', 'cars', '--skill', skill], cwd=ROOT)
        self.assertEqual(hashlib.sha256(output).hexdigest(), fixture['sha256'][skill])

    def test_remember_output_unchanged(self):
        self.check_previous_demo('remember')

    def test_guess_output_unchanged(self):
        self.check_previous_demo('guess')

    def test_certify_output_unchanged(self):
        self.check_previous_demo('certify')

    def test_flow_cli(self):
        cmd = [sys.executable, '-m', 'shrinking_code.demo', '--domain', 'cars', '--skill', 'flow']
        output = subprocess.check_output(cmd, cwd=ROOT, text=True)
        for text in ['CAR-07', 'CAR-08', 'Initial inventory: 318', 'incoming records: 80',
                     'accepted arrivals: 80; final inventory: 398', 'Full-batch Remember verification: PASS',
                     'mean=2970.424623', 'median=2803.5', '395 0.167460', 'Synthetic arrival order']:
            self.assertIn(text, output)
        self.assertEqual(output, subprocess.check_output(cmd, cwd=ROOT, text=True))
