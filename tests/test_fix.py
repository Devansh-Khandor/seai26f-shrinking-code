import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

from shrinking_code.alternatives import Alternatives, target_status
from shrinking_code.data import Row, Table
from shrinking_code.demo import fix
from shrinking_code.distance import XDistance
from shrinking_code.explanation import Explainer
from shrinking_code.ranking import rank_rows

ROOT = Path(__file__).resolve().parents[1]


def table(text):
    return Table.from_csv(io.StringIO(text))


class FixTests(unittest.TestCase):
    def test_worst_complete_source_and_ties(self):
        t = table('Size,Reward+\n0,0\n1,0\n2,10\n3,?\n')
        self.assertEqual(Alternatives(t).worst().number, 1)
        self.assertEqual(Alternatives(Table(t.columns, tuple(reversed(t.rows)))).worst().number, 1)
        self.assertIsNone(Alternatives(table('Size,Reward+\n1,?\n')).worst())

    def test_better_filters_self_equal_and_incomplete(self):
        t = table('Size,Reward+\n0,0\n1,0\n2,10\n3,?\n')
        results = Alternatives(t).better(1)
        self.assertEqual([r.comparison.b.row.number for r in results], [3])
        self.assertEqual(results[0].improvement, 1)
        self.assertEqual(results[0].comparison.a.score, 1)
        self.assertEqual(results[0].comparison.b.score, 0)

    def test_better_ranking_distance_then_score_then_id(self):
        t = table('Size,Reward+\n0,0\n1,5\n1,8\n1,8\n2,10\n')
        search = Alternatives(t)
        self.assertEqual([r.comparison.b.row.number for r in search.better(1)], [3, 4, 2, 5])
        self.assertEqual(len(search.better(1, 2)), 2)
        self.assertEqual(search.better(1), Alternatives(Table(t.columns, tuple(reversed(t.rows)))).better(1))

    def test_mixed_distance_and_changes_ignore_ignored_fields(self):
        t = table('Size,origin,HiddenX,Reward+\n0,1,100,0\n5,1,200,5\n0,9,999,7\n10,1,0,10\n')
        results = Alternatives(t).better(1)
        self.assertEqual([r.comparison.b.row.number for r in results], [2, 4, 3])
        self.assertAlmostEqual(results[0].distance, math.sqrt(.125))
        self.assertAlmostEqual(results[2].distance, math.sqrt(.5))
        self.assertEqual([(c.name, c.before, c.after) for c in results[0].changed], [('Size', 0, 5)])
        self.assertEqual([(c.name, c.before, c.after) for c in results[2].changed], [('origin', '1', '9')])
        distance = XDistance.fit(t)
        for result in results:
            self.assertEqual(result.distance, distance(t.rows[0], result.comparison.b.row))
            self.assertEqual(len(result.changed), 1)

    def test_exact_numeric_change_no_tolerance(self):
        t = table('Size,Reward+\n1,0\n1.000000000001,1\n')
        self.assertEqual(len(Alternatives(t).better(1)[0].changed), 1)

    def test_target_directions_inclusive_missing_and_subsets(self):
        t = table('Size,Cost-,Gain+\n0,10,20\n1,11,19\n2,?,21\n')
        thresholds = {'Gain+': 20, 'Cost-': 10}
        status = target_status(t.rows[0], t.columns, thresholds)
        self.assertEqual(status.satisfied, ('Cost-', 'Gain+'))
        self.assertEqual(status.count, 2)
        self.assertEqual(target_status(t.rows[1], t.columns, thresholds).failed, ('Cost-', 'Gain+'))
        partial = target_status(t.rows[2], t.columns, thresholds)
        self.assertEqual((partial.satisfied, partial.failed, partial.unknown), (('Gain+',), (), ('Cost-',)))
        self.assertEqual(target_status(t.rows[0], t.columns, {'Gain+': 20}).count, 1)

    def test_invalid_targets(self):
        t = table('Size,Reward+\n1,1\n')
        for thresholds in [{}, {'Size': 1}, {'Reward+': None}, {'Reward+': float('nan')}, {'Reward+': True}]:
            with self.assertRaises(ValueError):
                target_status(t.rows[0], t.columns, thresholds)
            with self.assertRaises(ValueError):
                Alternatives(t).meet_more(1, thresholds)

    def test_target_eligibility_is_strict(self):
        t = table('Size,Cost-,Gain+\n0,10,0\n0,9,0\n1,8,10\n')
        result = Alternatives(t).meet_more(1, {'Cost-': 8, 'Gain+': 10})
        self.assertEqual(result.comparison.b.row.number, 3)
        self.assertEqual((result.source_targets.count, result.alternative_targets.count), (0, 2))
        self.assertEqual(result.ranking_key, (1, 1, -2, 0, 3))

    def test_target_ranking_changes_before_distance(self):
        t = table('Size,kind,Reward+\n0,a,0\n10,a,5\n1,b,10\n')
        result = Alternatives(t).meet_more(1, {'Reward+': 5})
        self.assertEqual(result.comparison.b.row.number, 2)  # 1 change beats nearer 2 changes
        self.assertEqual(len(result.changed), 1)
        self.assertAlmostEqual(result.distance, math.sqrt(.5))

    def test_target_ranking_distance_before_count(self):
        t = table('Size,Cost-,Gain+\n0,10,0\n1,9,5\n2,0,10\n')
        result = Alternatives(t).meet_more(1, {'Cost-': 0, 'Gain+': 5})
        self.assertEqual(result.comparison.b.row.number, 2)
        self.assertEqual(result.alternative_targets.count, 1)

    def test_target_ranking_count_before_score(self):
        t = table('Size,Cost-,Gain+\n0,10,0\n1,0,4\n1,5,5\n2,10,10\n')
        result = Alternatives(t).meet_more(1, {'Cost-': 5, 'Gain+': 5})
        self.assertEqual(result.comparison.b.row.number, 3)
        self.assertEqual(result.alternative_targets.count, 2)
        self.assertLess(Explainer(t).explain(2).score, result.comparison.b.score)

    def test_target_ranking_score_then_identity(self):
        t = table('Size,Reward+\n0,0\n1,5\n1,10\n1,10\n')
        result = Alternatives(t).meet_more(1, {'Reward+': 5})
        self.assertEqual(result.comparison.b.row.number, 3)
        self.assertEqual(result, Alternatives(Table(t.columns, tuple(reversed(t.rows)))).meet_more(1, {'Reward+': 5}))

    def test_more_targets_need_not_improve_d2h(self):
        t = table('Size,Cost-,Gain+\n0,0,4\n1,10,5\n2,10,10\n')
        result = Alternatives(t).meet_more(1, {'Gain+': 5})
        self.assertEqual(result.comparison.b.row.number, 2)
        self.assertLess(result.improvement, 0)
        self.assertFalse(result.pareto_dominates)

    def test_overall_improvement_is_not_pareto_dominance(self):
        t = table('Size,Cost-,Gain+\n0,4,10\n1,0,8\n2,10,0\n')
        result = Alternatives(t).better(1)[0]
        self.assertAlmostEqual(result.improvement, math.sqrt(.08)-math.sqrt(.02))
        self.assertEqual(result.comparison.favors_b, ('Cost-',))
        self.assertEqual(result.comparison.favors_a, ('Gain+',))
        self.assertFalse(result.pareto_dominates)
        self.assertTrue(Alternatives(t).better(3)[0].pareto_dominates)

    def test_missing_x_unknown_not_false_unchanged(self):
        t = table('Size,kind,Reward+\n0,a,0\n?,a,10\n1,a,5\n?,?,9\n')
        results = Alternatives(t).better(1)
        self.assertEqual([r.comparison.b.row.number for r in results], [2, 3])
        self.assertEqual(results[0].unknown_x, ('Size',))
        self.assertEqual(results[0].changed, ())  # known changes only; unknown disclosed
        self.assertEqual(Alternatives(t).meet_more(1, {'Reward+': 5}).comparison.b.row.number, 3)
        self.assertIsNone(Alternatives(t).meet_more(2, {'Reward+': 11}))

    def test_incomplete_goals_source_and_alternative(self):
        t = table('Size,Cost-,Gain+\n0,10,0\n1,0,?\n2,5,10\n')
        search = Alternatives(t)
        with self.assertRaisesRegex(ValueError, 'complete'):
            search.better(2)
        self.assertEqual([r.comparison.b.row.number for r in search.better(1)], [3])
        self.assertEqual(search.meet_more(1, {'Cost-': 0}), None)
        # Incomplete row still defines common Cost minimum.
        self.assertEqual(search.better(1)[0].comparison.b.goals[0].normalized, .5)

    def test_constant_goals_and_no_solution(self):
        t = table('Size,Reward+\n1,5\n1,5\n')
        search = Alternatives(t)
        self.assertEqual(search.better(1), ())
        self.assertIsNone(search.meet_more(1, {'Reward+': 5}))
        self.assertEqual(search.explainer.explain(1).score, .5)
        single = Alternatives(table('Size,Reward+\n0,1\n'))
        self.assertEqual(single.better(1), ())
        self.assertIsNone(single.meet_more(1, {'Reward+': 2}))

    def test_invalid_source_and_limit(self):
        search = Alternatives(table('Size,Reward+\n0,0\n1,1\n'))
        with self.assertRaises(ValueError):
            search.better(999)
        for limit in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                search.better(1, limit)

    def test_cars_matches_independent_rules_and_existing_explanations(self):
        t = Table.load(ROOT / 'data/raw/auto93.csv')
        search = Alternatives(t)
        source = search.worst()
        self.assertEqual(source.number, 15)
        e = Explainer(t)
        scores = {r.number: score for r, score in rank_rows(t)}
        # Independent Cars X metric: ranges 3..8, 68..455, 70..82 and symbolic origin.
        def distance(row):
            a, b = source.cells, row.cells
            return math.sqrt(((a[0]-b[0])**2/25 + (a[1]-b[1])**2/387**2
                              + (a[3]-b[3])**2/144 + (a[4]!=b[4]))/4)
        expected = sorted((r for r in t.rows if scores[r.number] < scores[source.number]),
                          key=lambda r: (distance(r), scores[r.number], r.number))[:5]
        results = search.better(source.number)
        self.assertEqual([r.comparison.b.row.number for r in results], [r.number for r in expected])
        for result in results:
            pair = result.comparison
            self.assertEqual(pair, e.compare(source.number, pair.b.row.number))
            self.assertIn(pair.b.row, t.rows)
            self.assertAlmostEqual(result.distance, distance(pair.b.row))
        thresholds = {'Lbs-': 2803.5, 'Acc+': 15.5, 'Mpg+': 20}
        def count(row):
            return sum((row.cells[5] <= 2803.5, row.cells[6] >= 15.5, row.cells[7] >= 20))
        eligible = [r for r in t.rows if count(r) > count(source)]
        expected = min(eligible, key=lambda r: (sum(r.cells[i]!=source.cells[i] for i in (0,1,3,4)),
                                               distance(r), -count(r), scores[r.number], r.number))
        result = search.meet_more(source.number, thresholds)
        self.assertEqual(result.comparison.b.row, expected)
        self.assertEqual(expected.number, 58)

    def test_cli_and_explicit_no_solution(self):
        output = subprocess.check_output([sys.executable, '-m', 'shrinking_code.demo', '--domain', 'cars', '--skill', 'fix'], cwd=ROOT, text=True)
        for text in ['CAR-15', 'CAR-16', 'row=15', 'row=40', 'row=14', 'row=8', 'row=58', 'row=18',
                     'target count=0 -> 1', 'Pareto_dominates=False', 'historical medians', 'causal improvements']:
            self.assertIn(text, output)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            fix(table('Size,Reward+\n1,5\n'))
        self.assertIn('No qualifying better alternative', stream.getvalue())
        self.assertIn('No qualifying alternative with more', stream.getvalue())

    def test_all_seven_previous_outputs_and_oracle_module_unchanged(self):
        fixture = json.loads((ROOT / 'tests/fixtures/pre_fix_demo_hashes.json').read_text())
        for skill, digest in fixture['sha256'].items():
            output = subprocess.check_output([sys.executable, '-m', 'shrinking_code.demo', '--domain', 'cars', '--skill', skill], cwd=ROOT)
            self.assertEqual(hashlib.sha256(output).hexdigest(), digest, skill)
        original = subprocess.check_output(['git', 'show', fixture['source_commit'] + ':shrinking_code/selection.py'], cwd=ROOT)
        self.assertEqual(original, (ROOT / 'shrinking_code/selection.py').read_bytes())
