import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

from shrinking_code.data import Row, Table
from shrinking_code.demo import justify
from shrinking_code.explanation import Explainer
from shrinking_code.ranking import rank_rows

ROOT = Path(__file__).resolve().parents[1]


def table(text):
    return Table.from_csv(io.StringIO(text))


class ExplanationTests(unittest.TestCase):
    def test_known_contributions_and_directions(self):
        t = table('Size,Cost-,Gain+\n0,10,60\n1,30,20\n2,15,40\n')
        result = Explainer(t).explain(3)
        cost, gain = result.goals
        self.assertEqual((cost.raw, cost.direction, cost.minimum, cost.maximum), (15, 'minimize', 10, 30))
        self.assertEqual((cost.normalized, cost.ideal, cost.squared), (0.25, 0, 1/16))
        self.assertEqual((gain.normalized, gain.ideal, gain.squared), (0.5, 1, 0.25))
        self.assertAlmostEqual(result.score, math.sqrt(5/32))
        self.assertAlmostEqual(sum(g.squared for g in result.goals)/2, result.score**2)

    def test_constants(self):
        result = Explainer(table('Cost-,Gain+\n7,5\n7,5\n')).explain(1)
        self.assertEqual(result.score, 0.5)
        self.assertEqual([g.normalized for g in result.goals], [0.5,0.5])
        self.assertEqual([g.squared for g in result.goals], [0.25,0.25])

    def test_missing_goals_and_partial_comparison(self):
        e = Explainer(table('Size,Cost-,Gain+\n0,0,?\n1,10,0\n2,20,10\n'))
        partial = e.explain(1)
        self.assertIsNone(partial.score)
        self.assertIsNone(partial.rank)
        self.assertEqual(partial.goals[0].squared, 0)
        self.assertIsNone(partial.goals[1].normalized)
        self.assertIsNone(partial.goals[1].squared)
        comparison = e.compare(1,2)
        self.assertEqual(comparison.unknown, ('Gain+',))
        self.assertIsNone(comparison.winner)
        # Known value in incomplete row sets Cost minimum for all rows.
        self.assertEqual(e.explain(2).goals[0].normalized, 0.5)

    def test_reference_bounds_not_pair_bounds(self):
        e = Explainer(table('Size,Cost-\n0,0\n1,20\n2,40\n3,100\n'))
        pair = e.compare(2,3)
        self.assertEqual((pair.a.goals[0].normalized,pair.b.goals[0].normalized),(0.2,0.4))
        self.assertEqual(pair.winner,2)

    def test_neighbors_mixed_types_exclude_self(self):
        e = Explainer(table('Size,origin,Goal+\n0,1,1\n5,1,2\n0,2,3\n10,1,4\n'))
        neighbors=e.neighbors(1)
        self.assertEqual([n.explanation.row.number for n in neighbors],[2,3,4])
        self.assertAlmostEqual(neighbors[0].distance,math.sqrt(0.125))
        self.assertAlmostEqual(neighbors[1].distance,math.sqrt(0.5))
        self.assertEqual(neighbors[1].distance,neighbors[2].distance)
        self.assertEqual(e.rows[1].cells[1],'1')

    def test_neighbor_order_ignores_goals_and_ignored_fields(self):
        t=table('Size,origin,Goal+,SecretX\n0,1,1,7\n5,1,2,8\n0,2,3,9\n10,1,4,10\n')
        altered=Table(t.columns,tuple(Row(r.number,(r.cells[0],r.cells[1],None,999)) for r in t.rows))
        def neighbors(t):
            return [(n.explanation.row.number,n.distance) for n in Explainer(t).neighbors(1)]
        self.assertEqual(neighbors(t),neighbors(altered))

    def test_neighbor_roundoff_ties(self):
        e=Explainer(table('Size,Goal+\n2,1\n4,2\n0,3\n12,4\n3,5\n'))
        self.assertEqual([n.explanation.row.number for n in e.neighbors(5,2)],[1,2])

    def test_pairwise_tradeoff_and_winner(self):
        e=Explainer(table('Size,Cost-,Gain+\n0,0,8\n1,4,10\n2,10,0\n'))
        pair=e.compare(1,2)
        self.assertEqual(pair.favors_a,('Cost-',))
        self.assertEqual(pair.favors_b,('Gain+',))
        self.assertEqual(pair.winner,1)
        self.assertAlmostEqual(pair.a.score,math.sqrt(0.02))
        self.assertAlmostEqual(pair.b.score,math.sqrt(0.08))
        self.assertEqual(e.compare(2,1).winner,1)

    def test_pair_selection_deterministic(self):
        t=table('Size,Cost-,Gain+\n0,0,8\n1,4,10\n2,10,0\n')
        pair=Explainer(t).tradeoff_pair()
        self.assertEqual((pair.a.row.number,pair.b.row.number),(1,2))
        self.assertEqual(pair,Explainer(Table(t.columns,tuple(reversed(t.rows)))).tradeoff_pair())

    def test_no_tradeoff_pair_and_equal_scores(self):
        for text in ['Size,Cost-\n0,0\n1,1\n','Cost-,Gain+\n0,0\n10,10\n','Cost-\n5\n','Cost-\n']:
            self.assertIsNone(Explainer(table(text)).tradeoff_pair())
        pair=Explainer(table('Cost-,Gain+\n0,0\n10,10\n')).compare(1,2)
        self.assertIsNone(pair.winner)  # Row-order tie-break is not a better score.
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            justify(table('Size,Cost-\n0,0\n1,1\n'))
        self.assertIn('No complete-goal pair',out.getvalue())

    def test_small_and_incomparable_neighbor_sets(self):
        self.assertEqual(Explainer(table('Size,Goal+\n1,2\n')).neighbors(1),())
        self.assertEqual(Explainer(table('Size,Goal+\n?,2\n?,3\n')).neighbors(1),())
        e=Explainer(table('Size,Goal+\n1,2\n2,?\n'))
        self.assertEqual(len(e.neighbors(1,5)),1)
        self.assertIsNone(e.neighbors(1)[0].explanation.score)

    def test_invalid_requests(self):
        e=Explainer(table('Size,Goal+\n1,2\n'))
        with self.assertRaises(ValueError): e.explain(99)
        with self.assertRaises(ValueError): e.compare(1,1)
        for k in [0,-1,True,1.5]:
            with self.assertRaises(ValueError): e.neighbors(1,k)
        with self.assertRaises(ValueError): Explainer(table('X\n1\n'))
        with self.assertRaises(ValueError): Explainer(Table(e.reference.columns,(e.rows[1],e.rows[1])))


class JustifyIntegrationTests(unittest.TestCase):
    def test_all_cars_reconstruct_remember_scores(self):
        t=Table.load(ROOT/'data/raw/auto93.csv')
        e=Explainer(t)
        for rank,(row,score) in enumerate(rank_rows(t),1):
            explanation=e.explain(row.number)
            self.assertEqual((explanation.rank,explanation.score),(rank,score))
            self.assertAlmostEqual(sum(g.squared for g in explanation.goals)/3,score**2)
        pair=e.tradeoff_pair()
        self.assertEqual((pair.a.row.number,pair.b.row.number),(395,393))
        self.assertEqual(pair.favors_a,('Acc+',))
        self.assertEqual(pair.favors_b,('Lbs-',))
        self.assertEqual(pair.equal,('Mpg+',))

    def test_justify_cli(self):
        cmd=[sys.executable,'-m','shrinking_code.demo','--domain','cars','--skill','justify']
        out=subprocess.check_output(cmd,cwd=ROOT,text=True)
        self.assertEqual(out,subprocess.check_output(cmd,cwd=ROOT,text=True))
        for text in ['CAR-11','CAR-12','row=395, rank=1','row=393, rank=3',
                     'squared=0.062500000','row=368, X-distance=','row=361, X-distance=',
                     'row=347, X-distance=','Population context only','not a cause',
                     'Observed outcomes were not predicted']:
            self.assertIn(text,out)

    def test_all_previous_outputs_unchanged(self):
        fixture=json.loads((ROOT/'tests/fixtures/pre_justify_demo_hashes.json').read_text())
        for skill,digest in fixture['sha256'].items():
            with self.subTest(skill=skill):
                out=subprocess.check_output([sys.executable,'-m','shrinking_code.demo',
                                             '--domain','cars','--skill',skill],cwd=ROOT)
                self.assertEqual(hashlib.sha256(out).hexdigest(),digest)
