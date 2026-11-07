import contextlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

from shrinking_code.certification import compare, same
from shrinking_code.data import Table
from shrinking_code.demo import cylinder_groups, model_year_groups, print_comparison

ROOT = Path(__file__).resolve().parents[1]


class CertificationTests(unittest.TestCase):
    def test_cohen_known_pooled_variance(self):
        # Means 1,2; each sample variance=2, hence pooled SD=sqrt(2).
        result = compare([0, 2], [1, 3])
        self.assertAlmostEqual(result.cohen, 1 / math.sqrt(2))
        self.assertFalse(result.same)
        self.assertNotIn('ks', result.exceeded)  # Effect size alone is not significance.

    def test_ks_known_statistic_and_decision(self):
        result = compare([0, 0, 1, 1], [0, 1, 1, 2])
        self.assertEqual(result.ks_d, 0.25)
        self.assertAlmostEqual(result.ks, 0.25 * math.sqrt(2))
        self.assertNotIn('ks', result.exceeded)
        self.assertIn('ks', compare([0]*10, [1]*10).exceeded)

    def test_cliffs_ties_and_absolute_orientation(self):
        # Across 16 pairs: 2 left wins, 8 right wins, 6 ties -> |2-8|/16.
        self.assertEqual(compare([0, 0, 1, 1], [0, 1, 1, 2]).cliffs, 0.375)
        self.assertEqual(compare([0, 1, 1, 2], [0, 0, 1, 1]).cliffs, 0.375)

    def test_identical_and_constant_samples(self):
        for a,b in [([1,2,3],[1,2,3]),([7]*3,[7]*5)]:
            result = compare(a,b)
            self.assertTrue(result.same)
            self.assertEqual((result.cohen,result.ks,result.cliffs),(0,0,0))
        different = compare([0]*10,[1]*10)
        self.assertAlmostEqual(different.cohen / 1e32, 1)
        self.assertEqual(different.cliffs, 1)
        self.assertFalse(different.same)

    def test_unequal_groups(self):
        result = compare([0,2],[1,1,1])
        self.assertEqual(result.cohen,0)
        self.assertEqual(result.cliffs,0)
        self.assertEqual(result.ks_d,0.5)
        self.assertAlmostEqual(result.ks, math.sqrt(0.3))
        self.assertTrue(result.same)

    def test_shape_difference_without_mean_or_rank_imbalance(self):
        result=compare([-1]*50+[1]*50,[0]*100)
        self.assertEqual(result.exceeded,('ks',))
        self.assertFalse(result.same)

    def test_threshold_equality_and_combination(self):
        result=compare([0,2],[1,3])
        self.assertTrue(same([0,2],[1,3],result.cohen,result.ks,result.cliffs))
        self.assertFalse(same([0,2],[1,3],result.cohen,result.ks,result.cliffs-0.01))

    def test_insufficient_and_missing_observations(self):
        for a,b in [([],[]),([1],[2]),([],[1,2]),([None,1],[2,3])]:
            result=compare(a,b)
            self.assertIsNone(result.same)
            self.assertIsNone(result.cohen)
        result=compare([None,1,2],[1,2,None])
        self.assertEqual(result.left.count,2)
        self.assertTrue(result.same)
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            print_comparison('empty',[],[1])
        self.assertIn('insufficient data',out.getvalue())

    def test_nonfinite_rejected(self):
        for value in [float('nan'),float('inf')]:
            with self.assertRaises(ValueError):
                compare([1,value],[1,2])
        with self.assertRaises(ValueError):
            compare([1,2],[1,2],Cohen=-1)

    def test_actual_lua_reference_fixtures(self):
        fixture=json.loads((ROOT/'tests/fixtures/same_reference.json').read_text())
        for case in fixture['cases']:
            with self.subTest(case=case['name']):
                result=compare(case['x'],case['y'])
                self.assertEqual(result.same,case['same'])
                for key in ['cohen','ks','cliffs']:
                    self.assertTrue(math.isclose(getattr(result,key),case[key],rel_tol=1e-12,abs_tol=1e-12))

    def test_determinism_and_unsorted_input(self):
        self.assertEqual(compare([3,1,2],[4,2,3]),compare([1,2,3],[2,3,4]))


class CarsCertificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table=Table.load(ROOT/'data/raw/auto93.csv')

    def test_observed_cylinder_groups(self):
        groups=cylinder_groups(self.table)
        self.assertEqual({k:len(v) for k,v in groups.items()},{3:4,4:204,6:87,8:103})
        self.assertEqual(groups[3],[20]*4)
        synthetic=Table.from_csv(io.StringIO('Clndrs,Mpg+\n5,30\n5,?\n9,40\n'))
        self.assertEqual(cylinder_groups(synthetic),{5:[30],9:[40]})

    def test_fixed_year_groups_and_shared_bounds(self):
        # Full Cost range is 0..100, including year 69 outside both groups.
        t=Table.from_csv(io.StringIO('Model,Cost-\n70,20\n76,40\n77,60\n82,80\n69,0\n83,100\n75,?\n'))
        self.assertEqual(model_year_groups(t),{'70-76':[0.2,0.4],'77-82':[0.6,0.8]})
        self.assertEqual({k:len(v) for k,v in model_year_groups(self.table).items()},
                         {'70-76':216,'77-82':182})

    def test_cli_all_comparisons_and_reproducibility(self):
        cmd=[sys.executable,'-m','shrinking_code.demo','--domain','cars','--skill','certify']
        out=subprocess.check_output(cmd,cwd=ROOT,text=True)
        self.assertEqual(out,subprocess.check_output(cmd,cwd=ROOT,text=True))
        for text in ['CAR-05','CAR-06','Records: 398','3 vs 4 cylinders','3 vs 6 cylinders',
                     '3 vs 8 cylinders','4 vs 6 cylinders','4 vs 8 cylinders','6 vs 8 cylinders',
                     '70-76 vs 77-82','same=True','not causal','not proven equivalent']:
            self.assertIn(text,out)
        self.assertEqual(out.count('; same='),7)
