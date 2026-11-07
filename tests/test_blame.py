import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

from shrinking_code.data import Row, Table
from shrinking_code.diagnosis import compare_groups, outcome_groups, select_quartiles
from shrinking_code.ranking import rank_rows

ROOT = Path(__file__).resolve().parents[1]


def table(text):
    return Table.from_csv(io.StringIO(text))


def groups(text, cut):
    t = table(text)
    return Table(t.columns, t.rows[:cut]), Table(t.columns, t.rows[cut:])


class DiagnosisTests(unittest.TestCase):
    def test_quartiles_by_count_and_disjointness(self):
        t = table('X,Goal+\n0,8\n1,1\n2,7\n3,2\n4,6\n5,3\n6,5\n7,4\n8,9\n')
        result = outcome_groups(t, 'Goal+')
        self.assertEqual([r.number for r in result.poor.rows], [2, 4])
        self.assertEqual([r.number for r in result.favorable.rows], [1, 9])
        self.assertEqual(result.eligible_count, 9)
        self.assertEqual((result.poor_outcome.mean, result.favorable_outcome.mean), (1.5, 8.5))
        self.assertFalse(set(result.poor.rows) & set(result.favorable.rows))

    def test_all_ties_and_input_order(self):
        t = table('X,Goal+\n0,5\n1,5\n2,5\n3,5\n4,5\n5,5\n6,5\n7,5\n')
        result = outcome_groups(t, 'Goal+')
        self.assertEqual([r.number for r in result.poor.rows], [1, 2])
        self.assertEqual([r.number for r in result.favorable.rows], [7, 8])
        self.assertEqual(result, outcome_groups(Table(t.columns, tuple(reversed(t.rows))), 'Goal+'))
        self.assertEqual(result.poor_outcome, result.favorable_outcome)

    def test_missing_outcomes_and_insufficient_data(self):
        t = table('X,Goal+\n0,?\n1,1\n2,2\n3,3\n4,4\n')
        result = outcome_groups(t, 'Goal+')
        self.assertEqual(result.eligible_count, 4)
        self.assertEqual(result.poor.rows[0].number, 2)
        for rows in [(), t.rows[:3]]:
            with self.assertRaisesRegex(ValueError, 'four'):
                outcome_groups(Table(t.columns, rows), 'Goal+')
        with self.assertRaises(ValueError):
            outcome_groups(t, 'absent')
        with self.assertRaises(ValueError):
            select_quartiles(t, {1: float('inf')})

    def test_minimize_outcome_and_custom_scores(self):
        t = table('X,Cost-\n0,0\n1,10\n2,20\n3,30\n')
        result = outcome_groups(t, 'Cost-', higher_better=False)
        self.assertEqual(result.poor.rows[0].number, 4)
        self.assertEqual(result.favorable.rows[0].number, 1)
        self.assertEqual(result, select_quartiles(t, {1: 0, 2: 10, 3: 20, 4: 30}, False))

    def test_d2h_reuses_full_table_bounds(self):
        t = table('X,Cost-\n0,0\n1,20\n2,40\n3,60\n4,80\n5,100\n6,120\n7,140\n')
        result = outcome_groups(t)
        self.assertEqual([r.number for r in result.favorable.rows], [1, 2])
        self.assertEqual([r.number for r in result.poor.rows], [7, 8])
        self.assertAlmostEqual(result.favorable_outcome.mean, 1 / 14)
        self.assertAlmostEqual(result.poor_outcome.mean, 13 / 14)

    def test_numeric_evidence_and_rank(self):
        a,b = groups('Weak,Strong,Goal+\n0,10,0\n2,20,0\n0,0,1\n2,1,1\n', 2)
        findings = compare_groups(a,b)
        self.assertEqual([f.column.name for f in findings], ['Strong', 'Weak'])
        strong, weak = findings
        self.assertEqual((strong.poor.mean, strong.poor.median), (15, 15))
        self.assertEqual((strong.favorable.mean, strong.favorable.median), (0.5, 0.5))
        self.assertEqual(strong.separation, 1)
        self.assertEqual(strong.direction, 'higher mean in poor')
        self.assertFalse(strong.certification.same)
        self.assertEqual(weak.separation, 0)

    def test_categorical_codes_absent_categories_and_tv(self):
        a,b = groups('origin,Goal+\n1,0\n1,0\n2,0\n1,1\n3,1\n3,1\n', 3)
        finding = compare_groups(a,b)[0]
        self.assertFalse(finding.column.numeric)
        self.assertAlmostEqual(finding.separation, 2 / 3)
        self.assertEqual(finding.poor.frequencies, {'1': 2, '2': 1})
        self.assertEqual(finding.favorable.frequencies, {'1': 1, '3': 2})
        self.assertEqual([e.value for e in finding.categories], ['1','2','3'])
        self.assertAlmostEqual(finding.categories[0].difference, 1 / 3)
        self.assertEqual(finding.categories[2].poor_count, 0)
        self.assertIsNone(finding.certification)
        self.assertTrue(finding.limitations)

    def test_constant_features_and_tied_rank(self):
        a,b = groups('A,B,kind,Goal+\n5,5,x,0\n5,5,x,0\n5,5,x,1\n5,5,x,1\n', 2)
        findings = compare_groups(a,b)
        self.assertEqual([f.column.name for f in findings], ['A','B','kind'])
        self.assertEqual([f.separation for f in findings], [0,0,0])
        self.assertTrue(findings[0].certification.same)

    def test_missing_values_and_unequal_sizes(self):
        a,b = groups('Size,kind,Goal+\n0,a,0\n2,a,0\n?, ?,0\n4,a,1\n6,b,1\n', 3)
        numeric, categorical = compare_groups(a,b)
        self.assertEqual((numeric.poor_total, numeric.favorable_total), (3,2))
        self.assertEqual((numeric.poor.count, numeric.favorable.count), (2,2))
        self.assertEqual(numeric.separation, 1)
        self.assertEqual(numeric.direction, 'lower mean in poor')
        self.assertEqual(categorical.separation, 0.5)
        self.assertTrue(numeric.limitations)
        self.assertEqual(categorical.categories[0].poor_proportion, 1)

    def test_insufficient_feature_evidence_and_empty_groups(self):
        a,b = groups('Size,kind,Goal+\n?,a,0\n?, ?,0\n1,b,1\n2,b,1\n', 2)
        findings = compare_groups(a,b)
        self.assertTrue(all(f.separation is None for f in findings))
        self.assertEqual(findings[0].direction, 'unknown')
        self.assertIsNone(findings[0].certification.same)
        self.assertTrue(all(f.limitations for f in findings))
        self.assertTrue(all(f.separation is None for f in compare_groups(Table(a.columns,()),b)))

    def test_overlap_and_schema_rejected(self):
        a,b = groups('X,Goal+\n0,0\n1,1\n',1)
        with self.assertRaises(ValueError):
            compare_groups(a,a)
        with self.assertRaises(ValueError):
            compare_groups(a,table('other,Goal+\nx,1\n'))

    def test_no_y_ignored_or_row_id_explanations(self):
        a,b = groups('Size,kind,HpX,Goal+,Other-\n10,a,1,0,10\n20,b,2,0,20\n0,a,3,1,30\n1,a,4,1,40\n',2)
        expected = compare_groups(a,b)
        self.assertEqual({f.column.name for f in expected},{'Size','kind'})
        def alter(t):
            return Table(t.columns,tuple(Row(r.number+100,(r.cells[0],r.cells[1],999,None,-999))
                                         for r in t.rows))
        self.assertEqual(expected, compare_groups(alter(a),alter(b)))

    def test_categorical_ranking_separate_from_numeric(self):
        a,b = groups('weak,strong,Numeric,Goal+\nx,a,1,0\ny,a,1,0\nx,b,1,1\ny,b,1,1\n',2)
        findings = compare_groups(a,b)
        self.assertEqual([f.column.name for f in findings],['Numeric','strong','weak'])
        self.assertEqual([f.separation for f in findings],[0,1,0])


class BlameIntegrationTests(unittest.TestCase):
    def test_real_groups_and_original_ranking(self):
        t = Table.load(ROOT/'data/raw/auto93.csv')
        mpg = outcome_groups(t,'Mpg+')
        overall = outcome_groups(t)
        self.assertEqual((len(mpg.poor.rows),len(mpg.favorable.rows)),(99,99))
        self.assertEqual([r.number for r in mpg.poor.rows], list(range(1,100)))
        self.assertEqual([r.number for r in mpg.favorable.rows], list(range(300,399)))
        ranked = rank_rows(t)
        self.assertEqual(overall.favorable.rows, tuple(r for r,_ in ranked[:99]))
        self.assertEqual(overall.poor.rows, tuple(r for r,_ in ranked[-99:]))

    def test_cli_reproducible_and_evidence_present(self):
        command = [sys.executable,'-m','shrinking_code.demo','--domain','cars','--skill','blame']
        out = subprocess.check_output(command,cwd=ROOT,text=True)
        self.assertEqual(out,subprocess.check_output(command,cwd=ROOT,text=True))
        for text in ['CAR-09','CAR-10','poor: n=99','favorable: n=99','Cliff delta',
                     'total variation','Volume (numeric)','origin (categorical)',
                     'counts=96/25','not causal','Coarse tied MPG']:
            self.assertIn(text,out)

    def test_all_prior_demo_outputs_unchanged(self):
        fixture = json.loads((ROOT/'tests/fixtures/pre_blame_demo_hashes.json').read_text())
        for skill,digest in fixture['sha256'].items():
            with self.subTest(skill=skill):
                out = subprocess.check_output([sys.executable,'-m','shrinking_code.demo',
                                               '--domain','cars','--skill',skill],cwd=ROOT)
                self.assertEqual(hashlib.sha256(out).hexdigest(),digest)
