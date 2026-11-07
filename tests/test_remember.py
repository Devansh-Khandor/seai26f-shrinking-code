import csv
import io
import math
from pathlib import Path
import subprocess
import sys
import unittest

from shrinking_code.data import Table
from shrinking_code.ranking import distance_to_heaven, rank_rows
from shrinking_code.stats import summarize

ROOT = Path(__file__).resolve().parents[1]


def table(text):
    return Table.from_csv(io.StringIO(text))


class RememberTests(unittest.TestCase):
    def test_normalization_and_directions(self):
        t = table('Cost-,Gain+\n10,20\n30,60\n15,30\n')
        low, high = t.columns
        self.assertEqual((low.ideal, high.ideal), (0, 1))
        for c in t.columns:
            s = summarize(t, c)
            self.assertEqual(s.normalize(t.rows[0].cells[c.index]), 0)
            self.assertEqual(s.normalize(t.rows[1].cells[c.index]), 1)
            self.assertEqual(s.normalize(t.rows[2].cells[c.index]), 0.25)

    def test_independently_known_distances(self):
        # Ranges 10..30 and 20..60. Deviations: (0,0), (1,1), (.25,.5).
        t = table('Cost-,Gain+\n10,60\n30,20\n15,40\n')
        goals = t.role('y')
        summaries = {c.index: summarize(t, c) for c in goals}
        expected = [0, 1, math.sqrt(5 / 32)]
        for row, answer in zip(t.rows, expected):
            self.assertAlmostEqual(distance_to_heaven(row, goals, summaries), answer)
        self.assertEqual([r.number for r, _ in rank_rows(t)], [1, 3, 2])

    def test_ties_follow_original_row_numbers(self):
        t = table('Cost-,Gain+\n0,0\n10,10\n0,0\n')
        reversed_table = Table(t.columns, tuple(reversed(t.rows)))
        self.assertEqual([r.number for r, _ in rank_rows(reversed_table)], [1, 2, 3])

    def test_constant_goals(self):
        t = table('Cost-,Gain+\n7,5\n7,5\n')
        self.assertEqual(summarize(t, t.columns[0]).normalize(7), 0.5)
        self.assertEqual([(r.number, s) for r, s in rank_rows(t)], [(1, 0.5), (2, 0.5)])
        mixed = table('Cost-,Gain+\n7,0\n7,10\n')
        self.assertAlmostEqual(rank_rows(mixed)[0][1], math.sqrt(0.25 / 2))

    def test_missing_goals_excluded_but_observed_range_retained(self):
        t = table('Cost-,Gain+\n0,?\n10,0\n20,10\n?,10\n')
        goals = t.role('y')
        summaries = {c.index: summarize(t, c) for c in goals}
        self.assertIsNone(distance_to_heaven(t.rows[0], goals, summaries))
        ranked = rank_rows(t)
        self.assertEqual([r.number for r, _ in ranked], [3, 2])
        # Cost min=0 comes from the excluded first row, not the complete-case subset.
        self.assertAlmostEqual(ranked[0][1], math.sqrt(1 / 2))
        self.assertAlmostEqual(ranked[1][1], math.sqrt(1.25 / 2))

    def test_no_complete_rows_and_no_goals(self):
        self.assertEqual(rank_rows(table('Cost-,Gain+\n?,1\n?,2\n')), [])
        self.assertEqual(rank_rows(table('Cost-\n')), [])
        with self.assertRaisesRegex(ValueError, 'at least one'):
            rank_rows(table('value\na\n'))

    def test_car_top_five_against_independent_raw_csv_calculation(self):
        # Read raw cells separately; compare all scores using explicit car ranges/formula.
        path = ROOT / 'data/raw/auto93.csv'
        with path.open() as source:
            records = list(csv.DictReader(source))
        expected = []
        for number, r in enumerate(records, 1):
            weight = (float(r['Lbs-']) - 1613) / 3527
            acceleration = (24.8 - float(r['Acc+'])) / 16.8
            mpg = (50 - float(r['Mpg+'])) / 40
            expected.append((math.sqrt((weight**2 + acceleration**2 + mpg**2) / 3), number))
        expected.sort()
        actual = rank_rows(Table.load(path))
        self.assertEqual(len(actual), 398)
        self.assertEqual([r.number for r, _ in actual[:5]], [n for _, n in expected[:5]])
        for (row, score), (answer, number) in zip(actual, expected):
            self.assertEqual(row.number, number)
            self.assertAlmostEqual(score, answer)

    def test_ranking_responds_to_arbitrary_data(self):
        # Different headers, sizes, ranges, input categories and a changed winner.
        a = table('label,Reward+\nz,100\ny,500\nx,300\n')
        b = table('label,Reward+\nz,900\ny,500\nx,300\n')
        self.assertEqual([r.number for r, _ in rank_rows(a)], [2, 3, 1])
        self.assertEqual([r.number for r, _ in rank_rows(b)], [1, 2, 3])
        self.assertEqual(rank_rows(a)[0][1], 0)

    def test_cli_demo(self):
        result = subprocess.run([sys.executable, '-m', 'shrinking_code.demo',
                                 '--domain', 'cars', '--skill', 'remember'],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for text in ['Records: 398', 'CAR-01', 'CAR-02', 'Ignored: HpX',
                     'n=398', 'mean=2970.424623', 'median=2803.5',
                     'Excluded for missing goals: 0', 'not prediction']:
            self.assertIn(text, result.stdout)
        printed = [line.split() for line in result.stdout.splitlines()
                   if line and line[0].isdigit()]
        expected = rank_rows(Table.load(ROOT / 'data/raw/auto93.csv'))[:5]
        self.assertEqual(len(printed), 5)
        for fields, (row, score) in zip(printed, expected):
            self.assertEqual(int(fields[0]), row.number)
            self.assertAlmostEqual(float(fields[-1]), score, places=6)

    def test_unsupported_cli_requests(self):
        for domain, skill in [('nasa', 'remember'), ('apache', 'remember'), ('cars', 'guess')]:
            result = subprocess.run([sys.executable, '-m', 'shrinking_code.demo',
                                     '--domain', domain, '--skill', skill],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('only cars / remember is implemented', result.stderr)
            self.assertEqual(result.stdout, '')
