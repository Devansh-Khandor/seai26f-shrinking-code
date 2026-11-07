import io
from pathlib import Path
import unittest

from shrinking_code.data import Table
from shrinking_code.stats import NumericSummary, SymbolicSummary, summarize

ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = Table.load(ROOT / 'data/raw/auto93.csv')

    def test_rows_and_original_order(self):
        self.assertEqual(len(self.table.rows), 398)
        self.assertEqual([r.number for r in self.table.rows], list(range(1, 399)))
        self.assertEqual([c.name for c in self.table.columns],
                         ['Clndrs', 'Volume', 'HpX', 'Model', 'origin', 'Lbs-', 'Acc+', 'Mpg+'])
        self.assertEqual(self.table.rows[0].cells, (8, 304, 193, 70, '1', 4732, 18.5, 10))

    def test_roles(self):
        expected = {'x': ['Clndrs', 'Volume', 'Model', 'origin'],
                    'y': ['Lbs-', 'Acc+', 'Mpg+'], 'ignored': ['HpX']}
        for role, names in expected.items():
            self.assertEqual([c.name for c in self.table.role(role)], names)

    def test_types(self):
        self.assertEqual([c.numeric for c in self.table.columns],
                         [True, True, True, True, False, True, True, True])
        table = Table.from_csv(io.StringIO('flag,Gain+,loss-,noteX\n1,2,3,hello\n'))
        self.assertEqual(table.rows[0].cells, ('1', 2, 3, 'hello'))
        self.assertTrue(table.columns[2].numeric)  # Goal suffix makes it numeric.

    def test_missing_horsepower(self):
        missing = [r.number for r in self.table.rows if r.cells[2] is None]
        self.assertEqual(missing, [178, 207, 210, 233, 361, 391])
        self.assertEqual(summarize(self.table, self.table.columns[2]).count, 392)

    def test_car_numerical_summaries(self):
        expected = [(398, 1613, 5140, 2970.424623115578, 2803.5),
                    (398, 8, 24.8, 15.568090452261307, 15.5),
                    (398, 10, 50, 23.844221105527637, 20)]
        for c, (n, low, high, avg, med) in zip(self.table.role('y'), expected):
            s = summarize(self.table, c)
            self.assertEqual((s.count, s.minimum, s.maximum, s.median), (n, low, high, med))
            self.assertAlmostEqual(s.mean, avg)

    def test_synthetic_summary(self):
        s = NumericSummary.from_values([1, None, 2, 6, 11])
        self.assertEqual((s.count, s.minimum, s.maximum, s.mean, s.median), (4, 1, 11, 5, 4))
        self.assertEqual(NumericSummary.from_values([9, 1, 2]).median, 2)

    def test_categorical_summary(self):
        s = summarize(self.table, self.table.columns[4])
        self.assertEqual(s.count, 398)
        self.assertEqual(s.frequencies, {'1': 249, '2': 70, '3': 79})
        self.assertEqual(s.mode, '1')
        tied = SymbolicSummary.from_values(['b', None, 'a', 'a', 'b'])
        self.assertEqual((tied.count, tied.mode), (4, 'b'))

    def test_empty_summaries(self):
        s = NumericSummary.from_values([None])
        self.assertEqual((s.count, s.minimum, s.maximum, s.mean, s.median),
                         (0, None, None, None, None))
        with self.assertRaises(ValueError):
            s.normalize(0)
        sym = SymbolicSummary.from_values([None])
        self.assertEqual((sym.count, sym.frequencies, sym.mode), (0, {}, None))

    def test_whitespace_and_missing_symbols(self):
        t = Table.from_csv(io.StringIO('kind,Value+\n  h , ? \n?,2\n'))
        self.assertEqual(t.rows[0].cells, ('h', None))
        self.assertEqual(t.rows[1].cells, (None, 2))

    def test_invalid_csv_rejected(self):
        for text in ['', 'A,A\n1,2\n', 'A,b\n1\n', 'A\ninf\n',
                     'A\nnan\n', 'A\nno\n', 'A,b\n1,\n']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                Table.from_csv(io.StringIO(text))
