"""Behavioral tests of the Apache T7 Justify bridge."""
import csv
import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'apache_justify_adapter',
    ROOT / 't7_validation' / 'adapters' / 'apache' / 'justify.py',
)
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


class TestApacheJustifyAdapter(unittest.TestCase):
    def setUp(self):
        self.csv = ROOT / 'data' / 'raw' / 'Apache_AllMeasurements.csv'

    def raw(self):
        with self.csv.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            return reader.fieldnames, list(reader)

    def temp_csv(self, fields, records):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / 'apache.csv'
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        return path

    def test_recommendation_uses_historical_evidence(self):
        rec, pair = ADAPTER.answers(self.csv)
        self.assertEqual(rec['recommended'] % 5, 0)
        self.assertGreaterEqual(len(rec['neighbors']), 3)
        fields, records = self.raw()
        for neighbor in rec['neighbors']:
            self.assertNotEqual(neighbor['row'] % 5, 0)
            self.assertNotEqual(neighbor['row'], rec['recommended'])
            actual = float(records[neighbor['row'] - 1]['Performance-'])
            self.assertAlmostEqual(neighbor['values']['Performance-'], actual)
        self.assertIn(pair['preferred'], ('a', 'b'))

    def test_pair_preference_matches_recorded_outcome(self):
        _, pair = ADAPTER.answers(self.csv)
        _, records = self.raw()
        a, b = pair['a'], pair['b']
        self.assertNotEqual(a, b)
        self.assertTrue(a % 5 and b % 5)
        self.assertEqual(len(pair['differences']), 1)
        change = pair['differences'][0]
        self.assertNotEqual(change['a'], change['b'])
        self.assertEqual(str(change['a']), records[a-1][change['feature']])
        self.assertEqual(str(change['b']), records[b-1][change['feature']])
        preferred_row = a if pair['preferred'] == 'a' else b
        other = b if preferred_row == a else a
        self.assertLess(float(records[preferred_row-1]['Performance-']),
                        float(records[other-1]['Performance-']))
        self.assertEqual(pair['tradeoffs'], [])

    def test_shuffling_recalculates_row_ids_and_evidence(self):
        before, pair_before = ADAPTER.answers(self.csv)
        fields, records = self.raw()
        random.Random(20261117).shuffle(records)
        path = self.temp_csv(fields, records)
        after, pair_after = ADAPTER.answers(path)
        self.assertNotEqual(before, after)
        _, shuffled_records = self.raw_from(path)
        for evidence in after['neighbors']:
            value = float(shuffled_records[evidence['row']-1]['Performance-'])
            self.assertAlmostEqual(evidence['values']['Performance-'], value)
        self.assertNotEqual((pair_before['a'], pair_before['b']),
                            (pair_after['a'], pair_after['b']))

    @staticmethod
    def raw_from(path):
        with path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            return reader.fieldnames, list(reader)


if __name__ == '__main__':
    unittest.main()
