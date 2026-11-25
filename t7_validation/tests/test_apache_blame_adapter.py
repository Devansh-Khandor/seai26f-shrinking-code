"""Tests for the Apache T7 Blame adapter and its source-data behavior."""
import csv
import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    't7_apache_blame_adapter',
    ROOT / 't7_validation' / 'adapters' / 'apache' / 'blame.py',
)
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


class TestApacheBlameAdapter(unittest.TestCase):
    def setUp(self):
        self.csv = ROOT / 'data' / 'raw' / 'Apache_AllMeasurements.csv'

    def load_raw(self):
        with self.csv.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            return reader.fieldnames, list(reader)

    def temp_csv(self, names, records):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / 'apache.csv'
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=names)
            writer.writeheader()
            writer.writerows(records)
        return path

    def test_rank_and_pattern_share_accuracy(self):
        first, second = ADAPTER.answers(self.csv)
        self.assertGreaterEqual(len(first['poor']), 5)
        self.assertGreaterEqual(len(first['ranking']), 3)
        self.assertEqual(len(set(first['ranking'])), len(first['ranking']))
        self.assertFalse(set(second['bad']) & set(second['good']))
        self.assertEqual(len(second['bad']) + len(second['good']), 192)
        names, records = self.load_raw()
        for item in second['combinations']:
            self.assertGreaterEqual(len(item['settings']), 2)
            self.assertTrue(set(item['settings']) <= set(names) - {'base', 'Performance-'})
            for group, prop in (('bad', 'bad_share'), ('good', 'good_share')):
                matches = sum(
                    all(int(records[r - 1][k]) == v for k, v in item['settings'].items())
                    for r in second[group]
                )
                self.assertAlmostEqual(item[prop], matches / len(second[group]))

    def test_shuffling_recomputes_row_numbers(self):
        original, _ = ADAPTER.answers(self.csv)
        names, records = self.load_raw()
        random.Random(20261117).shuffle(records)
        shuffled, _ = ADAPTER.answers(self.temp_csv(names, records))
        self.assertNotEqual(original['poor'], shuffled['poor'])
        self.assertEqual(len(original['poor']), len(shuffled['poor']))
        self.assertEqual(original['ranking'], shuffled['ranking'])

    def test_goal_changes_recalculate_associations(self):
        old_first, old_second = ADAPTER.answers(self.csv)
        names, records = self.load_raw()
        for rec in records:
            if rec['keepAlive'] == '1':
                rec['Performance-'] = '500'
            else:
                rec['Performance-'] = '2500'
        changed_first, changed_second = ADAPTER.answers(self.temp_csv(names, records))
        self.assertNotEqual(set(old_first['poor']), set(changed_first['poor']))
        self.assertNotEqual(old_second['combinations'], changed_second['combinations'])


if __name__ == '__main__':
    unittest.main()
