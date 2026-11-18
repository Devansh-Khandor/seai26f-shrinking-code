"""Behavioral checks for the Apache T7 Choose shared-harness adapter."""
import csv
import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    't7_apache_choose_adapter',
    ROOT / 't7_validation' / 'adapters' / 'apache' / 'choose.py',
)
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


class TestApacheChooseAdapter(unittest.TestCase):
    def setUp(self):
        self.csv = ROOT / 'data' / 'raw' / 'Apache_AllMeasurements.csv'

    def records(self):
        with self.csv.open(newline='', encoding='utf-8') as inp:
            reader = csv.DictReader(inp)
            return reader.fieldnames, list(reader)

    def modified_csv(self, names, records):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / 'Apache_AllMeasurements.csv'
        with path.open('w', newline='', encoding='utf-8') as out:
            writer = csv.DictWriter(out, fieldnames=names)
            writer.writeheader()
            writer.writerows(records)
        return path

    def test_budget_and_best_evaluated(self):
        best, ordering = ADAPTER.answers(self.csv)
        self.assertEqual(len(best['evaluated']), 20)
        self.assertEqual(ordering['order'], best['evaluated'])
        self.assertEqual(len(set(ordering['order'])), 20)
        self.assertIn(best['best'], best['evaluated'])
        self.assertTrue(all(1 <= r <= 192 and r % 5 == 0 for r in ordering['order']))
        _, records = self.records()
        expected_best = min(best['evaluated'], key=lambda r: (float(records[r-1]['Performance-']), r))
        self.assertEqual(best['best'], expected_best)

    def test_hidden_candidate_goals_cannot_change_order(self):
        best, ordering = ADAPTER.answers(self.csv)
        names, records = self.records()
        for index, rec in enumerate(records, start=1):
            if index % 5 == 0:
                rec['Performance-'] = str(50000 + index * 100)
        mutated = self.modified_csv(names, records)
        after_best, after_order = ADAPTER.answers(mutated)
        self.assertEqual(ordering['order'], after_order['order'])
        self.assertEqual(best['evaluated'], after_best['evaluated'])
        # Once inspected, changed observations must be used in computing the winner.
        expected_best = min(after_best['evaluated'], key=lambda r: (float(records[r-1]['Performance-']), r))
        self.assertEqual(after_best['best'], expected_best)

    def test_shuffled_rows_recompute_order(self):
        before, order_before = ADAPTER.answers(self.csv)
        names, records = self.records()
        random.Random(20261118).shuffle(records)
        after, order_after = ADAPTER.answers(self.modified_csv(names, records))
        self.assertEqual(len(after['evaluated']), 20)
        self.assertEqual(len(set(order_after['order'])), 20)
        self.assertTrue(all(r % 5 == 0 for r in after['evaluated']))
        self.assertNotEqual(order_before['order'], order_after['order'])
        self.assertIn(after['best'], after['evaluated'])


if __name__ == '__main__':
    unittest.main()
