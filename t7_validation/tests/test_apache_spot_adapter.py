"""Source-level spot adapter regression checks."""
import csv
import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    't7_apache_spot_adapter',
    ROOT / 't7_validation' / 'adapters' / 'apache' / 'spot.py')
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


class TestApacheSpotAdapter(unittest.TestCase):
    def setUp(self):
        self.csv = ROOT / 'data' / 'raw' / 'Apache_AllMeasurements.csv'

    def temporary_csv(self, rows):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / 'data.csv'
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def records(self):
        with self.csv.open(newline='', encoding='utf-8') as f:
            return list(csv.DictReader(f))

    def test_original_anomaly_fields_and_valid_rows(self):
        y, x = ADAPTER.answers(self.csv)
        self.assertGreaterEqual(len(y['flagged']), 1)
        self.assertGreaterEqual(len(x['flagged']), 1)
        for finding in y['flagged']:
            self.assertEqual(finding['column'], 'Performance-')
            self.assertTrue(1 <= finding['row'] <= 192)
            self.assertIsInstance(finding['expected'], (float, int))
        self.assertEqual(set(x['scores']), set(map(str, x['flagged'])))
        self.assertEqual(len(x['flagged']), len(set(x['flagged'])))

    def test_shuffled_rows_recompute_ids(self):
        rows = self.records()
        random.Random(20261117).shuffle(rows)
        new_csv = self.temporary_csv(rows)
        before_y, before_x = ADAPTER.answers(self.csv)
        after_y, after_x = ADAPTER.answers(new_csv)
        self.assertNotEqual(before_y['flagged'][0]['row'], after_y['flagged'][0]['row'])
        self.assertNotEqual(before_x['flagged'], after_x['flagged'])

    def test_affine_performance_transform_preserves_input_anomalies(self):
        rows = self.records()
        for r in rows:
            r['Performance-'] = str(float(r['Performance-']) * 2 + 100)
        transformed = self.temporary_csv(rows)
        original_y, original_x = ADAPTER.answers(self.csv)
        changed_y, changed_x = ADAPTER.answers(transformed)
        self.assertEqual(original_x, changed_x)
        self.assertEqual([e['row'] for e in original_y['flagged']],
                         [e['row'] for e in changed_y['flagged']])
        for old, new in zip(original_y['flagged'], changed_y['flagged']):
            self.assertAlmostEqual(new['actual'], 2*old['actual']+100)
            self.assertAlmostEqual(new['expected'], 2*old['expected']+100)


if __name__ == '__main__':
    unittest.main()
