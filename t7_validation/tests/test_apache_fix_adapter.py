"""Behavioral checks for the Apache T7 Fix common-check bridge."""
import csv
import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    't7_apache_fix_adapter',
    ROOT / 't7_validation' / 'adapters' / 'apache' / 'fix.py',
)
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


class TestApacheFixAdapter(unittest.TestCase):
    def setUp(self):
        self.csv = ROOT / 'data' / 'raw' / 'Apache_AllMeasurements.csv'

    def records(self):
        with self.csv.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return reader.fieldnames, list(reader)

    def temporary_csv(self, names, records):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        output = Path(folder.name) / 'Apache_AllMeasurements.csv'
        with output.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=names)
            writer.writeheader()
            writer.writerows(records)
        return output

    def check_semantics(self, path):
        first, second = ADAPTER.answers(path)
        names, data = self.records() if path == self.csv else self._read(path)
        features = [k for k in names if k not in ('base', 'Performance-')]
        src = data[first['row']-1]
        alt = data[first['alternative']-1]
        self.assertGreater(float(src['Performance-']), float(alt['Performance-']))
        changed = {k for k in features if src[k] != alt[k]}
        self.assertEqual(set(first['changes']), changed)
        self.assertIn(len(changed), (1, 2))
        self.assertEqual(first['performance']['before'], float(src['Performance-']))
        self.assertEqual(first['performance']['after'], float(alt['Performance-']))
        self.assertEqual(second['row'], first['row'])
        target = second['target']
        chosen = data[second['alternative']-1]
        changed2 = {k for k in features if src[k] != chosen[k]}
        self.assertEqual(set(second['changed']), changed2)
        old_gap = abs(float(src['Performance-']) - target)
        new_gap = abs(float(chosen['Performance-']) - target)
        self.assertLess(new_gap, old_gap)
        eligible = [
            r for r in data
            if abs(float(r['Performance-']) - target) < old_gap
        ]
        minimum = min(sum(src[k] != r[k] for k in features) for r in eligible)
        self.assertEqual(len(changed2), minimum)
        return first, second

    @staticmethod
    def _read(path):
        with path.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return reader.fieldnames, list(reader)

    def test_observed_repair_and_minimum_changed_settings(self):
        self.check_semantics(self.csv)

    def test_shuffling_recomputes_row_identifiers_and_repairs(self):
        names, rows = self.records()
        random.Random(20261117).shuffle(rows)
        shuffled = self.temporary_csv(names, rows)
        before, _ = ADAPTER.answers(self.csv)
        after, _ = self.check_semantics(shuffled)
        self.assertNotEqual(before['row'], after['row'])

    def test_predictions_not_hardcoded_to_original_target_values(self):
        names, rows = self.records()
        # Alter *all* observed outcomes with a positive affine transform:
        # rankings and nearest-neighbor predicted-improvement signs remain
        # invariant, but outputs must report the altered recorded outcomes.
        for r in rows:
            r['Performance-'] = str(float(r['Performance-']) * 2 + 100)
        changed = self.temporary_csv(names, rows)
        original, _ = ADAPTER.answers(self.csv)
        modified, _ = self.check_semantics(changed)
        self.assertAlmostEqual(modified['performance']['before'],
                               original['performance']['before'] * 2 + 100)
        self.assertAlmostEqual(modified['performance']['after'],
                               original['performance']['after'] * 2 + 100)


if __name__ == '__main__':
    unittest.main()
