"""Behavioral tests for the Apache T7 Guess adapter."""
import csv
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    't7_apache_guess_adapter',
    ROOT / 't7_validation' / 'adapters' / 'apache' / 'guess.py',
)
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


class TestApacheGuessAdapter(unittest.TestCase):
    def setUp(self):
        self.csv = ROOT / 'data' / 'raw' / 'Apache_AllMeasurements.csv'

    def test_all_held_out_rows_and_baseline(self):
        prediction, pairs = ADAPTER.answers(self.csv)
        self.assertEqual(set(map(int, prediction['predictions'])), set(range(5, 193, 5)))
        self.assertEqual(len(prediction['predictions']), 38)
        self.assertTrue(math.isfinite(prediction['mae']))
        self.assertLess(prediction['mae'], 464.60)
        self.assertEqual(len(pairs['pairs']), 10)

    def test_observed_pairs_have_exactly_one_setting_difference(self):
        from remember import load_configurations, adjustable_settings
        rows = load_configurations(self.csv)
        settings = adjustable_settings(rows)
        _, pairs = ADAPTER.answers(self.csv)
        for pair in pairs['pairs']:
            a, b = rows[pair['a'] - 1], rows[pair['b'] - 1]
            differs = [s for s in settings if a[s] != b[s]]
            self.assertEqual(differs, [pair['setting']])
            self.assertLessEqual(pair['low'], pair['difference'])
            self.assertLessEqual(pair['difference'], pair['high'])

    def test_masking_held_out_targets_does_not_change_predictions(self):
        original, _ = ADAPTER.answers(self.csv)
        with tempfile.TemporaryDirectory() as tmp:
            masked = Path(tmp) / 'masked.csv'
            with self.csv.open(newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                rows = list(reader)
            for index, row in enumerate(rows, start=1):
                if index % 5 == 0:
                    row['Performance-'] = '99999'
            with masked.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            updated, _ = ADAPTER.answers(masked)
        self.assertEqual(original['predictions'], updated['predictions'])
        # MAE is retrospective evaluation and correctly changes with targets.
        self.assertNotEqual(original['mae'], updated['mae'])


if __name__ == '__main__':
    unittest.main()
