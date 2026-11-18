"""Behavioral tests for Apache T7 Race under the shared prudence protocol."""
import csv
import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    't7_apache_race_adapter',
    ROOT / 't7_validation' / 'adapters' / 'apache' / 'race.py',
)
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


class TestApacheRaceAdapter(unittest.TestCase):
    def setUp(self):
        self.csv = ROOT / 'data' / 'raw' / 'Apache_AllMeasurements.csv'

    def changed_file(self, change):
        with self.csv.open(newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            names, records = reader.fieldnames, list(reader)
        change(records)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / 'Apache_AllMeasurements.csv'
        with path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=names)
            writer.writeheader()
            writer.writerows(records)
        return path

    def test_fair_repeated_selection_and_reproducibility(self):
        first, predictions = ADAPTER.race_answers(self.csv)
        second, _ = ADAPTER.race_answers(self.csv)
        self.assertEqual(first, second)
        self.assertEqual(first['budget'], 30)
        self.assertEqual(first['repeats'], 30)
        self.assertEqual(len(first['ours']), 30)
        self.assertEqual(len(first['random']), 30)
        self.assertTrue(all(1 <= r <= 192 for r in first['ours']))
        self.assertTrue(all(1 <= r <= 192 for r in first['random']))
        self.assertIn(first['verdict'], ('ours better', 'random better', 'no clear difference'))
        self.assertEqual(len(predictions['predictions']), 38)

    def test_prediction_targets_are_not_training_features(self):
        _, before = ADAPTER.race_answers(self.csv)

        def mask_heldout(rows):
            for index, row in enumerate(rows, start=1):
                if index % 5 == 0:
                    row['Performance-'] = str(70000 + 100 * index)

        masked = self.changed_file(mask_heldout)
        _, after = ADAPTER.race_answers(masked)
        self.assertEqual(before['predictions'], after['predictions'])
        self.assertEqual(before['baseline'], after['baseline'])
        # Recalculated error metrics must reflect the changed measured goals.
        self.assertNotEqual(before['mae_model'], after['mae_model'])

    def test_inspection_order_ignores_unseen_goals(self):
        from sys import path as syspath
        import sys
        source = str(ROOT / 'human_t7' / 'apache')
        if source not in syspath:
            syspath.insert(0, source)
        from guess import train_test_split
        from choose import plan_inspections
        from remember import adjustable_settings, load_configurations

        original = load_configurations(self.csv)
        history, candidates = train_test_split(original, seed=42)
        plan = plan_inspections(candidates, history, adjustable_settings(original))
        hidden_changed = [dict(r) for r in candidates]
        for r in hidden_changed:
            r['Performance-'] += 99999.0
        plan_changed = plan_inspections(hidden_changed, history, adjustable_settings(original))
        self.assertEqual(
            [r['row_number'] for r in plan],
            [r['row_number'] for r in plan_changed],
        )

    def test_shuffled_dataset_recomputes_and_still_has_valid_ids(self):
        def shuffle(rows):
            random.Random(20261117).shuffle(rows)
        changed = self.changed_file(shuffle)
        selection, pred = ADAPTER.race_answers(changed)
        self.assertEqual(len(selection['ours']), 30)
        self.assertEqual(len(selection['random']), 30)
        self.assertEqual(len(pred['predictions']), 38)
        self.assertTrue(all(int(k) % 5 == 0 for k in pred['predictions']))


if __name__ == '__main__':
    unittest.main()
