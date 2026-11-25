"""Regression tests for NASA Fix adapter and common answer contract."""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 't7_validation/adapters/nasa/fix.py'
spec = importlib.util.spec_from_file_location('t7_nasa_fix_adapter_tests', SRC)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class TestNASAFixAdapter(unittest.TestCase):
    def setUp(self):
        self.csv = ROOT / 'data/raw/nasa93dem.csv'

    def altered(self, mutate):
        with self.csv.open(newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f))
        mutate(rows)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'altered.csv'
            with path.open('w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerows(rows)
            return adapter.answers(path)

    def test_answers_follow_original_t7_repair(self):
        effort, defects = adapter.answers(self.csv)
        rows = adapter.remember.load_projects(self.csv)
        history = [r for r in rows if r['row_number'] % 5]
        features = adapter.guess.get_features(rows)
        for answer, target in ((effort, 'Effort-'), (defects, 'Defects-')):
            project = next(r for r in rows if r['row_number'] == answer['project'])
            self.assertEqual(answer['project'] % 5, 0)
            baseline, repair, _ = adapter.fix.find_repair(
                project, history, features, target,
                adapter.fix.MAX_EFFORT_INCREASE if target == 'Defects-' else None)
            self.assertIsNotNone(repair)
            self.assertEqual(answer['changes'], {k: [a, b] for k, (a, b) in repair['changes'].items()})
            key = 'predicted_effort' if target == 'Effort-' else 'predicted_defects'
            self.assertEqual(answer[key], {'before': baseline[target], 'after': repair['predicted'][target]})
            self.assertIn(len(answer['changes']), (1, 2))
            if target == 'Defects-':
                self.assertEqual(answer['predicted_effort'], {
                    'before': baseline['Effort-'], 'after': repair['predicted']['Effort-']})

    def test_masking_all_heldout_goals_cannot_change_repair(self):
        baseline = adapter.answers(self.csv)
        def mutate(rows):
            idxs = [rows[0].index(g) for g in ('Kloc+', 'Effort-', 'Defects-')]
            for i in range(5, len(rows), 5):
                for j in idxs:
                    rows[i][j] = '123456789'
        self.assertEqual(self.altered(mutate), baseline)

    def test_shuffled_data_recomputes_repairs_and_row_numbers(self):
        originals = adapter.answers(self.csv)
        def mutate(rows):
            rows[1:] = list(reversed(rows[1:]))
        result = self.altered(mutate)
        self.assertEqual(len(result), 2)
        for answer in result:
            self.assertEqual(answer['project'] % 5, 0)
            self.assertTrue(answer['changes'])
        self.assertNotEqual([r['project'] for r in result], [r['project'] for r in originals])

    def test_import_isolation_from_apache(self):
        self.assertEqual(adapter.fix.__file__, str(ROOT / 'human_t7/nasa/fix.py'))
        self.assertEqual(adapter.guess.__file__, str(ROOT / 'human_t7/nasa/guess.py'))
        self.assertEqual(adapter.remember.__file__, str(ROOT / 'human_t7/nasa/remember.py'))


if __name__ == '__main__':
    unittest.main()
