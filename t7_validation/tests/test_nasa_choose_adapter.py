"""Regression tests for NASA Choose bridge and common semantics."""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 't7_validation/adapters/nasa/choose.py'
spec = importlib.util.spec_from_file_location('t7_nasa_choose_adapter_tests', SRC)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class TestNASAChooseAdapter(unittest.TestCase):
    def setUp(self):
        self.csv = ROOT / 'data/raw/nasa93dem.csv'

    def copy_modified(self, modify):
        with self.csv.open(newline='', encoding='utf-8') as file:
            rows = list(csv.reader(file))
        modify(rows)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'changed.csv'
            with path.open('w', newline='', encoding='utf-8') as file:
                csv.writer(file).writerows(rows)
            return adapter.answers(path)

    def test_historical_rule_has_real_support_and_favorable_score(self):
        rule, _ = adapter.answers(self.csv)
        projects = adapter.remember.load_projects(self.csv)
        self.assertEqual(len(rule['rule']), 1)
        feature, value = next(iter(rule['rule'].items()))
        matches = [r for r in projects if r[feature] == value]
        self.assertEqual(len(matches), rule['matching'])
        self.assertGreaterEqual(len(matches), 3)

    def test_order_uses_t7_original_prediction_and_budget(self):
        _, answer = adapter.answers(self.csv)
        projects = adapter.remember.load_projects(self.csv)
        features = adapter.guess.get_features(projects)
        history = [r for r in projects if r['row_number'] % 5]
        candidates = [r for r in projects if r['row_number'] % 5 == 0]
        full = adapter.choose.select_candidates(adapter.choose.mask_proposals(candidates, features),
                                                history, features)
        self.assertEqual(answer['order'], [r['row_number'] for r in full[:12]])
        self.assertEqual(len(answer['order']), 12)
        self.assertLessEqual(len(answer['order']), min(30, len(projects) // 5))

    def test_changing_hidden_goals_does_not_change_inspection_order(self):
        _, baseline = adapter.answers(self.csv)
        def modify(rows):
            for i in range(5, len(rows), 5):
                for goal in ('Kloc+', 'Effort-', 'Defects-'):
                    rows[i][rows[0].index(goal)] = '123456789'
        _, changed = self.copy_modified(modify)
        self.assertEqual(changed, baseline)

    def test_shuffled_csv_recomputes_rule_and_order(self):
        original_rule, original_order = adapter.answers(self.csv)
        def modify(rows):
            rows[1:] = list(reversed(rows[1:]))
        new_rule, new_order = self.copy_modified(modify)
        self.assertEqual(len(new_rule['rule']), 1)
        self.assertNotEqual(original_order['order'], new_order['order'])

    def test_import_isolation_from_apache(self):
        import sys
        self.assertEqual(adapter.choose.__file__, str(ROOT / 'human_t7/nasa/choose.py'))
        self.assertEqual(adapter.guess.__file__, str(ROOT / 'human_t7/nasa/guess.py'))
        self.assertFalse(str(adapter.remember.__file__).endswith('/apache/remember.py'))


if __name__ == '__main__':
    unittest.main()
