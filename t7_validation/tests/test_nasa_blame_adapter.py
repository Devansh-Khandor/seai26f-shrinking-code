"""Contract, data-perturbation, and import isolation tests for NASA Blame."""
import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 't7_validation/adapters/nasa/blame.py'
spec = importlib.util.spec_from_file_location('test_nasa_blame_adapter_module', SCRIPT)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class TestNASABlameAdapter(unittest.TestCase):
    def setUp(self):
        self.csv = ROOT / 'data/raw/nasa93dem.csv'

    def modified_csv(self, transform):
        with self.csv.open(newline='', encoding='utf-8') as fh:
            rows = list(csv.reader(fh))
        transform(rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.csv'
            with path.open('w', newline='', encoding='utf-8') as fh:
                csv.writer(fh).writerows(rows)
            return adapter.answers(path)

    def test_answer_contract_and_t7_sources(self):
        a, b = adapter.answers(self.csv)
        projects = adapter.remember.load_projects(self.csv)
        _, high_effort, effort = adapter.blame.diagnose(projects, 'Effort-')
        low_defects, high_defects, defect = adapter.blame.diagnose(projects, 'Defects-')
        self.assertEqual(a['poor'], [row['row_number'] for row in high_effort])
        self.assertEqual(b['bad'], [row['row_number'] for row in high_defects])
        self.assertEqual(b['good'], [row['row_number'] for row in low_defects])
        self.assertEqual(len(set(a['ranking'])), len(a['ranking']))
        self.assertEqual(set(a['ranking']), set(adapter.guess.get_features(projects)))
        self.assertEqual([d['feature'] for d in b['differences']],
                         [item['feature'] for item in defect[:10]])

    def test_ordinal_ranking_distinct_from_original_tv_ranking(self):
        projects = adapter.remember.load_projects(self.csv)
        low, high, tv_rank = adapter.blame.diagnose(projects, 'Effort-')
        ranked = adapter.effort_ranking(projects, low, high, tv_rank)
        self.assertEqual(tv_rank[0]['feature'], 'acap')
        self.assertNotEqual(ranked[0], tv_rank[0]['feature'])
        self.assertIn(ranked[0], {'data', 'time', 'rely'})

    def test_shuffling_recomputes_which_rows_are_poor(self):
        original = adapter.answers(self.csv)
        changed = self.modified_csv(lambda rows: rows.__setitem__(slice(1, None), list(reversed(rows[1:]))))
        self.assertNotEqual(original[0]['poor'], changed[0]['poor'])
        self.assertEqual(original[0]['ranking'], changed[0]['ranking'])

    def test_changing_effort_and_defects_changes_diagnosis(self):
        original = adapter.answers(self.csv)
        def flip(rows):
            h = rows[0]
            for r in rows[1:]:
                for target in ('Effort-', 'Defects-'):
                    i = h.index(target)
                    r[i] = str(100000.0 - float(r[i]))
        changed = self.modified_csv(flip)
        self.assertNotEqual(original[0]['poor'], changed[0]['poor'])
        self.assertNotEqual(original[1]['bad'], changed[1]['bad'])

    def test_loading_does_not_override_apache_modules(self):
        self.assertEqual(adapter.blame.__name__, 't7_nasa_original_blame')
        self.assertNotEqual(adapter.remember.__file__, str(ROOT/'human_t7/apache/remember.py'))


if __name__ == '__main__':
    unittest.main()
