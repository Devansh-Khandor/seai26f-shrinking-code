"""NASA Spot adapter contract, data dependence and import-isolation tests."""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 't7_validation/adapters/nasa/spot.py'
spec = importlib.util.spec_from_file_location('nasa_spot_adapter_test_module', SRC)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class TestNASASpotAdapter(unittest.TestCase):
    def setUp(self):
        self.csv = ROOT / 'data/raw/nasa93dem.csv'

    def altered(self, edit):
        with self.csv.open(newline='', encoding='utf-8') as fh:
            rows = list(csv.reader(fh))
        edit(rows)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'altered.csv'
            with path.open('w', newline='', encoding='utf-8') as fh:
                csv.writer(fh).writerows(rows)
            return adapter.answers(path)

    def test_input_anomaly_contract_and_original_scores(self):
        x, _ = adapter.answers(self.csv)
        projects = adapter.remember.load_projects(self.csv)
        features = adapter.guess.get_features(projects)
        expected = adapter.spot.input_anomalies(projects, features)[:5]
        self.assertEqual(x['flagged'], [r['row_number'] for r in expected])
        self.assertEqual(set(x['scores']), {str(r) for r in x['flagged']})
        for r in expected:
            self.assertAlmostEqual(x['scores'][str(r['row_number'])], r['score'])

    def test_residuals_match_original_loo_predictions(self):
        _, y = adapter.answers(self.csv)
        projects = adapter.remember.load_projects(self.csv)
        features = adapter.guess.get_features(projects)
        self.assertEqual(len(y['flagged']), 5)
        self.assertEqual({r['column'] for r in y['flagged']}, {'Effort-', 'Defects-'})
        for e in y['flagged']:
            r = projects[e['row'] - 1]
            prediction = adapter.guess.predict(r, adapter.spot.other_projects(r, projects), features, e['column'])
            self.assertAlmostEqual(e['actual'], r[e['column']])
            self.assertAlmostEqual(e['expected'], prediction['estimate'])

    def test_hidden_outcome_changes_do_not_change_input_anomalies(self):
        base, _ = adapter.answers(self.csv)
        def change(rows):
            cols = [rows[0].index(x) for x in ('Kloc+', 'Effort-', 'Defects-')]
            for row in rows[1:]:
                for c in cols:
                    row[c] = '1234567'
        changed, _ = self.altered(change)
        self.assertEqual(base, changed)

    def test_shuffle_recalculates_identifiers(self):
        baseline = adapter.answers(self.csv)
        def change(rows):
            rows[1:] = list(reversed(rows[1:]))
        a, b = self.altered(change)
        self.assertEqual(len(a['flagged']), 5)
        self.assertEqual(len(b['flagged']), 5)
        self.assertNotEqual(a['flagged'], baseline[0]['flagged'])
        with self.csv.open(newline='', encoding='utf-8') as fh:
            table = list(csv.DictReader(fh))
        reversed_rows = list(reversed(table))
        for entry in b['flagged']:
            recorded = reversed_rows[entry['row'] - 1]
            self.assertEqual(entry['actual'], float(recorded[entry['column']]))

    def test_nasa_imports_are_isolated(self):
        for module, name in ((adapter.remember, 'remember'), (adapter.guess, 'guess'), (adapter.spot, 'spot')):
            self.assertEqual(Path(module.__file__), ROOT / f'human_t7/nasa/{name}.py')


if __name__ == '__main__':
    unittest.main()
