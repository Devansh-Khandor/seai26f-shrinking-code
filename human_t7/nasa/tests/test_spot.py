"""Unit tests of the untouched NASA Spot implementation (NASA-17/18)."""
import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NASA = ROOT / 'human_t7/nasa'


def load(alias, path):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


remember = load('nasa_spot_unit_remember', NASA / 'remember.py')
old = {x: sys.modules.get(x) for x in ('remember', 'guess')}
try:
    sys.modules['remember'] = remember
    guess = load('nasa_spot_unit_guess', NASA / 'guess.py')
    sys.modules['guess'] = guess
    spot = load('nasa_spot_original_unit', NASA / 'spot.py')
finally:
    for x, mod in old.items():
        if mod is None:
            sys.modules.pop(x, None)
        else:
            sys.modules[x] = mod


class TestOriginalNASASpot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.projects = remember.load_projects(ROOT / 'data/raw/nasa93dem.csv')
        cls.features = guess.get_features(cls.projects)

    def test_excludes_same_row_from_neighbor_pool(self):
        row = self.projects[17]
        others = spot.other_projects(row, self.projects)
        self.assertEqual(len(others), len(self.projects) - 1)
        self.assertNotIn(row['row_number'], [r['row_number'] for r in others])

    def test_input_scores_are_mean_neighbor_distances(self):
        ranked = spot.input_anomalies(self.projects, self.features)
        self.assertEqual(len(ranked), len(self.projects))
        self.assertEqual(len({r['row_number'] for r in ranked}), len(self.projects))
        self.assertEqual(ranked, sorted(ranked, key=lambda r: (-r['score'], r['row_number'])))
        for entry in ranked[:4]:
            row = self.projects[entry['row_number'] - 1]
            neighbors = guess.nearest_neighbors(row, spot.other_projects(row, self.projects), self.features, k=spot.K)
            expected = sum(guess.distance(row, n, self.features) for n in neighbors) / len(neighbors)
            self.assertAlmostEqual(entry['score'], expected)
            self.assertEqual(entry['neighbors'], [n['row_number'] for n in neighbors])
            self.assertNotIn(entry['row_number'], entry['neighbors'])

    def test_input_scores_do_not_depend_on_outcome_columns(self):
        mutated = [{**r, 'Effort-': -123456.0, 'Defects-': 999999.0,
                    'Kloc+': -999.0} for r in self.projects]
        self.assertEqual(spot.input_anomalies(self.projects, self.features),
                         spot.input_anomalies(mutated, self.features))

    def test_outcome_residuals_are_leave_one_out_and_correct(self):
        for goal in ('Effort-', 'Defects-'):
            ranked = spot.outcome_anomalies(self.projects, self.features, goal)
            self.assertEqual(len(ranked), len(self.projects))
            self.assertEqual(ranked, sorted(ranked, key=lambda r: (-r['absolute_error'], r['row_number'])))
            for result in ranked[:3]:
                project = self.projects[result['row_number'] - 1]
                expected = guess.predict(project, spot.other_projects(project, self.projects), self.features, goal, k=spot.K)['estimate']
                self.assertAlmostEqual(result['actual'], project[goal])
                self.assertAlmostEqual(result['predicted'], expected)
                self.assertAlmostEqual(result['residual'], result['actual'] - expected)
                self.assertAlmostEqual(result['absolute_error'], abs(result['residual']))

    def test_single_feature_corner_case(self):
        rows = [
            {'row_number': 1, 'x': 'a', 'Effort-': 10.0},
            {'row_number': 2, 'x': 'a', 'Effort-': 20.0},
            {'row_number': 3, 'x': 'b', 'Effort-': 100.0},
        ]
        scores = spot.input_anomalies(rows, ['x'])
        self.assertEqual(scores[0]['row_number'], 3)
        outcomes = spot.outcome_anomalies(rows, ['x'], 'Effort-')
        self.assertEqual(len(outcomes), 3)

    def test_display_and_main_work(self):
        with redirect_stdout(io.StringIO()) as stream:
            spot.main()
        out = stream.getvalue()
        self.assertIn('NASA-17:', out)
        self.assertIn('NASA-18:', out)
        self.assertIn('Effort-', out)
        self.assertIn('Defects-', out)


if __name__ == '__main__':
    unittest.main()
