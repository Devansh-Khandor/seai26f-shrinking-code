"""Unit tests of the original, unchanged NASA T7 Blame implementation."""
from collections import Counter
import importlib.util
import math
import sys
import unittest
from pathlib import Path

NASA = Path(__file__).resolve().parents[1]


def load(alias, path):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


remember = load('test_nasa_blame_original_remember', NASA / 'remember.py')
old_remember, old_guess = sys.modules.get('remember'), sys.modules.get('guess')
try:
    sys.modules['remember'] = remember
    guess = load('test_nasa_blame_original_guess', NASA / 'guess.py')
    sys.modules['guess'] = guess
    blame = load('test_nasa_blame_original_module', NASA / 'blame.py')
finally:
    for key, old in [('remember', old_remember), ('guess', old_guess)]:
        if old is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = old


class TestOriginalNASABlame(unittest.TestCase):
    def test_split_selects_quartiles_by_outcome(self):
        rows = [{'row_number': i + 1, 'Effort-': float(i)} for i in range(12)]
        low, high = blame.split_projects(rows, 'Effort-')
        self.assertEqual([r['row_number'] for r in low], [1, 2, 3])
        self.assertEqual([r['row_number'] for r in high], [10, 11, 12])

    def test_split_breaks_ties_by_row_identifier(self):
        rows = [{'row_number': i, 'Effort-': 3.} for i in (4, 1, 3, 2)]
        low, high = blame.split_projects(rows, 'Effort-')
        self.assertEqual(low[0]['row_number'], 1)
        self.assertEqual(high[0]['row_number'], 4)

    def test_frequencies_are_relative(self):
        freq = blame.frequencies([{'x': 'h'}, {'x': 'h'}, {'x': 'l'}], 'x')
        self.assertEqual(freq, {'h': 2 / 3, 'l': 1 / 3})
        self.assertAlmostEqual(sum(freq.values()), 1.)

    def test_total_variation_known_cases(self):
        self.assertEqual(blame.total_variation({'a': 1.}, {'a': 1.}), 0.)
        self.assertEqual(blame.total_variation({'a': 1.}, {'b': 1.}), 1.)
        self.assertAlmostEqual(blame.total_variation({'a': .8, 'b': .2}, {'a': .2, 'b': .8}), .6)

    def test_real_effort_and_defect_diagnoses(self):
        rows = remember.load_projects()
        features = set(guess.get_features(rows))
        for goal in ('Effort-', 'Defects-'):
            low, high, ranking = blame.diagnose(rows, goal)
            self.assertEqual(len(low), 23)
            self.assertEqual(len(high), 23)
            self.assertFalse({r['row_number'] for r in low} & {r['row_number'] for r in high})
            self.assertEqual({r['feature'] for r in ranking}, features)
            self.assertTrue(all(0 <= item['tv'] <= 1 for item in ranking))
            self.assertEqual(ranking, sorted(ranking, key=lambda item: (-item['tv'], item['feature'])))

    def test_diagnosis_recalculates_when_outcomes_change(self):
        rows = remember.load_projects()
        before = blame.diagnose(rows, 'Effort-')
        changed = [r.copy() for r in rows]
        for row in changed:
            row['Effort-'] = 10000 - row['Effort-']
        after = blame.diagnose(changed, 'Effort-')
        self.assertNotEqual([r['row_number'] for r in before[1]],
                            [r['row_number'] for r in after[1]])


if __name__ == '__main__':
    unittest.main()
