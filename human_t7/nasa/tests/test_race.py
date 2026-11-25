"""Unit tests for the untouched NASA Race implementation (NASA-19/20)."""
import importlib.util
import io
import statistics
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NASA = ROOT / 'human_t7/nasa'


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


remember = load('nasa_race_unit_remember', NASA / 'remember.py')
prev = {name: sys.modules.get(name) for name in ('remember', 'guess', 'choose')}
try:
    sys.modules['remember'] = remember
    guess = load('nasa_race_unit_guess', NASA / 'guess.py')
    sys.modules['guess'] = guess
    choose = load('nasa_race_unit_choose', NASA / 'choose.py')
    sys.modules['choose'] = choose
    race = load('nasa_race_original_unit', NASA / 'race.py')
finally:
    for name, old in prev.items():
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old


class TestOriginalNASARace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = remember.load_projects(ROOT / 'data/raw/nasa93dem.csv')
        cls.history, cls.test = cls.rows[:74], cls.rows[74:]
        cls.features = guess.get_features(cls.rows)

    def test_random_orders_deterministic_and_complete(self):
        ids = list(range(1, 20))
        a = race.random_rankings(ids, trials=30, seed=42)
        self.assertEqual(a, race.random_rankings(ids, trials=30, seed=42))
        self.assertEqual(len(a), 30)
        self.assertTrue(all(sorted(e['row_number'] for e in order) == ids for order in a))
        self.assertGreater(len({tuple(e['row_number'] for e in o) for o in a}), 1)

    def test_random_orders_only_identifiers(self):
        rankings = race.random_rankings([101, 102, 103], trials=2)
        for ranking in rankings:
            self.assertEqual({tuple(e.keys()) for e in ranking}, {('row_number',)})

    def test_prediction_race_uses_original_guess_evaluation(self):
        with redirect_stdout(io.StringIO()) as stream:
            race.prediction_race(self.history, self.test, self.features)
        text = stream.getvalue()
        for goal in ('Effort-', 'Defects-'):
            maem, maeb = guess.evaluate(self.history, self.test, self.features, goal)
            self.assertIn(f'{maem:>12.2f}', text)
            self.assertIn(f'{maeb:>12.2f}', text)
            self.assertIn(f'{goal}: kNN lower error on', text)

    def test_guided_vs_random_share_pool_and_budgets(self):
        with redirect_stdout(io.StringIO()):
            plan, random_orders, results = race.selection_race(self.history, self.test, self.features)
        self.assertEqual(len(random_orders), race.RANDOM_TRIALS)
        self.assertEqual([e['budget'] for e in results], list(choose.BUDGETS))
        observed = {p['row_number']: p for p in self.test}
        bounds = remember.goal_bounds(self.history)
        for entry in results:
            b = entry['budget']
            best = choose.evaluate_budget(plan, observed, bounds, b)
            self.assertEqual(entry['model_row'], best['row_number'])
            self.assertAlmostEqual(entry['model_d2h'], best['d2h'])
            expected_random = [choose.evaluate_budget(order, observed, bounds, b)['d2h']
                               for order in random_orders]
            self.assertEqual(entry['random_scores'], expected_random)
            self.assertAlmostEqual(entry['random_mean'], statistics.mean(expected_random))

    def test_selection_plan_does_not_read_candidate_goals(self):
        base = choose.select_candidates(choose.mask_proposals(self.test, self.features),
                                        self.history, self.features)
        changed = [{**p, 'Kloc+': 9e15, 'Effort-': -8e15, 'Defects-': 7e15}
                   for p in self.test]
        updated = choose.select_candidates(choose.mask_proposals(changed, self.features),
                                           self.history, self.features)
        self.assertEqual(base, updated)

    def test_main_runs_and_displays_both_requirements(self):
        with redirect_stdout(io.StringIO()) as stream:
            race.main()
        text = stream.getvalue()
        self.assertIn('NASA-19:', text)
        self.assertIn('NASA-20:', text)
        self.assertIn('Random trials: 30', text)


if __name__ == '__main__':
    unittest.main()
