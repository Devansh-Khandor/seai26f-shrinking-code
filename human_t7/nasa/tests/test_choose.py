"""Unit tests for the unchanged original NASA Choose implementation."""
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
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


remember = load('t7_nasa_choose_unit_remember', NASA / 'remember.py')
previous = {name: sys.modules.get(name) for name in ('remember', 'guess')}
try:
    sys.modules['remember'] = remember
    guess = load('t7_nasa_choose_unit_guess', NASA / 'guess.py')
    sys.modules['guess'] = guess
    choose = load('t7_nasa_choose_unit_choose', NASA / 'choose.py')
finally:
    for name, old in previous.items():
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old


class TestOriginalNASAChoose(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = remember.load_projects(ROOT / 'data/raw/nasa93dem.csv')
        cls.features = guess.get_features(cls.rows)
        cls.history = cls.rows[:74]
        cls.candidates = cls.rows[74:]

    def test_mask_proposals_excludes_every_goal(self):
        masked = choose.mask_proposals(self.candidates, self.features)
        self.assertEqual(len(masked), 19)
        for original, proposal in zip(self.candidates, masked):
            self.assertEqual(proposal['row_number'], original['row_number'])
            self.assertTrue(all(proposal[f] == original[f] for f in self.features))
            self.assertTrue(all(g not in proposal for g in remember.GOALS))

    def test_original_selection_is_repeatable_and_input_only(self):
        masked = choose.mask_proposals(self.candidates, self.features)
        ranked = choose.select_candidates(masked, self.history, self.features)
        self.assertEqual(len(ranked), len(masked))
        self.assertEqual(len({r['row_number'] for r in ranked}), len(masked))
        self.assertEqual(ranked, choose.select_candidates(masked, self.history, self.features))
        self.assertEqual([r['predicted_d2h'] for r in ranked],
                         sorted(r['predicted_d2h'] for r in ranked))

    def test_hidden_candidate_targets_do_not_change_inspection_plan(self):
        baseline = choose.select_candidates(self.candidates, self.history, self.features)
        changed = [{**r, 'Effort-': 1e12, 'Defects-': -1e12, 'Kloc+': 1e12}
                   for r in self.candidates]
        self.assertEqual(baseline, choose.select_candidates(changed, self.history, self.features))

    def test_budget_best_reveals_only_selected_goals(self):
        ranked = choose.select_candidates(choose.mask_proposals(self.candidates, self.features),
                                          self.history, self.features)
        observed = {r['row_number']: r for r in self.candidates}
        bounds = remember.goal_bounds(self.history)
        for b in choose.BUDGETS:
            result = choose.evaluate_budget(ranked, observed, bounds, b)
            expected = min(ranked[:b], key=lambda r: remember.distance_to_heaven(
                observed[r['row_number']], bounds))
            self.assertEqual(result['row_number'], expected['row_number'])
            self.assertEqual(result['d2h'], remember.distance_to_heaven(
                observed[expected['row_number']], bounds))

    def test_promising_history_returns_actual_top_five(self):
        with redirect_stdout(io.StringIO()):
            got = choose.promising_history(self.history, self.features, count=5)
        self.assertEqual([r['row_number'] for r in got],
                         [r['row_number'] for r in remember.rank_projects(self.history)[:5]])

    def test_original_demo_runs(self):
        with redirect_stdout(io.StringIO()):
            choose.main()


if __name__ == '__main__':
    unittest.main()
