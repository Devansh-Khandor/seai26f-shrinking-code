"""Unit tests of the unchanged NASA Step 8 Fix implementation."""
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


remember = load('t7_nasa_fix_unit_remember', NASA / 'remember.py')
previous = {name: sys.modules.get(name) for name in ('remember', 'guess')}
try:
    sys.modules['remember'] = remember
    guess = load('t7_nasa_fix_unit_guess', NASA / 'guess.py')
    sys.modules['guess'] = guess
    fix = load('t7_nasa_fix_unit_fix', NASA / 'fix.py')
finally:
    for name, old in previous.items():
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old


class TestOriginalNASAFix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = remember.load_projects(ROOT / 'data/raw/nasa93dem.csv')
        cls.features = guess.get_features(cls.rows)
        cls.history = cls.rows[:74]
        cls.proposals = cls.rows[74:]

    def test_adjustable_fields_and_historical_values(self):
        self.assertEqual(fix.ADJUSTABLE, ('docu', 'tool', 'site', 'sced'))
        values = fix.possible_values(self.history)
        self.assertEqual(set(values), set(fix.ADJUSTABLE))
        for feature, choices in values.items():
            self.assertEqual(choices, sorted({r[feature] for r in self.history}))

    def test_generated_candidates_change_only_one_or_two_settings(self):
        proposal = self.proposals[0]
        values = fix.possible_values(self.history)
        seen = 0
        for candidate, changes in fix.candidate_changes(proposal, values):
            seen += 1
            self.assertIn(len(changes), (1, 2))
            self.assertEqual(set(changes), {k for k in fix.ADJUSTABLE if candidate[k] != proposal[k]})
            for k, (old, new) in changes.items():
                self.assertEqual(old, proposal[k])
                self.assertNotEqual(old, new)
                self.assertIn(new, values[k])
            for k in self.features:
                if k not in changes:
                    self.assertEqual(proposal[k], candidate[k])
        self.assertGreater(seen, 0)

    def test_predictions_use_existing_knn(self):
        proposal = self.proposals[0]
        got = fix.predict_outcomes(proposal, self.history, self.features)
        for goal in ('Effort-', 'Defects-'):
            self.assertEqual(got[goal], guess.predict(proposal, self.history, self.features, goal)['estimate'])

    def test_repairs_minimize_number_of_changed_settings(self):
        proposal = self.proposals[5]
        baseline, repair, tested = fix.find_repair(proposal, self.history, self.features, 'Effort-')
        self.assertGreater(tested, 0)
        self.assertIsNotNone(repair)
        self.assertIn(len(repair['changes']), (1, 2))
        self.assertGreaterEqual(repair['improvement'], fix.MIN_IMPROVEMENT)
        self.assertLess(repair['predicted']['Effort-'], baseline['Effort-'])
        # Independently enumerate qualifying candidates and check minimum count.
        qualifying = []
        for candidate, changes in fix.candidate_changes(proposal, fix.possible_values(self.history)):
            pred = fix.predict_outcomes(candidate, self.history, self.features)
            if (baseline['Effort-'] - pred['Effort-']) / baseline['Effort-'] >= fix.MIN_IMPROVEMENT:
                qualifying.append(len(changes))
        self.assertEqual(len(repair['changes']), min(qualifying))

    def test_defect_repairs_respect_effort_constraint(self):
        proposal = self.proposals[5]
        baseline, repair, _ = fix.find_repair(
            proposal, self.history, self.features, 'Defects-', fix.MAX_EFFORT_INCREASE)
        self.assertIsNotNone(repair)
        self.assertLess(repair['predicted']['Defects-'], baseline['Defects-'])
        self.assertLessEqual(repair['predicted']['Effort-'], baseline['Effort-'] * (1 + fix.MAX_EFFORT_INCREASE))

    def test_hidden_candidate_outcomes_do_not_change_risk_selection(self):
        masked = [{k: v for k, v in r.items() if k not in remember.GOALS} for r in self.proposals]
        changed = [{**r, 'Effort-': 1e15, 'Defects-': -1e15, 'Kloc+': 1e15} for r in self.proposals]
        for goal in ('Effort-', 'Defects-'):
            a = fix.highest_risk_proposal(masked, self.history, self.features, goal)
            b = fix.highest_risk_proposal(changed, self.history, self.features, goal)
            self.assertEqual(a['row_number'], b['row_number'])

    def test_original_demonstration_runs(self):
        with redirect_stdout(io.StringIO()) as out:
            fix.main()
        self.assertIn('NASA-15:', out.getvalue())
        self.assertIn('NASA-16:', out.getvalue())


if __name__ == '__main__':
    unittest.main()
