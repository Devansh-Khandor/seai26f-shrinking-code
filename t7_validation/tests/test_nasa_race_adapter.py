"""NASA Race adapter contract, fairness, data dependence, and import isolation."""
import csv
import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 't7_validation/adapters/nasa/race.py'
spec = importlib.util.spec_from_file_location('nasa_race_adapter_tests', SRC)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class TestNASARaceAdapter(unittest.TestCase):
    def setUp(self):
        self.csv = ROOT / 'data/raw/nasa93dem.csv'

    def changed(self, modify):
        with self.csv.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.reader(stream))
        modify(rows)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'changed.csv'
            with path.open('w', newline='', encoding='utf-8') as stream:
                csv.writer(stream).writerows(rows)
            return adapter.answers(path)

    def test_prediction_contract_uses_only_history_for_estimates(self):
        prediction, _ = adapter.answers(self.csv)
        rows = adapter.remember.load_projects(self.csv)
        features = adapter.guess.get_features(rows)
        history = [p for p in rows if p['row_number'] % 5]
        test = [p for p in rows if p['row_number'] % 5 == 0]
        self.assertEqual(set(prediction['predictions']), {str(p['row_number']) for p in test})
        for project in test:
            masked = adapter.choose.mask_proposals([project], features)[0]
            expected = adapter.guess.predict(masked, history, features, 'Effort-')['estimate']
            self.assertAlmostEqual(prediction['predictions'][str(project['row_number'])], expected)
        self.assertAlmostEqual(prediction['baseline'],
                               sum(p['Effort-'] for p in history)/len(history))
        self.assertAlmostEqual(prediction['mae_model'], sum(abs(
            prediction['predictions'][str(p['row_number'])] - p['Effort-']) for p in test)/len(test))
        self.assertIn(prediction['verdict'], ('better', 'worse', 'no clear difference'))

    def test_masked_heldout_targets_do_not_change_predictions(self):
        base, _ = adapter.answers(self.csv)
        def edit(rows):
            col = rows[0].index('Effort-')
            for n in range(5, len(rows), 5):
                rows[n][col] = '98765432'
        after, _ = self.changed(edit)
        self.assertEqual(base['predictions'], after['predictions'])
        self.assertEqual(base['baseline'], after['baseline'])

    def test_random_selection_has_same_budget_and_real_winners(self):
        _, response = adapter.answers(self.csv)
        rows = adapter.remember.load_projects(self.csv)
        self.assertEqual(response['budget'], 18)
        self.assertEqual(response['repeats'], adapter.REPEATS)
        self.assertEqual(len(response['ours']), adapter.REPEATS)
        self.assertEqual(len(response['random']), adapter.REPEATS)
        self.assertTrue(all(1 <= x <= len(rows) for x in response['ours'] + response['random']))
        self.assertIn(response['verdict'],
                      ('ours better', 'random better', 'no clear difference'))
        self.assertEqual(response, adapter.selection_answer(rows, adapter.guess.get_features(rows)))

    def test_each_repeat_actually_inspects_same_candidate_pool(self):
        rows = adapter.remember.load_projects(self.csv)
        features = adapter.guess.get_features(rows)
        result = adapter.selection_answer(rows, features)
        budget = result['budget']
        for trial in range(adapter.REPEATS):
            shuffled = list(rows)
            random.Random(adapter.SEED + trial).shuffle(shuffled)
            history, candidates = shuffled[:adapter.HISTORY_SIZE], shuffled[adapter.HISTORY_SIZE:]
            masked = adapter.choose.mask_proposals(candidates, features)
            order = adapter.choose.select_candidates(masked, history, features)
            observed = {p['row_number']: p for p in candidates}
            bounds = adapter.remember.goal_bounds(history)
            random_order = adapter.race.random_rankings(
                [p['row_number'] for p in masked], 1, adapter.SEED + 10000 + trial)[0]
            self.assertEqual(result['ours'][trial],
                             adapter.choose.evaluate_budget(order, observed, bounds, budget)['row_number'])
            self.assertEqual(result['random'][trial],
                             adapter.choose.evaluate_budget(random_order, observed, bounds, budget)['row_number'])
            self.assertGreater(len(candidates), budget)

    def test_honeypot_like_outcomes_do_not_change_predictions_for_masked_candidates(self):
        rows = adapter.remember.load_projects(self.csv)
        features = adapter.guess.get_features(rows)
        history, candidates = rows[:adapter.HISTORY_SIZE], rows[adapter.HISTORY_SIZE:]
        original = adapter.choose.select_candidates(
            adapter.choose.mask_proposals(candidates, features), history, features)
        altered = [{**p, 'Kloc+': 999999.0, 'Effort-': 0., 'Defects-': 0.}
                   for p in candidates]
        changed = adapter.choose.select_candidates(
            adapter.choose.mask_proposals(altered, features), history, features)
        self.assertEqual(original, changed)

    def test_shuffling_changes_record_ids_but_recomputes_answers(self):
        base = adapter.answers(self.csv)
        def edit(rows):
            rows[1:] = list(reversed(rows[1:]))
        after = self.changed(edit)
        self.assertEqual(len(after[0]['predictions']), 18)
        self.assertEqual(after[1]['budget'], 18)
        self.assertNotEqual(base[0]['predictions'], after[0]['predictions'])

    def test_nasa_imports_do_not_clobber_apache(self):
        for mod, name in ((adapter.remember, 'remember'), (adapter.guess, 'guess'),
                          (adapter.choose, 'choose'), (adapter.race, 'race')):
            self.assertEqual(Path(mod.__file__), ROOT / f'human_t7/nasa/{name}.py')


if __name__ == '__main__':
    unittest.main()
