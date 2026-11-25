#!/usr/bin/env python3
"""NASA original T7 Race -> NASA-19/20 common ANSWER contract.

NASA-19: original 5-NN on every fifth held-out row, historical-mean baseline.
NASA-20: independent seeded partitions (60 historical, remaining candidates),
         original Choose 5-NN ranking and original Race seeded random orders,
         common budget floor(n/5) with 30 repeats. Outcomes only revealed
         AFTER ranking within each repeat. This is NOT the original T7's
         fixed 74/19 split with budgets 3, 5, and 10.
"""
import importlib.util
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NASA = ROOT / 'human_t7/nasa'
REPEATS = 30
SEED = 42
HISTORY_SIZE = 60


def load(alias, path):
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


remember = load('t7_nasa_race_remember', NASA / 'remember.py')
previous = {name: sys.modules.get(name) for name in ('remember', 'guess', 'choose')}
try:
    sys.modules['remember'] = remember
    guess = load('t7_nasa_race_guess', NASA / 'guess.py')
    sys.modules['guess'] = guess
    choose = load('t7_nasa_race_choose', NASA / 'choose.py')
    sys.modules['choose'] = choose
    race = load('t7_nasa_original_race', NASA / 'race.py')
finally:
    for name, mod in previous.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def _clear_difference(xs, ys):
    """Distributional assessment independent of Jack's oracle.

    Return 'same', 'different', or None, using a magnitude and two-sample KS
    screen. This supplies an honest conservative label for repeated trials.
    """
    gt = sum(x > y for x in xs for y in ys)
    lt = sum(x < y for x in xs for y in ys)
    delta = abs((gt - lt) / (len(xs) * len(ys)))
    positions = sorted(set(xs) | set(ys))
    d = max(abs(sum(x <= v for x in xs) / len(xs) -
                sum(y <= v for y in ys) / len(ys)) for v in positions)
    ks = d > 1.36 * math.sqrt((len(xs) + len(ys)) / (len(xs) * len(ys)))
    if delta >= 0.474 and ks:
        return 'different'
    if delta < 0.147 and not ks:
        return 'same'
    return None


def prediction_answer(projects, features):
    history = [p for p in projects if p['row_number'] % 5]
    heldout = [p for p in projects if p['row_number'] % 5 == 0]
    baseline = statistics.mean(p['Effort-'] for p in history)
    predicted = {
        str(p['row_number']): guess.predict(
            choose.mask_proposals([p], features)[0], history, features, 'Effort-'
        )['estimate']
        for p in heldout
    }
    model_errors = [abs(predicted[str(p['row_number'])] - p['Effort-']) for p in heldout]
    mean_errors = [abs(baseline - p['Effort-']) for p in heldout]
    clear = _clear_difference(model_errors, mean_errors)
    if clear == 'same' or clear is None:
        verdict = 'no clear difference'
    else:
        verdict = 'better' if statistics.median(model_errors) < statistics.median(mean_errors) else 'worse'
    return {
        'predictions': predicted,
        'baseline': baseline,
        'mae_model': statistics.mean(model_errors),
        'mae_baseline': statistics.mean(mean_errors),
        'verdict': verdict,
    }


def selection_answer(projects, features, repeats=REPEATS, seed=SEED):
    """Repeated independent train/candidate partitions and equal-budget arms.

    Original T7 5-NN ranks masked candidate inputs. Original T7 random_rankings
    generates random candidate orders. Only after ranking do we score the
    best of the first budget rows in each arm.
    """
    budget = min(30, len(projects) // 5)
    if len(projects) - HISTORY_SIZE <= budget:
        raise ValueError('Need more candidates than inspection budget')
    ours, random_rows = [], []
    for trial in range(repeats):
        shuffled = list(projects)
        random.Random(seed + trial).shuffle(shuffled)
        history, candidates = shuffled[:HISTORY_SIZE], shuffled[HISTORY_SIZE:]
        masked = choose.mask_proposals(candidates, features)
        plan = choose.select_candidates(masked, history, features)
        observed = {p['row_number']: p for p in candidates}
        bounds = remember.goal_bounds(history)
        random_plan = race.random_rankings(
            [p['row_number'] for p in masked], trials=1, seed=seed + 10_000 + trial
        )[0]
        ours.append(choose.evaluate_budget(plan, observed, bounds, budget)['row_number'])
        random_rows.append(choose.evaluate_budget(random_plan, observed, bounds, budget)['row_number'])

    # Derive verdict from recorded goals using a common whole-dataset
    # distance-to-heaven reference, not predictions, to compare winners.
    bounds_all = remember.goal_bounds(projects)
    recorded = {p['row_number']: remember.distance_to_heaven(p, bounds_all) for p in projects}
    a = [recorded[r] for r in ours]
    b = [recorded[r] for r in random_rows]
    clear = _clear_difference(a, b)
    if clear == 'same' or clear is None:
        verdict = 'no clear difference'
    else:
        verdict = 'ours better' if statistics.median(a) < statistics.median(b) else 'random better'
    return {'budget': budget, 'repeats': repeats, 'ours': ours, 'random': random_rows, 'verdict': verdict}


def answers(csv_path):
    projects = remember.load_projects(csv_path)
    features = guess.get_features(projects)
    return prediction_answer(projects, features), selection_answer(projects, features)


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: race.py PATH_TO_NASA_CSV')
    prediction, selection = answers(Path(sys.argv[1]))
    print('== NASA-19 ==')
    print('Original NASA 5-NN predictions versus training-mean effort baseline')
    print('Predictions fit on non-held-out projects only')
    print('ANSWER NASA-19 ' + json.dumps(prediction, sort_keys=True))
    print('== NASA-20 ==')
    print('Original NASA 5-NN selection versus seeded random orders')
    print(f"{selection['repeats']} independent splits; equal budget of {selection['budget']} inspections")
    print('Outcomes revealed only after selecting projects')
    print('ANSWER NASA-20 ' + json.dumps(selection, sort_keys=True))


if __name__ == '__main__':
    main()
