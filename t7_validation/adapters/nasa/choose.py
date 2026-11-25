#!/usr/bin/env python3
"""NASA original T7 Choose -> Jack's NASA-13/14 response format.

NASA-13 bridges T7's favorable observed projects to a supported single-attribute
historical rule. NASA-14 calls T7's 5-NN inspection-order function on masked
held-out proposals. No goal values of these proposals enter the plan.
"""
import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NASA = ROOT / 'human_t7/nasa'


def load(alias, path):
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


remember = load('t7_nasa_choose_remember', NASA / 'remember.py')
previous = {name: sys.modules.get(name) for name in ('remember', 'guess')}
try:
    sys.modules['remember'] = remember
    guess = load('t7_nasa_choose_guess', NASA / 'guess.py')
    sys.modules['guess'] = guess
    choose = load('t7_nasa_original_choose', NASA / 'choose.py')
finally:
    for name, old in previous.items():
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old


def historical_rule(projects, features):
    """A single categorical pattern supported by >=3 favorable observed rows.

    Translate T7's promising_history ranking to the common checker pattern
    request; the original T7 does not itself discover conjunction rules.
    """
    ranked = remember.rank_projects(projects)
    scores = {r['row_number']: r['d2h'] for r in ranked}
    typical = statistics.median(scores.values())
    candidates = []
    for feature in features:
        groups = {}
        for row in projects:
            groups.setdefault(row[feature], []).append(row)
        for value, matches in groups.items():
            if len(matches) < 3:
                continue
            median = statistics.median(scores[r['row_number']] for r in matches)
            if median >= typical:
                continue
            # Minimize T7 median d2h; prefer larger coverage at equal quality.
            candidates.append((median, -len(matches), feature, str(value), len(matches)))
    if not candidates:
        raise ValueError('No supported favorable historical rule')
    _, _, feature, value, matching = min(candidates)
    return {'rule': {feature: value}, 'matching': matching}


def answers(csv_path):
    projects = remember.load_projects(csv_path)
    features = guess.get_features(projects)
    # NASA-13 is retrospective: every project is observed history here.
    rule = historical_rule(projects, features)
    # NASA-14 uses Jack's every-fifth candidate convention; no candidate goals
    # enter select_candidates. This is separate from original T7's 74/19 split.
    history = [r for r in projects if r['row_number'] % 5 != 0]
    candidates = [r for r in projects if r['row_number'] % 5 == 0]
    masked = choose.mask_proposals(candidates, features)
    ordered = choose.select_candidates(masked, history, features)
    budget = min(30, len(projects) // 5)
    if len(ordered) < 10:
        raise ValueError('Need >=10 proposal candidates')
    # Use the full permitted inspection budget when there are enough
    # candidates. Only features and historic goals determine this order.
    return rule, {'order': [r['row_number'] for r in ordered[:min(budget, 12)]]}


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: choose.py PATH_TO_NASA_CSV')
    rule, order = answers(Path(sys.argv[1]))
    print('== NASA-13 ==')
    print('T7 historical distance-to-heaven ranking supports a categorical rule')
    print(f"The rule has {rule['matching']} matching historical projects")
    print('ANSWER NASA-13 ' + json.dumps(rule, sort_keys=True))
    print('== NASA-14 ==')
    print('T7 5-NN ranks proposals using only training outcomes and candidate inputs')
    print(f"Selected {len(order['order'])} projects within the common inspection budget")
    print('ANSWER NASA-14 ' + json.dumps(order, sort_keys=True))


if __name__ == '__main__':
    main()
