#!/usr/bin/env python3
"""NASA Spot common-check bridge, reusing the original T7 anomaly algorithms.

NASA-17 uses original input_anomalies, a mean 5-NN categorical mismatch
score. NASA-18 uses original leave-one-out predicted residuals for effort
and defects. Scores and flagged records are recomputed from the supplied CSV.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NASA = ROOT / 'human_t7/nasa'


def load(alias, file):
    spec = importlib.util.spec_from_file_location(alias, file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


remember = load('t7_nasa_spot_remember', NASA / 'remember.py')
prior = {name: sys.modules.get(name) for name in ('remember', 'guess')}
try:
    sys.modules['remember'] = remember
    guess = load('t7_nasa_spot_guess', NASA / 'guess.py')
    sys.modules['guess'] = guess
    spot = load('t7_nasa_original_spot', NASA / 'spot.py')
finally:
    for name, old in prior.items():
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old


def answers(csv_path):
    projects = remember.load_projects(csv_path)
    features = guess.get_features(projects)
    top_n = max(1, min(5, len(projects) // 10 + 1))

    inputs = spot.input_anomalies(projects, features)[:top_n]
    x = {
        'flagged': [r['row_number'] for r in inputs],
        'scores': {str(r['row_number']): r['score'] for r in inputs},
    }

    # Original T7 orders residuals by absolute observed-minus-predicted
    # error within each target. Include both targets for an interpretable
    # screening list, rather than ranking incompatible raw units together.
    effort = spot.outcome_anomalies(projects, features, 'Effort-')
    defects = spot.outcome_anomalies(projects, features, 'Defects-')
    selection = [(r, 'Effort-') for r in effort[:(top_n + 1)//2]]
    selection += [(r, 'Defects-') for r in defects[:top_n//2]]
    flagged = [
        {'row': r['row_number'], 'column': goal,
         'actual': r['actual'], 'expected': r['predicted']}
        for r, goal in selection
    ]
    return x, {'flagged': flagged}


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: spot.py PATH_TO_NASA_CSV')
    x, y = answers(Path(sys.argv[1]))
    print('== NASA-17 ==')
    print('Original T7: most isolated inputs using five nearest historical neighbors')
    print('Input-only anomaly scores use the twenty-two project characteristics, not goals')
    print('ANSWER NASA-17 ' + json.dumps(x, sort_keys=True))
    print('== NASA-18 ==')
    print('Original T7: leave-one-out residuals using five nearest historical neighbors')
    print('Reported actual outcomes are read from the supplied CSV after prediction')
    print('ANSWER NASA-18 ' + json.dumps(y, sort_keys=True))


if __name__ == '__main__':
    main()
