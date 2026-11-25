#!/usr/bin/env python3
"""NASA T7 Flow -> Jack's NASA-07/08 stream contract.

Original Flow's `assess_proposal` and `process_arrivals` are reused unchanged.
Only the common arrival protocol (last round(20%) of rows) and checkpoint
presentation are added. Historical summaries and five-row shortlist use the
original NASA Remember computations. No harness/oracle is imported here.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NASA = ROOT / 'human_t7' / 'nasa'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Both domains use the names `remember` and `guess`. Supply original NASA
# dependencies only for loading NASA source, then restore the import cache.
remember = load('t7_nasa_flow_dep_remember', NASA/'remember.py')
old_remember = sys.modules.get('remember')
old_guess = sys.modules.get('guess')
try:
    sys.modules['remember'] = remember
    guess = load('t7_nasa_flow_dep_guess', NASA/'guess.py')
    sys.modules['guess'] = guess
    flow = load('t7_nasa_original_flow', NASA/'flow.py')
finally:
    for key, old in [('remember', old_remember), ('guess', old_guess)]:
        if old is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = old


def common_arrivals(projects):
    """Common protocol: last int(0.2*n + .5) rows are incoming."""
    k = int(0.2*len(projects) + 0.5)
    if not (0 < k < len(projects)):
        raise ValueError('Need history and arriving projects')
    return projects[:-k], projects[-k:]


def checkpoint(history, arrivals, after):
    """Summaries and top five across exactly the records available at m."""
    observed = history + arrivals[:after]
    s = remember.summarize(observed)
    return {
        'after': after,
        'summary': {goal: {'n': len(observed), 'mean': x['mean'],
                           'min': x['minimum'], 'max': x['maximum']}
                    for goal, x in s.items()},
        'shortlist': [r['row_number'] for r in remember.rank_projects(observed)[:5]],
    }


def answers(csv_path):
    projects = remember.load_projects(csv_path)
    history, arrivals = common_arrivals(projects)
    features = guess.get_features(projects)
    if not features:
        raise ValueError('No predictive features')
    updated, reports = flow.process_arrivals(history, arrivals, features)
    if len(updated) != len(projects):
        raise AssertionError('Streaming must retain all completed projects')
    alerts = [
        {'row': x['row_number'], 'predicted_effort': x['predicted_effort'],
         'predicted_defects': x['predicted_defects'],
         'alert': bool(x['effort_alert'] or x['defect_alert'])}
        for x in reports
    ]
    k = len(arrivals)
    return ({'assessments': alerts},
            {'checkpoints': [checkpoint(history, arrivals, m)
                             for m in (0, k//2, k)]})


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: flow.py PATH_TO_NASA_CSV')
    alerts, summaries = answers(Path(sys.argv[1]))
    print('== NASA-07 ==')
    print('NASA T7 sequential 5-NN prediction; proposals assessed before outcomes arrive')
    print(f"Assessed {len(alerts['assessments'])} incoming proposals")
    print('ANSWER NASA-07 ' + json.dumps(alerts, sort_keys=True))
    print('== NASA-08 ==')
    print('Historical summaries and favorable project shortlists use T7 Remember')
    print(f"Produced {len(summaries['checkpoints'])} sequential historical checkpoints")
    print('ANSWER NASA-08 ' + json.dumps(summaries, sort_keys=True))

if __name__ == '__main__':
    main()
