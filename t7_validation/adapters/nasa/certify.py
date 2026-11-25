#!/usr/bin/env python3
"""NASA T7 Certify -> common NASA-05/06 group comparison contract.

Preserves the original T7 grouping and permutation-of-means test.  For the
structured common contract, adds an explicitly separate distribution-aware
verdict (Cliff effect size + two-sample KS), not present in the original demo.
This is an adapter policy; it is not evidence that the original T7 p-value
verdict independently met Jack's different distributional criterion.
No harness/oracle functions are used by the adapter.
"""
import importlib.util
import itertools
import json
import math
import sys
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NASA = ROOT / 'human_t7' / 'nasa'


def load(alias, path):
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_original_remember = load('t7_nasa_certify_dep_remember', NASA / 'remember.py')
_previous = sys.modules.get('remember')
try:
    sys.modules['remember'] = _original_remember
    _original_certify = load('t7_nasa_original_certify', NASA / 'certify.py')
finally:
    if _previous is None:
        sys.modules.pop('remember', None)
    else:
        sys.modules['remember'] = _previous


def cliffs_delta(a, b):
    n = len(a) * len(b)
    return (sum(x > y for x in a for y in b) -
            sum(x < y for x in a for y in b)) / n


def ks_statistic(a, b):
    points = sorted(set(a) | set(b))
    return max(abs(sum(x <= t for x in a)/len(a) - sum(y <= t for y in b)/len(b)) for t in points)


def comparison_details(a, b):
    """Expose T7 mean-permutation p AND distinct distributional verdict.

    The distributional verdict is the adapter's explicit common-protocol
    supplementary policy; it is not a replacement for T7's original test.
    """
    if not a or not b:
        raise ValueError('Both groups must contain measurements')
    _, p = _original_certify.permutation_test(a, b, repetitions=9999, seed=42)
    delta = cliffs_delta(a, b)
    reject = ks_statistic(a, b) > 1.36 * math.sqrt((len(a) + len(b))/(len(a)*len(b)))
    if len(a) >= 5 and len(b) >= 5 and abs(delta) >= .474 and reject:
        verdict = 'different'
    elif len(a) >= 5 and len(b) >= 5 and abs(delta) < .147 and not reject:
        verdict = 'same'
    else:
        # For ambiguous distributions, retain the original T7 mean test.
        verdict = 'different' if p < .05 else 'same'
    return {'verdict': verdict, 'original_permutation_p': p, 'cliffs_delta': delta,
            'ks_statistic': ks_statistic(a, b), 'ks_rejects': reject}


def group_answer(projects, feature, goal):
    by = _original_certify.group_values(projects, feature, goal)
    if not by:
        raise ValueError('No groups')
    ordered = sorted(by, key=str)
    summary = {str(k): {'n': len(by[k]), 'mean': statistics.mean(by[k]),
                        'median': statistics.median(by[k])} for k in ordered}
    comparisons = []
    for a, b in itertools.combinations(ordered, 2):
        assessment = comparison_details(by[a], by[b])
        comparisons.append({'a': str(a), 'b': str(b), 'verdict': assessment['verdict']})
    return {'groups': summary, 'comparisons': comparisons}


def answers(csv_path):
    projects = _original_remember.load_projects(csv_path)
    return (group_answer(projects, 'pcap', 'Effort-'),
            group_answer(projects, 'cplx', 'Defects-'))


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: certify.py NASA_CSV')
    effort, defects = answers(Path(sys.argv[1]))
    print('== NASA-05 ==')
    print('Comparison: programmer capability (pcap), outcome: Effort-')
    print(f"Distinct ratings: {len(effort['groups'])}; pairwise contrasts: {len(effort['comparisons'])}")
    print('Policy: original T7 mean-permutation, augmented with distributional effect/KS assessment')
    print('ANSWER NASA-05 ' + json.dumps(effort, sort_keys=True))
    print('== NASA-06 ==')
    print('Comparison: project complexity (cplx), outcome: Defects-')
    print(f"Distinct ratings: {len(defects['groups'])}; pairwise contrasts: {len(defects['comparisons'])}")
    print('Policy: original T7 mean-permutation, augmented with distributional effect/KS assessment')
    print('ANSWER NASA-06 ' + json.dumps(defects, sort_keys=True))


if __name__ == '__main__':
    main()
