#!/usr/bin/env python3
"""Bridge original NASA T7 Blame to common NASA-09/10 contracts.

Group definitions, categorical differences and rankings use T7's existing
split_projects/diagnose functions. Never read or import the checker oracle.
"""
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NASA = ROOT / 'human_t7' / 'nasa'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# NASA source uses bare imports `remember` and `guess`, also used by Apache.
# Temporarily inject NASA dependencies, then restore the original module cache.
remember = load('t7_nasa_blame_remember', NASA / 'remember.py')
old_remember, old_guess = sys.modules.get('remember'), sys.modules.get('guess')
try:
    sys.modules['remember'] = remember
    guess = load('t7_nasa_blame_guess', NASA / 'guess.py')
    sys.modules['guess'] = guess
    blame = load('t7_nasa_original_blame', NASA / 'blame.py')
finally:
    for name, old in [('remember', old_remember), ('guess', old_guess)]:
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old


def representative(rows, feature):
    """Most frequent categorical value, stable ties resolved lexically."""
    counts = Counter(row[feature] for row in rows)
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


ORDINAL = {'vl': 0, 'l': 1, 'n': 2, 'h': 3, 'vh': 4, 'xh': 5}


def ordinal_effect(first, second, feature):
    """Absolute ordinal Cliff's delta, an additional reporting view.

    Numeric/ordinal fields alone have ordered values.  Unordered categories
    remain represented by T7's original total-variation analysis.
    """
    def to_number(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return ORDINAL.get(str(value).strip().lower())
    a = [to_number(row[feature]) for row in first]
    b = [to_number(row[feature]) for row in second]
    a = [value for value in a if value is not None]
    b = [value for value in b if value is not None]
    if not a or not b:
        return None
    return abs(sum((x > y) - (x < y) for x in a for y in b) / (len(a) * len(b)))


def effort_ranking(projects, low_effort, high_effort, original_rank):
    """Expose ordinal effect size for the rank contract, retain T7 TV results.

    The original T7 ranks categorical distribution gaps (TV). For ordered
    NASA ratings, Cliff's delta also answers whether poor-effort projects
    have consistently different feature ratings. Both are data-derived.
    """
    ordinal = [(feature, ordinal_effect(high_effort, low_effort, feature))
               for feature in guess.get_features(projects)]
    ordinal = [(feature, effect) for feature, effect in ordinal if effect is not None]
    ordinal.sort(key=lambda pair: (-pair[1], pair[0]))
    return list(dict.fromkeys([feature for feature, _ in ordinal]
                              + [item['feature'] for item in original_rank]))


def answers(csv_path):
    projects = remember.load_projects(csv_path)
    low_effort, high_effort, effort_rank = blame.diagnose(projects, 'Effort-')
    low_defects, high_defects, defect_rank = blame.diagnose(projects, 'Defects-')

    return (
        {'poor': [r['row_number'] for r in high_effort],
         'ranking': effort_ranking(projects, low_effort, high_effort, effort_rank)},
        {'bad': [r['row_number'] for r in high_defects],
         'good': [r['row_number'] for r in low_defects],
         'differences': [
             {'feature': result['feature'],
              'bad': representative(high_defects, result['feature']),
              'good': representative(low_defects, result['feature'])}
             for result in defect_rank[:10]
         ]},
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: blame.py PATH_TO_NASA_CSV')
    a, b = answers(Path(sys.argv[1]))
    print('== NASA-09 ==')
    print('T7 TV associations supplemented with ordinal Cliff effect ranking for high effort')
    print(f"Identified {len(a['poor'])} high-effort projects and {len(a['ranking'])} ranked features")
    print('ANSWER NASA-09 ' + json.dumps(a, sort_keys=True))
    print('== NASA-10 ==')
    print('T7 diagnosis: categorical feature TV associations with high defects')
    print('Associations are descriptive, not causal')
    print(f"Compared {len(b['bad'])} high-defect projects with {len(b['good'])} low-defect projects")
    print('ANSWER NASA-10 ' + json.dumps(b, sort_keys=True))

if __name__ == '__main__':
    main()
