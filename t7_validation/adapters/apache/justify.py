#!/usr/bin/env python3
"""Bridge Apache T7 Justify to the shared APACHE-11/12 checks.

No reference oracle or checker is imported. Read the supplied dataset on
*every* invocation; row identifiers follow current CSV row order.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'human_t7' / 'apache'))
from remember import GOAL, adjustable_settings, load_configurations  # noqa: E402
from guess import one_setting_pairs  # noqa: E402
from justify import compare_configurations, recommend_configuration, validate_explanation  # noqa: E402


def answers(csv_path):
    rows = load_configurations(csv_path)
    settings = adjustable_settings(rows)
    # Shared test protocol: every fifth row is held out for recommendations.
    history = [row for row in rows if row['row_number'] % 5 != 0]
    candidates = [row for row in rows if row['row_number'] % 5 == 0]
    if len(history) < 5 or not candidates:
        raise ValueError('Need at least five historical rows and held-out candidates')

    selected, explanation = recommend_configuration(candidates, history, settings)
    if not validate_explanation(explanation, history, settings):
        raise AssertionError('T7 recommendation evidence did not validate')
    first = {
        'recommended': selected['row_number'],
        'neighbors': [
            {'row': item['row_number'], 'values': {GOAL: item['performance']}}
            for item in explanation['evidence']
        ],
    }

    # APACHE-12 permits recorded outcomes. Compare an observed valid one-setting
    # pair from the history; unlike held-out prediction, its labels are known.
    pairs = one_setting_pairs(history, settings)
    pair = next(((a, b) for a, b, _ in pairs if a[GOAL] != b[GOAL]), None)
    if pair is None:
        raise ValueError('No distinct-performance one-setting historical pair')
    a, b = pair
    comparison = compare_configurations(a, b, history, settings)
    # Explain through existing T7 comparisons but use *recorded* outcomes for
    # preference, as explicitly allowed by the customer requirement.
    preferred = 'a' if a[GOAL] < b[GOAL] else 'b'
    preferred_row, other_row = (a, b) if preferred == 'a' else (b, a)
    second = {
        'a': a['row_number'],
        'b': b['row_number'],
        'preferred': preferred,
        'scores': {str(a['row_number']): -a[GOAL], str(b['row_number']): -b[GOAL]},
        'differences': [
            {'feature': item['setting'], 'a': item['a_value'], 'b': item['b_value']}
            for item in comparison['differences']
        ],
        'tradeoffs': [
            GOAL for goal in (GOAL,)
            if other_row[goal] < preferred_row[goal]
        ],
    }
    return first, second


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: justify.py PATH_TO_APACHE_CSV')
    first, second = answers(Path(sys.argv[1]))
    print('== APACHE-11 ==')
    print('Recommend by kNN and cite actual neighboring historical configurations')
    print(f"Recommended row: {first['recommended']}")
    print('ANSWER APACHE-11 ' + json.dumps(first, sort_keys=True))
    print('== APACHE-12 ==')
    print('Explain one-setting tradeoff between observed historical configurations')
    print(f"Compared rows: {second['a']} and {second['b']}")
    print('ANSWER APACHE-12 ' + json.dumps(second, sort_keys=True))


if __name__ == '__main__':
    main()
