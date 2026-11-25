#!/usr/bin/env python3
"""APACHE-15/16 common-check bridge using T7's repair enumeration.

IMPORTANT: APACHE-16's common contract optimizes *recorded* proximity to
its target, while original T7 find_minimum_repair() optimizes *predicted*
threshold satisfaction. The adapter explicitly uses the existing T7
valid_alternatives() to enumerate observed valid options, then chooses the
minimum-edit option whose *recorded* performance is nearer the target.
This is a documented evaluation-policy difference, NOT a demonstration that
T7's original find_minimum_repair passes that exact reference criterion.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'human_t7' / 'apache'))
from remember import GOAL, adjustable_settings, load_configurations  # noqa: E402
from fix import DEFAULT_TARGET, recommend_small_repair, valid_alternatives  # noqa: E402


def answers(csv_path):
    records = load_configurations(csv_path)
    settings = adjustable_settings(records)
    by_id = {r['row_number']: r for r in records}
    # The customer provides a previously benchmarked, poorly performing source.
    # Select the worst observed record without assuming a fixed CSV row id.
    source = max(records, key=lambda r: (r[GOAL], -r['row_number']))
    median = statistics.median(r[GOAL] for r in records)
    if source[GOAL] <= median:
        raise ValueError('No poorly performing source record')

    # Use the complete observed history for this historical repair task.
    # T7's independent-neighbor estimator excludes each candidate itself.
    repair = recommend_small_repair(source, records, records, settings)
    options = [
        item for item in valid_alternatives(source, records, records, settings)[1]
        if 1 <= item['changes'] <= 2
        and by_id[item['row_number']][GOAL] < source[GOAL]
        and item['predicted_improvement'] > 0
    ]
    # T7's first recommended repair should be used if it is also supported by
    # recorded historical data; otherwise find the earliest measured-supported
    # option from T7's predicted-improvement candidates.
    candidate = repair['recommendation']
    if candidate not in options:
        options.sort(key=lambda item: (
            item['changes'], item['prediction'], item['row_number']))
        candidate = options[0] if options else None
    if candidate is None:
        raise ValueError('No small predicted and observed repair available')

    alt = by_id[candidate['row_number']]
    def change_map(item):
        return {part['setting']: [part['from'], part['to']]
                for part in item['setting_changes']}

    first = {
        'row': source['row_number'],
        'alternative': alt['row_number'],
        'changes': change_map(candidate),
        # The common schema accepts measured or predicted performance; report
        # measured values explicitly, not a predicted guarantee.
        'performance': {'before': source[GOAL], 'after': alt[GOAL]},
    }

    # The human requirement and shared oracle specify "closer to target"
    # on recorded historical measurements; find the least-edited feasible
    # *observed* candidate, reusing T7 valid_alternatives for exact edit counts.
    target = DEFAULT_TARGET
    if target >= source[GOAL]:
        target = (min(r[GOAL] for r in records) + source[GOAL]) / 2
    baseline_gap = abs(source[GOAL] - target)
    candidates = [
        item for item in valid_alternatives(source, records, records, settings)[1]
        if abs(by_id[item['row_number']][GOAL] - target) < baseline_gap
    ]
    candidates.sort(key=lambda item: (
        item['changes'],
        abs(by_id[item['row_number']][GOAL] - target),
        item['row_number'],
    ))
    if not candidates:
        raise ValueError('No recorded alternative closer to the target')
    chosen = candidates[0]
    second = {
        'row': source['row_number'],
        'target': target,
        'alternative': chosen['row_number'],
        'changed': sorted(change_map(chosen)),
    }
    return first, second


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: fix.py PATH_TO_APACHE_CSV')
    first, second = answers(Path(sys.argv[1]))
    print('== APACHE-15 ==')
    print('T7 small predicted repair; measured outcome reported for observed option')
    print(f"Source row {first['row']}, observed alternative row {first['alternative']}")
    print('ANSWER APACHE-15 ' + json.dumps(first, sort_keys=True))
    print('== APACHE-16 ==')
    print('Minimum measured edit distance toward target among observed T7 alternatives')
    print(f"Target {second['target']:g}; source row {second['row']}")
    print('ANSWER APACHE-16 ' + json.dumps(second, sort_keys=True))


if __name__ == '__main__':
    main()
