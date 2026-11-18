#!/usr/bin/env python3
"""Bridge original Apache T7 Blame to the shared APACHE-09/10 contracts.

Read the supplied CSV anew on every call, including shuffled prudence reruns.
No evaluation oracle is used to generate answers.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'human_t7' / 'apache'))
from remember import GOAL, adjustable_settings, load_configurations  # noqa: E402
from blame import split_poor_performers, analyze_individual_settings, analyze_setting_pairs  # noqa: E402


def answers(csv_path):
    data = load_configurations(csv_path)
    settings = adjustable_settings(data)
    poor, other, _ = split_poor_performers(data)
    singles = analyze_individual_settings(poor, other, settings)
    pairs = analyze_setting_pairs(poor, other, settings)
    bad_ids = [row['row_number'] for row in poor]
    good_ids = [row['row_number'] for row in other]
    # A binary feature's enrichment effect is the absolute prevalence gap
    # between the poor and other group; keep each feature once, most
    # associated first, rather than reporting its 0/1 values as columns.
    names = list(dict.fromkeys(item['setting'] for item in singles))
    first = {'poor': bad_ids, 'ranking': names}
    second = {
        'bad': bad_ids,
        'good': good_ids,
        'combinations': [
            {'settings': {
                item['first_setting']: item['first_value'],
                item['second_setting']: item['second_value'],
            }, 'bad_share': item['poor_rate'], 'good_share': item['other_rate']}
            for item in pairs[:10]
        ],
    }
    return first, second


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: blame.py PATH_TO_APACHE_CSV')
    a, b = answers(Path(sys.argv[1]))
    print('APACHE T7 BLAME: historical associations, not causal claims')
    print('== APACHE-09 ==')
    print('Identify settings associated with poor performance')
    print('Report the poor group and rank the distinguishing settings')
    print('ANSWER APACHE-09 ' + json.dumps(a, sort_keys=True))
    print('== APACHE-10 ==')
    print('Find combinations enriched in poorly performing configurations')
    print('Report historical prevalence in the poor and other groups')
    print('ANSWER APACHE-10 ' + json.dumps(b, sort_keys=True))


if __name__ == '__main__':
    main()
