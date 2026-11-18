#!/usr/bin/env python3
"""Apache T7 Choose (APACHE-13/14) bridge to common semantic harness.

Predictions/order use only visible historical outcomes and candidate settings.
Outcomes for the selected evaluations are inspected *after* ordering.
No checker/oracle imports or hard-coded expected results.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'human_t7' / 'apache'))
from remember import GOAL, adjustable_settings, load_configurations  # noqa: E402
from choose import BUDGETS, plan_inspections, simulate_inspections  # noqa: E402


def answers(csv_path):
    rows = load_configurations(csv_path)
    settings = adjustable_settings(rows)
    # Shared Choose protocol uses a global inspection budget min(30, n//5).
    # Keep every fifth row as initially uninspected candidates, as in Guess.
    history = [r for r in rows if r['row_number'] % 5 != 0]
    candidates = [r for r in rows if r['row_number'] % 5 == 0]
    permitted = min(30, len(rows) // 5)
    budget = max((b for b in BUDGETS if b <= permitted), default=0)
    if len(history) < 5 or len(candidates) < budget:
        raise ValueError('Insufficient historical or candidate configurations')
    plan = plan_inspections(candidates, history, settings)
    selection = simulate_inspections(plan, candidates, budget)
    evaluated = [entry['row_number'] for entry in selection['inspected']]
    return ({'evaluated': evaluated, 'best': selection['best_row']},
            {'order': [entry['row_number'] for entry in plan[:budget]]})


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: choose.py PATH_TO_APACHE_CSV')
    first, second = answers(Path(sys.argv[1]))
    print('== APACHE-13 ==')
    print(f"Evaluated {len(first['evaluated'])} candidate configurations")
    print('Best measured row after budgeted inspections:', first['best'])
    print('ANSWER APACHE-13 ' + json.dumps(first, sort_keys=True))
    print('== APACHE-14 ==')
    print('Prioritized evaluation order from T7 kNN predictions')
    print('Budgeted recommendation count:', len(second['order']))
    print('ANSWER APACHE-14 ' + json.dumps(second, sort_keys=True))


if __name__ == '__main__':
    main()
