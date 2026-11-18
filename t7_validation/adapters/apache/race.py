#!/usr/bin/env python3
"""APACHE-19/20 adapter: reuse T7 model and compare fair benchmarking arms.

The adapter receives an arbitrary CSV path for independent perturbation runs.
Neither oracle nor semantic checker is imported to construct answers.
"""
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'human_t7' / 'apache'))
from remember import GOAL, adjustable_settings, load_configurations  # noqa: E402
from guess import predict_performance, train_test_split  # noqa: E402
from choose import plan_inspections, simulate_inspections  # noqa: E402
from race import random_inspection_trials, paired_error_summary  # noqa: E402

REPEATS = 30
SEED = 42


def split_shared(rows):
    """Use the shared harness's every-fifth-row held-out protocol."""
    history = [row for row in rows if row['row_number'] % 5 != 0]
    test = [row for row in rows if row['row_number'] % 5 == 0]
    return history, test


def race_answers(csv_path):
    rows = load_configurations(csv_path)
    settings = adjustable_settings(rows)
    history, candidates = split_shared(rows)
    budget = min(30, len(rows) // 5)
    if len(history) < 5 or len(candidates) < budget:
        raise ValueError('Not enough training rows or benchmark candidates')

    # Repeated full selection trials: each repeat uses a freshly seeded
    # 80/20 split (the original T7 procedure), with the SAME unseen pool
    # and number of actual inspections for the two competing strategies.
    ours, random_rows = [], []
    for trial in range(REPEATS):
        training, unseen = train_test_split(rows, seed=SEED + trial)
        if len(unseen) < budget:
            raise ValueError('Too few held-out rows for the shared budget')
        plan = plan_inspections(unseen, training, settings)
        guided = simulate_inspections(plan, unseen, budget)
        ours.append(guided['best_row'])
        random_order = list(unseen)
        random.Random(SEED + 10000 + trial).shuffle(random_order)
        random_winner = min(random_order[:budget],
                            key=lambda row: (row[GOAL], row['row_number']))
        random_rows.append(random_winner['row_number'])

    vals_by_id = {r['row_number']: r[GOAL] for r in rows}
    ours_vals = [vals_by_id[r] for r in ours]
    random_vals = [vals_by_id[r] for r in random_rows]
    # Descriptive verdict, not a claim of a statistically clear difference.
    ours_median = statistics.median(ours_vals)
    random_median = statistics.median(random_vals)
    if ours_median < random_median:
        selection_verdict = 'ours better'
    elif ours_median > random_median:
        selection_verdict = 'random better'
    else:
        selection_verdict = 'no clear difference'
    selection = {'budget': budget, 'repeats': REPEATS, 'ours': ours,
                 'random': random_rows, 'verdict': selection_verdict}

    baseline = statistics.mean(r[GOAL] for r in history)
    estimates = {r['row_number']: predict_performance(r, history, settings)['prediction']
                 for r in candidates}
    actual = [r[GOAL] for r in candidates]
    model = [estimates[r['row_number']] for r in candidates]
    control = [baseline] * len(candidates)
    errors = paired_error_summary(actual, model, control)
    model_errors = [abs(a - p) for a, p in zip(actual, model)]
    baseline_errors = [abs(a - b) for a, b in zip(actual, control)]
    med_model = statistics.median(model_errors)
    med_base = statistics.median(baseline_errors)
    if med_model < med_base:
        prediction_verdict = 'better'
    elif med_model > med_base:
        prediction_verdict = 'worse'
    else:
        prediction_verdict = 'no clear difference'
    prediction = {
        'predictions': {str(k): v for k, v in estimates.items()},
        'baseline': baseline,
        'mae_model': errors['model_mae'],
        'mae_baseline': errors['baseline_mae'],
        'verdict': prediction_verdict,
    }
    return selection, prediction


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: race.py PATH_TO_APACHE_CSV')
    select, predict = race_answers(Path(sys.argv[1]))
    print('== APACHE-19 ==')
    print('Guided selection and seeded random selection at equal budget')
    print('30 independently seeded 80/20 candidate partitions; neither arm peeks')
    print('Recorded winners from both methods are compared after inspections')
    print('ANSWER APACHE-19 ' + json.dumps(select, sort_keys=True))
    print('== APACHE-20 ==')
    print('Held-out five-nearest-neighbor predictions versus training mean')
    print('Training-mean baseline excludes all held-out target outcomes')
    print('ANSWER APACHE-20 ' + json.dumps(predict, sort_keys=True))


if __name__ == '__main__':
    main()
