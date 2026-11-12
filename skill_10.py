#!/usr/bin/env python3
"""
Extended data‑analysis script for the Auto93 dataset.

Adds two new client requests:
    CAR-19 – Compare the composite‑scoring vehicle selection system with
             random selection when both use the same evaluation budget.
    CAR-20 – Evaluate the performance of the regression‑based
             vehicle‑performance estimates against a simple historical‑average
             baseline on a hold‑out set of previously unseen records.

Usage:
    python skill_10.py <path_to_auto93.csv>
"""

import sys
import random
import statistics
from typing import List, Dict, Any

# --------------------------------------------------------------------------- #
# Imports from earlier skill modules
# --------------------------------------------------------------------------- #

from skill_01 import (
    read_csv,
    compute_score,
    format_float,
)

from skill_02 import (
    compute_regression_parameters,
    predict,
)

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def _bounds(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Return min/max bounds for weight, mpg, and acceleration."""
    return {
        'Lbs-': {
            'min': min(r['Lbs-'] for r in rows if r['Lbs-'] is not None),
            'max': max(r['Lbs-'] for r in rows if r['Lbs-'] is not None),
        },
        'Mpg+': {
            'min': min(r['Mpg+'] for r in rows if r['Mpg+'] is not None),
            'max': max(r['Mpg+'] for r in rows if r['Mpg+'] is not None),
        },
        'Acc+': {
            'min': min(r['Acc+'] for r in rows if r['Acc+'] is not None),
            'max': max(r['Acc+'] for r in rows if r['Acc+'] is not None),
        },
    }

def _compute_composite_scores(rows: List[Dict[str, Any]]) -> None:
    """Attach a composite 'score' field to every row that has complete data."""
    bnds = _bounds(rows)
    for r in rows:
        if None in (r['Lbs-'], r['Mpg+'], r['Acc+']):
            continue
        r['score'] = compute_score(
            r,
            {'min': bnds['Lbs-']['min'], 'max': bnds['Lbs-']['max']},
            {'min': bnds['Mpg+']['min'], 'max': bnds['Mpg+']['max']},
            {'min': bnds['Acc+']['min'], 'max': bnds['Acc+']['max']},
        )

# --------------------------------------------------------------------------- #
# CAR-19: Compare system vs random selection
# --------------------------------------------------------------------------- #

def _car19(rows: List[Dict[str, Any]]) -> None:
    print("== CAR-19 ==")

    # Ensure composite scores are available
    _compute_composite_scores(rows)

    # Filter to only rows with a score
    scored_rows = [r for r in rows if 'score' in r]
    if len(scored_rows) < 5:
        print("Not enough vehicles with complete data to perform comparison.")
        return

    # System: top‑k by score
    k = 5
    top_k = sorted(scored_rows, key=lambda r: r['score'], reverse=True)[:k]
    avg_sys = statistics.mean(r['score'] for r in top_k)

    # Random: 1000 trials, each selecting k distinct vehicles
    random.seed(42)
    n_trials = 1000
    rand_avgs = []
    indices = list(range(len(scored_rows)))
    for _ in range(n_trials):
        sample = random.sample(indices, k)
        rand_avgs.append(statistics.mean(scored_rows[i]['score'] for i in sample))
    avg_rand = statistics.mean(rand_avgs)
    std_rand = statistics.stdev(rand_avgs)

    diff = avg_sys - avg_rand

    print(f"Top-{k} average composite score: {format_float(avg_sys)}")
    print(f"Random-{k} average composite score: {format_float(avg_rand)} ± {format_float(std_rand)}")
    print(f"System outperforms random by {format_float(diff)} on average")

# --------------------------------------------------------------------------- #
# CAR-20: Regression estimates vs historical‑average baseline
# --------------------------------------------------------------------------- #

def _car20(rows: List[Dict[str, Any]]) -> None:
    print("\n== CAR-20 ==")

    # Random split 80% training / 20% test
    random.seed(123)
    shuffled = rows[:]
    random.shuffle(shuffled)
    split_point = int(len(shuffled) * 0.8)
    train_rows = shuffled[:split_point]
    test_rows = shuffled[split_point:]

    # Features for MPG prediction
    mpg_features = ['Lbs-', 'Acc+', 'Volume', 'HpX', 'Model', 'origin']

    # Features for Weight prediction
    weight_features = ['Acc+', 'Volume', 'HpX', 'Model', 'origin']  # exclude Lbs- target

    # ------------------------------------------------------------------ #
    # MPG estimation
    # ------------------------------------------------------------------ #
    # Training data for MPG
    train_mpg = [r for r in train_rows if r['Mpg+'] is not None and all(r.get(f) is not None for f in mpg_features)]
    if train_mpg:
        beta_mpg, _ = compute_regression_parameters(train_mpg, mpg_features, 'Mpg+')
        # Baseline: historical average from training
        train_mpg_vals = [r['Mpg+'] for r in train_mpg]
        baseline_mpg = sum(train_mpg_vals) / len(train_mpg_vals) if train_mpg_vals else float('nan')

        # Test predictions
        test_mpg = [r for r in test_rows if r['Mpg+'] is not None and all(r.get(f) is not None for f in mpg_features)]
        if test_mpg:
            mae_reg = statistics.mean(abs(r['Mpg+'] - predict(r, beta_mpg, mpg_features)) for r in test_mpg)
            mae_base = statistics.mean(abs(r['Mpg+'] - baseline_mpg) for r in test_mpg)
            improvement = (mae_base - mae_reg) / mae_base * 100.0 if mae_base != 0 else 0.0

            print(f"MPG MAE regression: {format_float(mae_reg)}")
            print(f"MPG MAE baseline   : {format_float(mae_base)}")
            print(f"Improvement         : {format_float(improvement)}%")
        else:
            print("No test rows with complete MPG features for evaluation.")
    else:
        print("Insufficient training data to build MPG regression model.")

    # ------------------------------------------------------------------ #
    # Weight estimation
    # ------------------------------------------------------------------ #
    train_wt = [r for r in train_rows if r['Lbs-'] is not None and all(r.get(f) is not None for f in weight_features)]
    if train_wt:
        beta_wt, _ = compute_regression_parameters(train_wt, weight_features, 'Lbs-')
        train_wt_vals = [r['Lbs-'] for r in train_wt]
        baseline_wt = sum(train_wt_vals) / len(train_wt_vals) if train_wt_vals else float('nan')

        test_wt = [r for r in test_rows if r['Lbs-'] is not None and all(r.get(f) is not None for f in weight_features)]
        if test_wt:
            mae_reg = statistics.mean(abs(r['Lbs-'] - predict(r, beta_wt, weight_features)) for r in test_wt)
            mae_base = statistics.mean(abs(r['Lbs-'] - baseline_wt) for r in test_wt)
            improvement = (mae_base - mae_reg) / mae_base * 100.0 if mae_base != 0 else 0.0

            print(f"Weight MAE regression: {format_float(mae_reg)}")
            print(f"Weight MAE baseline   : {format_float(mae_base)}")
            print(f"Improvement           : {format_float(improvement)}%")
        else:
            print("No test rows with complete Weight features for evaluation.")
    else:
        print("Insufficient training data to build Weight regression model.")

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <path_to_auto93.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        rows = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    _car19(rows)
    _car20(rows)

if __name__ == "__main__":
    main()
