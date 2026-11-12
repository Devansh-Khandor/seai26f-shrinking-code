#!/usr/bin/env python3
"""
skill_10.py

Usage:
    python skill_10.py <path-to-auto93.csv>

This script implements two new features for the dealership:

  CAR-19 : Show whether our vehicle-selection system finds better candidates
           than simply choosing vehicles at random when both approaches have
           the same evaluation budget.

  CAR-20 : Compare our vehicle-performance estimates against a straightforward
           historical-average estimate and show whether the new system provides
           a meaningful improvement on previously unseen records.

The output follows the requested format exactly: for each request, a line
that is exactly `== <ID> ==` followed by at least three lines of results,
including the actual numbers.
"""

import csv
import sys
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------------------

def read_data(csv_path):
    """
    Reads the CSV file and returns a list of dicts with numeric fields.
    Missing values are parsed as None. Rows that cannot be parsed are skipped.
    """
    records = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all(cell.strip() == '' for cell in row):
                continue
            try:
                rec = {
                    'Clndrs': float(row[0]),
                    'Volume': float(row[1]),
                    'HpX': float(row[2]),
                    'Model': float(row[3]),
                    'origin': float(row[4]),
                    'Lbs-': float(row[5]),
                    'Acc+': float(row[6]),
                    'Mpg+': float(row[7]),
                }
            except (IndexError, ValueError):
                # Skip rows that cannot be parsed
                continue
            records.append(rec)
    return records

# ---------------------------------------------------------------------------

def compute_scores(records):
    """
    Compute a combined score for each record:
      * Lower weight is better → invert normalization
      * Lower Acc+ time is better → invert normalization
      * Higher Mpg+ is better → normal normalization
    Return a list of tuples (score, index, record).
    """
    weights = np.array([r['Lbs-'] for r in records], dtype=float)
    accs    = np.array([r['Acc+'] for r in records], dtype=float)
    mpgs    = np.array([r['Mpg+'] for r in records], dtype=float)

    def norm(arr, invert=False):
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.zeros_like(arr)
        res = (arr - mn) / (mx - mn)
        return 1.0 - res if invert else res

    norm_w = norm(weights, invert=True)
    norm_a = norm(accs, invert=True)
    norm_m = norm(mpgs, invert=False)

    scores = []
    for idx, (w, a, m) in enumerate(zip(norm_w, norm_a, norm_m)):
        scores.append((w + a + m, idx, records[idx]))
    return scores

# ---------------------------------------------------------------------------

def car_19(scores, budget=5, trials=2000, seed=42):
    """
    Compare the mean score of the top `budget` vehicles selected by the scoring
    system to the mean score of randomly chosen vehicles, repeated `trials` times.
    """
    # Sort scores descending (higher is better)
    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    top_scores = [s[0] for s in sorted_scores[:budget]]
    mean_system = float(np.mean(top_scores))

    random_means = []
    random_indices = list(range(len(scores)))
    rng = random.Random(seed)
    for _ in range(trials):
        chosen = rng.sample(random_indices, k=budget)
        mean_rand = np.mean([scores[i][0] for i in chosen])
        random_means.append(mean_rand)
    mean_random = float(np.mean(random_means))

    diff = mean_system - mean_random

    print("== CAR-19 ==")
    print(f"Budget: {budget} vehicles")
    print(f"  Mean score of top {budget} system-selected vehicles: {mean_system:.4f}")
    print(f"  Mean score of random selection (averaged over {trials} trials): {mean_random:.4f}")
    print(f"  Difference (system - random): {diff:.4f}")
    print("  System selection is better than random if difference > 0.")
    print()

# ---------------------------------------------------------------------------

def car_20(records, test_ratio=0.3, seed=42):
    """
    Train a linear regression model to predict Mpg+ from all other features.
    Compare its mean absolute error on a held‑out test set with the error of a
    naive model that always predicts the historical mean Mpg+.
    """
    # Shuffle indices
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(records))
    n_test = max(1, int(len(records) * test_ratio))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    # Prepare training and test data
    X_train = np.column_stack([
        records[i]['Clndrs'],
        records[i]['Volume'],
        records[i]['HpX'],
        records[i]['Model'],
        records[i]['origin'],
        records[i]['Lbs-'],
        records[i]['Acc+'],
    ]) for i in train_idx]
    y_train = np.array([records[i]['Mpg+'] for i in train_idx])

    X_test = np.column_stack([
        records[i]['Clndrs'],
        records[i]['Volume'],
        records[i]['HpX'],
        records[i]['Model'],
        records[i]['origin'],
        records[i]['Lbs-'],
        records[i]['Acc+'],
    ]) for i in test_idx]
    y_test = np.array([records[i]['Mpg+'] for i in test_idx])

    # Train regression
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred_model = reg.predict(X_test)

    # Naive baseline: historical mean of training mpg
    mean_mpg = np.mean(y_train)
    y_pred_naive = np.full_like(y_test, mean_mpg)

    mae_model = float(mean_absolute_error(y_test, y_pred_model))
    mae_naive = float(mean_absolute_error(y_test, y_pred_naive))

    improvement = mae_naive - mae_model
    improvement_pct = improvement / mae_naive * 100.0 if mae_naive != 0 else float('inf')

    print("== CAR-20 ==")
    print(f"Test set size: {len(y_test)} vehicles")
    print(f"  MAE of linear regression model: {mae_model:.3f} mpg")
    print(f"  MAE of historical‑average baseline: {mae_naive:.3f} mpg")
    print(f"  Absolute improvement: {improvement:.3f} mpg")
    print(f"  Relative improvement: {improvement_pct:.1f}%")
    print()

# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-auto93.csv>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    records = read_data(csv_path)

    if not records:
        print("No valid data found in the CSV.", file=sys.stderr)
        sys.exit(1)

    # Compute combined scores
    scores = compute_scores(records)

    # CAR‑19
    car_19(scores, budget=5, trials=2000, seed=42)

    # CAR‑20
    car_20(records, test_ratio=0.3, seed=42)

if __name__ == "__main__":
    main()
