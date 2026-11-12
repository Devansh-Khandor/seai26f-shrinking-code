#!/usr/bin/env python3
"""
Skill 10: NASA-19 and NASA-20 – Comparative performance analysis
Author: Your Name
"""

import sys
import csv
import math
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from statistics import mean

# ----------------------------------------------------------------------
# CSV reading helper
# ----------------------------------------------------------------------
def read_csv(path):
    """
    Read the CSV and return a list of row dictionaries with numeric
    conversion where possible.
    """
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            row = {}
            for k, v in r.items():
                if v is None:
                    row[k] = v
                    continue
                v = v.strip()
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return rows

# ----------------------------------------------------------------------
# Build feature matrix for a target column
# ----------------------------------------------------------------------
def build_matrix(rows, target_col, exclude_cols=None):
    """
    Return X (n_samples, n_features), y (n_samples,) and feature names.
    """
    if exclude_cols is None:
        exclude_cols = []

    # Determine numeric predictor columns
    numeric_cols = []
    for key in rows[0].keys():
        if key in exclude_cols or key == target_col:
            continue
        if all(isinstance(r.get(key, None), (int, float)) or r.get(key, None) is None
               for r in rows):
            numeric_cols.append(key)

    X_rows = []
    y_rows = []
    for r in rows:
        tgt = r.get(target_col)
        if not isinstance(tgt, (int, float)):
            continue  # skip rows without valid target
        feat_vals = []
        missing = False
        for c in numeric_cols:
            val = r.get(c)
            if isinstance(val, (int, float)):
                feat_vals.append(val)
            else:
                missing = True
                break
        if missing:
            continue
        X_rows.append(feat_vals)
        y_rows.append(tgt)

    if not X_rows:
        raise ValueError(f"No valid rows found for target '{target_col}'.")
    return np.array(X_rows), np.array(y_rows), numeric_cols

# ----------------------------------------------------------------------
# NASA-19 – Prediction system vs historical average
# ----------------------------------------------------------------------
def nasa_19(rows):
    """
    Train a linear regression model on a training set and compare its
    prediction accuracy to a simple historical‑average estimate on a test set.
    """
    # Split into training (70%) and testing (30%) by index
    n = len(rows)
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    train_end = int(0.7 * n)
    train_idx = indices[:train_end]
    test_idx = indices[train_end:]

    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]

    # Prepare training data
    X_train, y_train, feat_names = build_matrix(train_rows, 'Effort-',
                                                exclude_cols=['idX', 'YearX', 'MonthsX'])
    model = LinearRegression().fit(X_train, y_train)

    # Prepare test data
    X_test, y_test, _ = build_matrix(test_rows, 'Effort-',
                                     exclude_cols=['idX', 'YearX', 'MonthsX'])
    pred_test = model.predict(X_test)

    # Historical average on training set
    avg_effort = mean(y_train)

    # Compute MAE for both
    mae_model = mean([abs(p - t) for p, t in zip(pred_test, y_test)])
    mae_hist = mean([abs(avg_effort - t) for t in y_test])

    diff = mae_hist - mae_model

    print("== NASA-19 ==")
    print(f"Number of projects: {n}")
    print(f"Training size: {len(y_train)}, Test size: {len(y_test)}")
    print(f"Model MAE on test set: {mae_model:.3f}")
    print(f"Historical‑average MAE on test set: {mae_hist:.3f}")
    print(f"MAE improvement (historical - model): {diff:.3f}")
    print()

# ----------------------------------------------------------------------
# NASA-20 – Comparison of prioritization vs random selection
# ----------------------------------------------------------------------
def nasa_20(rows):
    """
    Compare the benefit of selecting the top 5 projects using our
    predictive prioritization (NASA‑14 style) against randomly selecting
    5 projects under the same evaluation budget.
    """
    # Use the entire dataset for benefit calculation
    # Benefit metric: size / (effort * (defects + ε))
    eps = 1e-6

    # Build model for predictions to match NASA‑14's approach
    # We'll use a linear regression on Effort- and Defects- to predict
    X_eff, y_eff, eff_feats = build_matrix(rows, 'Effort-',
                                            exclude_cols=['idX', 'YearX', 'MonthsX'])
    eff_model = LinearRegression().fit(X_eff, y_eff)

    X_def, y_def, def_feats = build_matrix(rows, 'Defects-',
                                            exclude_cols=['idX', 'YearX', 'MonthsX'])
    def_model = LinearRegression().fit(X_def, y_def)

    # Compute benefit for each project using predicted values
    benefits = []
    for r in rows:
        # Build feature vectors for single row
        try:
            X_e, _, _ = build_matrix([r], 'Effort-',
                                     exclude_cols=['idX', 'YearX', 'MonthsX'])
            pred_eff = eff_model.predict(X_e)[0]
        except Exception:
            pred_eff = r.get('Effort-', None)
            if pred_eff is None:
                continue

        try:
            X_d, _, _ = build_matrix([r], 'Defects-',
                                     exclude_cols=['idX', 'YearX', 'MonthsX'])
            pred_def = def_model.predict(X_d)[0]
        except Exception:
            pred_def = r.get('Defects-', None)
            if pred_def is None:
                continue

        size = r.get('Kloc+', None)
        if not isinstance(size, (int, float)) or size <= 0:
            continue

        benefit = size / (pred_eff * (pred_def + eps))
        benefits.append((benefit, r))

    if not benefits:
        print("== NASA-20 ==")
        print("No valid projects to compute benefit.")
        print()
        return

    # Prioritized selection: top 5 by benefit
    benefits.sort(reverse=True, key=lambda x: x[0])
    top5 = benefits[:5]
    avg_prioritized = mean(b for b, _ in top5)

    # Random selection: 5 random projects
    random.seed(99)
    random_sample = random.sample(benefits, min(5, len(benefits)))
    avg_random = mean(b for b, _ in random_sample)

    diff = avg_prioritized - avg_random

    print("== NASA-20 ==")
    print(f"Total projects considered: {len(benefits)}")
    print(f"Average benefit (prioritized top 5): {avg_prioritized:.4f}")
    print(f"Average benefit (random 5): {avg_random:.4f}")
    print(f"Benefit improvement (prioritized - random): {diff:.4f}")
    print()

# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        rows = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if not rows:
        print("No data found in CSV.", file=sys.stderr)
        sys.exit(1)

    try:
        nasa_19(rows)
    except Exception as e:
        print(f"NASA-19 error: {e}", file=sys.stderr)

    try:
        nasa_20(rows)
    except Exception as e:
        print(f"NASA-20 error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
