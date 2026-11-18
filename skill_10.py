#!/usr/bin/env python3
"""
Apache configuration selection and prediction comparison.

This script implements the APACHE-19 and APACHE-20 analyses as requested.

Usage:
    python skill_10.py <csv_path>
"""

import csv
import json
import math
import random
import sys
from collections import defaultdict

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------

def read_csv(csv_path):
    """Read CSV and return list of row dicts and fieldnames."""
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows, reader.fieldnames

# ---------------------------------------------------------------------------

def to_float(val):
    """Convert string to float or None if missing."""
    if val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

# ---------------------------------------------------------------------------

def find_goal_column(fieldnames):
    """Return the column that ends with '+' or '-'."""
    for c in fieldnames:
        if c.endswith('+') or c.endswith('-'):
            return c
    return None

# ---------------------------------------------------------------------------

def get_setting_columns(fieldnames, goal_col):
    """Return columns that are inputs: not goal and not ending with 'X'."""
    return [c for c in fieldnames if c != goal_col and not c.endswith('X')]

# ---------------------------------------------------------------------------

def build_feature_matrix(rows, feature_cols):
    """Return numpy array of shape (n_rows, n_features) with np.nan for missing."""
    n_rows = len(rows)
    n_feats = len(feature_cols)
    X = np.empty((n_rows, n_feats), dtype=float)
    X[:] = np.nan
    for i, row in enumerate(rows):
        for j, col in enumerate(feature_cols):
            val = to_float(row[col])
            if val is not None:
                X[i, j] = val
    return X

# ---------------------------------------------------------------------------

def adaptive_selection(rows, goal_col, feature_cols, budget):
    """
    Return evaluated list (1‑based indices) and best index among them.
    Uses a simple greedy strategy: start from first evaluable row, then repeatedly
    train a linear regression on already evaluated rows and pick the un‑evaluated
    row with the best predicted goal.
    """
    goal_vals = np.array([to_float(r[goal_col]) for r in rows], dtype=object)
    valid_indices = [i for i, v in enumerate(goal_vals) if v is not None]
    if not valid_indices:
        return [], None

    raw_X = build_feature_matrix(rows, feature_cols)

    evaluated = []
    train_idx = []

    # start with first valid row
    start_idx = valid_indices[0]
    evaluated.append(start_idx + 1)  # 1‑based
    train_idx.append(start_idx)

    while len(evaluated) < min(budget, len(valid_indices)):
        # Train data
        X_train_raw = raw_X[train_idx, :]
        y_train = np.array([goal_vals[i] for i in train_idx], dtype=float)

        # Impute missing values
        imp = SimpleImputer(strategy='mean')
        X_train = imp.fit_transform(X_train_raw)

        # Train model
        reg = LinearRegression()
        reg.fit(X_train, y_train)

        # Remaining candidates
        remaining = [i for i in valid_indices if i not in train_idx]
        if not remaining:
            break

        X_rem_raw = raw_X[remaining, :]
        X_rem = imp.transform(X_rem_raw)

        # Predict
        preds = reg.predict(X_rem)

        # Choose best predicted
        if goal_col.endswith('-'):  # minimize
            best_pos = int(np.argmin(preds))
        else:
            best_pos = int(np.argmax(preds))

        next_idx = remaining[best_pos]
        evaluated.append(next_idx + 1)
        train_idx.append(next_idx)

    # Determine best actual row among evaluated
    best_actual = None
    best_value = None
    for idx in evaluated:
        v = goal_vals[idx - 1]
        if v is None:
            continue
        if best_value is None:
            best_value = v
            best_actual = idx
        else:
            if goal_col.endswith('-'):
                if v < best_value:
                    best_value = v
                    best_actual = idx
            else:
                if v > best_value:
                    best_value = v
                    best_actual = idx
    return evaluated, best_actual

# ---------------------------------------------------------------------------

def random_selection(rows, goal_col, budget, rng):
    """
    Randomly select up to `budget` rows that have a valid goal value.
    """
    goal_vals = [to_float(r[goal_col]) for r in rows]
    candidates = [i+1 for i, v in enumerate(goal_vals) if v is not None]
    if len(candidates) <= budget:
        selected = candidates
    else:
        selected = rng.sample(candidates, budget)

    # Find best among selected
    best_idx = None
    best_val = None
    for idx in selected:
        v = goal_vals[idx-1]
        if v is None:
            continue
        if best_val is None:
            best_val = v
            best_idx = idx
        else:
            if goal_col.endswith('-'):
                if v < best_val:
                    best_val = v
                    best_idx = idx
            else:
                if v > best_val:
                    best_val = v
                    best_idx = idx
    return selected, best_idx

# ---------------------------------------------------------------------------

def apache_19(rows, fieldnames, goal_col, feature_cols, budget, repeats=20):
    """Compare adaptive selection vs random selection."""
    ours_best = []
    random_best = []

    for rep in range(repeats):
        rng = random.Random(rep)  # deterministic per repeat
        _, best_ours = adaptive_selection(rows, goal_col, feature_cols, budget)
        _, best_rand = random_selection(rows, goal_col, budget, rng)
        ours_best.append(best_ours)
        random_best.append(best_rand)

    # Count wins
    ours_wins = 0
    rand_wins = 0
    ties = 0

    # For comparison, we need actual goal values
    goal_vals = [to_float(r[goal_col]) for r in rows]

    for o, r in zip(ours_best, random_best):
        vo = goal_vals[o-1] if o is not None else None
        vr = goal_vals[r-1] if r is not None else None
        if vo is None or vr is None:
            ties += 1
            continue
        if goal_col.endswith('-'):  # minimize
            if vo < vr:
                ours_wins += 1
            elif vo > vr:
                rand_wins += 1
            else:
                ties += 1
        else:  # maximize
            if vo > vr:
                ours_wins += 1
            elif vo < vr:
                rand_wins += 1
            else:
                ties += 1

    if ours_wins > rand_wins:
        verdict = "ours better"
    elif rand_wins > ours_wins:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    # Prepare answer JSON
    answer = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best,
        "random": random_best,
        "verdict": verdict
    }

    # Print output
    print("== APACHE-19 ==")
    print(f"Budget: {budget}")
    print(f"Repeats: {repeats}")
    print(f"Ours best rows: {ours_best[:3]} ...")
    print(f"Random best rows: {random_best[:3]} ...")
    print(f"Verdict: {verdict}")
    print(f"ANSWER APACHE-19 {json.dumps(answer, separators=(',', ':'))}")

# ---------------------------------------------------------------------------

def apache_20(rows, fieldnames, goal_col, feature_cols):
    """Compare predictions vs baseline mean on held‑out rows."""
    # Identify held‑out rows (row number divisible by 5, 1‑based)
    held_out_indices = [i+1 for i in range(len(rows)) if (i+1) % 5 == 0]
    # Build training set: non‑held‑out rows with known goal
    train_rows = [r for idx, r in enumerate(rows, start=1)
                  if idx not in held_out_indices and to_float(r[goal_col]) is not None]
    # Build test set: held‑out rows with known goal
    test_rows = [r for idx, r in enumerate(rows, start=1)
                 if idx in held_out_indices and to_float(r[goal_col]) is not None]
    # If no training or testing data, skip
    if not train_rows or not test_rows:
        predictions = {}
        baseline = None
        mae_model = None
        mae_baseline = None
        verdict = "no clear difference"
    else:
        # Train linear regression on training set
        X_train = build_feature_matrix(train_rows, feature_cols)
        y_train = np.array([to_float(r[goal_col]) for r in train_rows], dtype=float)

        imp = SimpleImputer(strategy='mean')
        X_train_imputed = imp.fit_transform(X_train)

        reg = LinearRegression()
        reg.fit(X_train_imputed, y_train)

        # Predict on test set
        X_test = build_feature_matrix(test_rows, feature_cols)
        X_test_imputed = imp.transform(X_test)
        y_pred = reg.predict(X_test_imputed)

        # Actual values
        y_true = np.array([to_float(r[goal_col]) for r in test_rows], dtype=float)

        # Compute MAE
        mae_model = float(np.mean(np.abs(y_true - y_pred)))

        # Baseline mean from training set
        baseline = float(np.mean(y_train))
        mae_baseline = float(np.mean(np.abs(y_true - baseline)))

        # Verdict
        if mae_model < mae_baseline:
            verdict = "better"
        elif mae_model > mae_baseline:
            verdict = "worse"
        else:
            verdict = "no clear difference"

        # Build predictions dict mapping row number to predicted value
        predictions = {}
        for r, pred in zip([rows[idx-1] for idx in held_out_indices], y_pred):
            row_num = rows.index(r) + 1
            predictions[str(row_num)] = float(pred)

    answer = {
        "predictions": predictions,
        "baseline": baseline,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict
    }

    print("== APACHE-20 ==")
    if predictions:
        sample_keys = list(predictions.keys())[:3]
        print(f"Predictions for rows {sample_keys} ...")
    else:
        print("No predictions available")
    print(f"Baseline mean: {baseline}")
    print(f"MAE model: {mae_model}")
    print(f"MAE baseline: {mae_baseline}")
    print(f"Verdict: {verdict}")
    print(f"ANSWER APACHE-20 {json.dumps(answer, separators=(',', ':'))}")

# ---------------------------------------------------------------------------

def main(csv_path):
    rows, fieldnames = read_csv(csv_path)
    goal_col = find_goal_column(fieldnames)
    if goal_col is None:
        print("No goal column found", file=sys.stderr)
        sys.exit(1)
    setting_cols = get_setting_columns(fieldnames, goal_col)

    # Determine budget
    n_rows = len(rows)
    budget = min(30, n_rows // 5)

    apache_19(rows, fieldnames, goal_col, setting_cols, budget, repeats=20)
    print()
    apache_20(rows, fieldnames, goal_col, setting_cols)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
