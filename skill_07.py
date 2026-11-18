#!/usr/bin/env python3
"""
Apache configuration selection and evaluation.

Usage:  python skill_07.py <csv_path>
"""

import csv
import json
import math
import sys
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
    """Convert string to float or return None if missing."""
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

def feature_columns(fieldnames, goal_col):
    """Return columns that are inputs: not a goal and not ending with 'X'."""
    return [c for c in fieldnames if c != goal_col and not c.endswith('X')]

# ---------------------------------------------------------------------------

def build_raw_feature_matrix(rows, feature_cols):
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

def adaptive_selection(rows, goal_col, feature_cols, max_evals=30):
    """
    Return evaluated list (1‑based indices) and best index.
    Uses linear regression to iteratively pick rows with best predicted performance.
    """
    n = len(rows)
    goal_vals = np.array([to_float(r[goal_col]) for r in rows], dtype=object)
    # Indices with known goal
    valid_indices = [i for i, v in enumerate(goal_vals) if v is not None]
    if not valid_indices:
        return [], None

    raw_X = build_raw_feature_matrix(rows, feature_cols)

    evaluated = []
    train_idx = []

    # Start with first valid row
    start_idx = valid_indices[0]
    evaluated.append(start_idx + 1)  # 1‑based
    train_idx.append(start_idx)

    while len(evaluated) < min(max_evals, len(valid_indices)):
        # Train data
        X_train_raw = raw_X[train_idx, :]
        y_train = np.array([goal_vals[i] for i in train_idx], dtype=float)

        # Impute
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

        # Choose best predicted (min or max)
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
        if best_value is None or v < best_value:
            best_value = v
            best_actual = idx
    return evaluated, best_actual

# ---------------------------------------------------------------------------

def main(csv_path):
    rows, fieldnames = read_csv(csv_path)
    goal_col = find_goal_column(fieldnames)
    if goal_col is None:
        print("No goal column found", file=sys.stderr)
        sys.exit(1)
    feature_cols = feature_columns(fieldnames, goal_col)

    # -------------------------------------------------------------
    # APACHE-13
    # -------------------------------------------------------------
    print("== APACHE-13 ==")
    evaluated, best_row = adaptive_selection(rows, goal_col, feature_cols, max_evals=30)
    # Print at least three lines of results
    print(f"Evaluated rows: {evaluated}")
    print(f"Best row: {best_row}")
    # Show best performance value
    best_perf = to_float(rows[best_row - 1][goal_col]) if best_row else None
    print(f"Performance of best: {best_perf}")
    answer_13 = {"evaluated": evaluated, "best": best_row}
    print(f"ANSWER APACHE-13 {json.dumps(answer_13, separators=(',', ':'))}")

    # -------------------------------------------------------------
    # APACHE-14
    # -------------------------------------------------------------
    print("\n== APACHE-14 ==")
    # The order recommended is simply the evaluation order from APACHE-13
    order = evaluated
    # Print at least three lines of results
    print(f"Order to evaluate: {order}")
    # Show first three rows in order as an example
    if len(order) >= 3:
        print(f"First three: {order[:3]}")
    else:
        print(f"First rows: {order}")
    answer_14 = {"order": order}
    print(f"ANSWER APACHE-14 {json.dumps(answer_14, separators=(',', ':'))}")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
