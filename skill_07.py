#!/usr/bin/env python3
"""
Skill 07: Active selection of configurations.
- APACHE-13: Find a low‑performance configuration while evaluating as few rows as possible.
- APACHE-14: Rank all configurations so that promising ones appear early, given a limited budget.

Both tasks use a greedy strategy that learns a linear regression model on the rows already evaluated,
then selects the next row with the lowest predicted performance.
"""

import csv
import json
import sys
import math
from collections import defaultdict
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# CSV reading – identical logic to skill_01
# ----------------------------------------------------------------------
def read_csv(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            cleaned = {}
            for k, v in row.items():
                if v == '?':
                    cleaned[k] = None
                else:
                    try:
                        cleaned[k] = float(v) if '.' in v else int(v)
                    except ValueError:
                        cleaned[k] = v
            rows.append((idx, cleaned))
    return rows, reader.fieldnames

# ----------------------------------------------------------------------
# Utility: determine goal & setting columns
# ----------------------------------------------------------------------
def identify_columns(fieldnames):
    goal_columns = [c for c in fieldnames if c.endswith(('+', '-'))]
    setting_columns = [c for c in fieldnames
                       if c not in goal_columns and not c.endswith('X')]
    return goal_columns, setting_columns

# ----------------------------------------------------------------------
# Build linear regression model from a set of evaluated indices
# ----------------------------------------------------------------------
def build_model(evaluated_indices, rows, setting_columns, goal_col):
    """Return (model, means) where means maps setting->mean for imputation."""
    # Compute means of each setting over evaluated rows
    means = {}
    for col in setting_columns:
        vals = [r[col] for idx, r in rows if idx in evaluated_indices and r[col] is not None]
        means[col] = sum(vals) / len(vals) if vals else 0.0

    # Prepare training data
    X_train, y_train = [], []
    for idx, r in rows:
        if idx not in evaluated_indices:
            continue
        if r[goal_col] is None:
            continue
        vec = []
        for col in setting_columns:
            val = r[col]
            if val is None:
                val = means[col]
            vec.append(val)
        X_train.append(vec)
        y_train.append(r[goal_col])

    if len(X_train) >= 2:
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        return lr, means
    # If only 0 or 1 training example, return a constant predictor
    class ConstModel:
        def predict(self, X):
            return [y_train[0] if X else 0.0] * len(X)
    return ConstModel(), means

# ----------------------------------------------------------------------
# Predict value for a single row using model and means
# ----------------------------------------------------------------------
def predict_row(row, model, means, setting_columns):
    vec = []
    for col in setting_columns:
        val = row[col]
        if val is None:
            val = means[col]
        vec.append(val)
    return model.predict([vec])[0]

# ----------------------------------------------------------------------
# APACHE-13: Greedy active selection
# ----------------------------------------------------------------------
def apache_13(rows, setting_columns, goal_col, budget):
    evaluated = []
    evaluated_set = set()
    best_idx = None
    best_val = None

    # Helper to select next row
    def select_next():
        if len(evaluated) < 2:
            # Pick smallest index not yet evaluated
            for idx, _ in rows:
                if idx not in evaluated_set:
                    return idx
        # Build model on evaluated rows
        model, means = build_model(evaluated_set, rows, setting_columns, goal_col)
        # Predict remaining
        best_pred = None
        best_row = None
        for idx, r in rows:
            if idx in evaluated_set:
                continue
            pred = predict_row(r, model, means, setting_columns)
            if best_pred is None or pred < best_pred:
                best_pred = pred
                best_row = idx
        return best_row

    # Evaluate until budget or all rows
    total = len(rows)
    while len(evaluated) < min(budget, total):
        nxt = select_next()
        if nxt is None:
            break
        # evaluate
        evaluated.append(nxt)
        evaluated_set.add(nxt)
        _, r = next(item for item in rows if item[0] == nxt)
        val = r[goal_col]
        if val is not None:
            if best_val is None or val < best_val:
                best_val = val
                best_idx = nxt

    return evaluated, best_idx

# ----------------------------------------------------------------------
# APACHE-14: Rank all rows for early discovery
# ----------------------------------------------------------------------
def apache_14(rows, setting_columns, goal_col, full_order_count):
    evaluated = []
    evaluated_set = set()

    def select_next():
        if len(evaluated) < 2:
            for idx, _ in rows:
                if idx not in evaluated_set:
                    return idx
        model, means = build_model(evaluated_set, rows, setting_columns, goal_col)
        best_pred = None
        best_row = None
        for idx, r in rows:
            if idx in evaluated_set:
                continue
            pred = predict_row(r, model, means, setting_columns)
            if best_pred is None or pred < best_pred:
                best_pred = pred
                best_row = idx
        return best_row

    total = len(rows)
    while len(evaluated) < min(full_order_count, total):
        nxt = select_next()
        if nxt is None:
            break
        evaluated.append(nxt)
        evaluated_set.add(nxt)

    return evaluated

# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)
    goal_columns, setting_columns = identify_columns(fieldnames)

    if not goal_columns:
        print("Error: No goal column found.")
        sys.exit(1)

    goal_col = goal_columns[0]          # assume single goal
    n_rows = len(rows)
    budget = min(30, n_rows // 5)      # per specification

    # ---------- APACHE-13 ----------
    print("== APACHE-13 ==")
    evaluated, best_idx = apache_13(rows, setting_columns, goal_col, budget)
    print(f"Evaluated rows (up to {budget}): {evaluated}")
    print(f"Best (lowest) Performance- row: {best_idx}")
    # JSON answer
    ans_13 = {
        "evaluated": evaluated,
        "best": best_idx
    }
    print(f"ANSWER APACHE-13 {json.dumps(ans_13)}")

    # ---------- APACHE-14 ----------
    print("\n== APACHE-14 ==")
    # Order of the first 30 rows (or all if fewer)
    order = apache_14(rows, setting_columns, goal_col, 30)
    print(f"Proposed evaluation order (first 30 rows): {order}")
    ans_14 = {
        "order": order
    }
    print(f"ANSWER APACHE-14 {json.dumps(ans_14)}")

if __name__ == "__main__":
    main()
