#!/usr/bin/env python3
"""
Skill 10: Vehicle‑selection benchmarking and predictive accuracy evaluation.

CAR‑19: Compare a simple input‑based vehicle‑selection system to random selection
        under a fixed evaluation budget.

CAR‑20: Evaluate the accuracy of a linear‑regression model for predicting
        Mpg+ against a historical‑average baseline on held‑out rows.
"""

import csv
import json
import sys
import math
import random
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def parse_value(val: str):
    """Parse a string from the CSV into a float or None."""
    if val == '?' or val.strip() == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def is_goal(col_name: str) -> bool:
    return (col_name.endswith('+') or col_name.endswith('-')) and not col_name.endswith('X')

def is_maximize(col_name: str) -> bool:
    return col_name.endswith('+')

def is_input(col_name: str) -> bool:
    return not is_goal(col_name) and not col_name.endswith('X')

# ----------------------------------------------------------------------
# CSV IO
# ----------------------------------------------------------------------
def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows, reader.fieldnames

# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def compute_minmax(rows, cols):
    mins, maxs = {}, {}
    for col in cols:
        vals = [parse_value(r[col]) for r in rows if parse_value(r[col]) is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
    return mins, maxs

def normalize_value(v, mn, mx, maximize=True):
    if mx == mn:
        return 1.0
    return (v - mn) / (mx - mn) if maximize else (mx - v) / (mx - mn)

# ----------------------------------------------------------------------
# Overall goal score (higher is better)
# ----------------------------------------------------------------------
def compute_goal_score(row, goal_cols, goal_mins, goal_maxs):
    score = 0.0
    for col in goal_cols:
        v = parse_value(row[col])
        if v is None:
            return None
        mn = goal_mins[col]
        mx = goal_maxs[col]
        norm = normalize_value(v, mn, mx, maximize=is_maximize(col))
        score += norm
    return score

# ----------------------------------------------------------------------
# CAR‑19 – selection comparison
# ----------------------------------------------------------------------
def car_19(rows, goal_cols):
    n_rows = len(rows)
    budget = min(30, n_rows // 5)
    repeats = 20

    # Pre‑compute min/max for input columns
    input_cols = [c for c in rows[0].keys() if is_input(c)]
    in_min, in_max = compute_minmax(rows, input_cols)

    # Compute overall goal scores for reference
    goal_mins, goal_maxs = compute_minmax(rows, goal_cols)
    overall_scores = []
    for idx, r in enumerate(rows, start=1):
        s = compute_goal_score(r, goal_cols, goal_mins, goal_maxs)
        overall_scores.append((idx, s if s is not None else float('-inf')))

    # Helper to pick best among a set of indices
    def best_from_indices(indices):
        best_idx, best_score = None, float('-inf')
        for i in indices:
            idx = i + 1  # zero‑based to 1‑based
            s = overall_scores[i][1]
            if s > best_score:
                best_idx, best_score = idx, s
        return best_idx, best_score

    # Our selection strategy – score rows based on inputs
    input_scores = []
    for idx, r in enumerate(rows):
        vals = [parse_value(r[c]) for c in input_cols]
        if any(v is None for v in vals):
            continue
        score = 0.0
        for c, v in zip(input_cols, vals):
            score += normalize_value(v, in_min[c], in_max[c], maximize=True)
        input_scores.append((idx, score))

    # Sort for our strategy
    input_scores.sort(key=lambda x: (-x[1], x[0]))
    our_candidates = [idx for idx, _ in input_scores[:budget]]

    # Random strategy – random sample
    random.seed(42)  # reproducible
    rng = random.Random(42)
    all_indices = list(range(n_rows))
    results = []
    for _ in range(repeats):
        random_indices = rng.sample(all_indices, budget)
        our_best_idx, our_best_score = best_from_indices(our_candidates)
        rand_best_idx, rand_best_score = best_from_indices(random_indices)
        results.append({
            "ours": our_best_idx,
            "rand": rand_best_idx,
            "ours_score": our_best_score,
            "rand_score": rand_best_score
        })

    # Aggregate results
    ours_list = [r["ours"] for r in results]
    rand_list = [r["rand"] for r in results]
    ours_better = sum(1 for r in results if r["ours_score"] > r["rand_score"])
    rand_better = sum(1 for r in results if r["rand_score"] > r["ours_score"])
    if ours_better > rand_better:
        verdict = "ours better"
    elif rand_better > ours_better:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    # Output
    print("== CAR-19 ==")
    print(f"Budget: {budget}")
    print(f"Repeats: {repeats}")
    print(f"Ours best per repeat: {ours_list}")
    print(f"Random best per repeat: {rand_list}")
    print(f"Verdict: {verdict}")

    answer = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_list,
        "random": rand_list,
        "verdict": verdict
    }
    print("ANSWER CAR-19", json.dumps(answer, ensure_ascii=False))

# ----------------------------------------------------------------------
# CAR‑20 – prediction accuracy
# ----------------------------------------------------------------------
def car_20(rows, goal_cols):
    n_rows = len(rows)
    # Identify held‑out rows (divisible by 5)
    held_out_indices = [i for i in range(n_rows) if (i + 1) % 5 == 0]
    train_indices = [i for i in range(n_rows) if (i + 1) % 5 != 0]

    # Prepare training data
    input_cols = [c for c in rows[0].keys() if is_input(c)]
    # Filter numeric input columns only
    numeric_inputs = []
    for col in input_cols:
        if all(parse_value(r[col]) is not None for r in rows if r[col] != '?'):
            numeric_inputs.append(col)
    if not numeric_inputs:
        print("No numeric input columns for regression.")
        sys.exit(1)

    X_train = []
    y_train = []
    for i in train_indices:
        row = rows[i]
        y_val = parse_value(row['Mpg+'])
        if y_val is None:
            continue
        feat = [parse_value(row[c]) for c in numeric_inputs]
        X_train.append(feat)
        y_train.append(y_val)
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=float)

    # Build pipeline
    pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    pipe.fit(X_train, y_train)

    # Baseline mean
    baseline = np.mean(y_train)

    # Predictions on held‑out
    predictions = {}
    mae_model = 0.0
    mae_baseline = 0.0
    count_model = 0
    count_baseline = 0

    for i in held_out_indices:
        row = rows[i]
        # Input features
        feat = [parse_value(row[c]) for c in numeric_inputs]
        pred = pipe.predict([feat])[0]
        predictions[i + 1] = round(pred, 6)
        actual_val = parse_value(row['Mpg+'])
        if actual_val is not None:
            mae_model += abs(pred - actual_val)
            mae_baseline += abs(baseline - actual_val)
            count_model += 1
            count_baseline += 1

    mae_model = mae_model / count_model if count_model else 0.0
    mae_baseline = mae_baseline / count_baseline if count_baseline else 0.0

    if mae_model < mae_baseline - 1e-6:
        verdict = "better"
    elif mae_model > mae_baseline + 1e-6:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    # Output
    print("== CAR-20 ==")
    print(f"Predictions for held‑out rows: {predictions}")
    print(f"Baseline mean Mpg+: {baseline:.6f}")
    print(f"MAE model: {mae_model:.6f}")
    print(f"MAE baseline: {mae_baseline:.6f}")
    print(f"Verdict: {verdict}")

    answer = {
        "predictions": predictions,
        "baseline": round(baseline, 6),
        "mae_model": round(mae_model, 6),
        "mae_baseline": round(mae_baseline, 6),
        "verdict": verdict
    }
    print("ANSWER CAR-20", json.dumps(answer, ensure_ascii=False))

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    rows, fieldnames = read_csv(path)
    goal_cols = [c for c in fieldnames if is_goal(c)]

    car_19(rows, goal_cols)
    car_20(rows, goal_cols)

if __name__ == "__main__":
    main()
