#!/usr/bin/env python3
"""
Skill 10 – NASA-19 & NASA-20

NASA-19: Compare a linear‑regression based Effort‑ predictor against a simple
historical‑average baseline on held‑out projects.
NASA-20: Compare a deterministic budgeted project‑selection strategy with random
selection, evaluating which method finds a better project in terms of all goal
metrics.

All statistics are computed from the CSV file supplied on the command line.
"""

import sys
import csv
import json
import math
import random
from collections import defaultdict

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def parse_float(val):
    """Convert a CSV cell into a float, handling missing data."""
    if val is None or val == '' or val.strip() == '?':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def is_goal(col):
    return col.endswith('+') or col.endswith('-')

def is_excluded(col):
    return col.endswith('X')

def read_csv(path):
    """Return data rows and fieldnames."""
    data = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            data.append((idx, row))
    return data, reader.fieldnames

# ----------------------------------------------------------------------
# NASA-19 implementation
# ----------------------------------------------------------------------
def nasa_19(data, fieldnames):
    print("== NASA-19 ==")

    # Identify held‑out rows (index divisible by 5)
    heldout_set = {idx for idx, _ in data if idx % 5 == 0}
    non_heldout = [(idx, row) for idx, row in data if idx not in heldout_set]

    # Prepare features and targets for model training
    feature_cols = [c for c in fieldnames if not is_excluded(c) and c != 'Effort-']
    X, y = [], []
    for idx, row in non_heldout:
        target = parse_float(row['Effort-'])
        if target is None:
            continue
        feat = []
        skip = False
        for c in feature_cols:
            v = parse_float(row[c])
            if v is None:
                skip = True
                break
            feat.append(v)
        if skip:
            continue
        X.append(feat)
        y.append(target)

    if not X:
        print("Insufficient data to train a model.")
        # Build baseline anyway
        baseline_val = None
    else:
        # Train a simple linear regression (closed‑form)
        import numpy as np
        X_np = np.array(X, dtype=float)
        y_np = np.array(y, dtype=float)
        # Add intercept
        X_b = np.hstack([np.ones((X_np.shape[0], 1)), X_np])
        beta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y_np
        model_func = lambda vec: float((np.hstack([1, vec]) @ beta))

    # Compute baseline mean (non‑held‑out)
    baseline_vals = [parse_float(row['Effort-']) for _, row in non_heldout if parse_float(row['Effort-']) is not None]
    baseline_mean = sum(baseline_vals) / len(baseline_vals) if baseline_vals else None

    # Predict on held‑out rows
    predictions = {}
    model_mae_total = 0.0
    baseline_mae_total = 0.0
    count = 0

    for idx, row in data:
        if idx not in heldout_set:
            continue
        actual = parse_float(row['Effort-'])
        if actual is None:
            continue
        # Predict using model if possible
        feat = []
        skip = False
        for c in feature_cols:
            v = parse_float(row[c])
            if v is None:
                skip = True
                break
            feat.append(v)
        if skip:
            pred = None
        else:
            pred = model_func(feat) if X else None
        predictions[idx] = round(pred, 2) if pred is not None else None

        if pred is not None:
            model_mae_total += abs(pred - actual)
        baseline_mae_total += abs(baseline_mean - actual)
        count += 1

    model_mae = model_mae_total / count if count else None
    baseline_mae = baseline_mae_total / count if count else None

    # Determine verdict
    if model_mae is None or baseline_mae is None:
        verdict = "no clear difference"
    elif abs(model_mae - baseline_mae) < 1e-6:
        verdict = "no clear difference"
    elif model_mae < baseline_mae:
        verdict = "better"
    else:
        verdict = "worse"

    # Print some summary
    print(f"Total rows: {len(data)}")
    print(f"Held‑out rows: {len(heldout_set)}")
    print(f"Model MAE: {model_mae:.2f}")
    print(f"Baseline MAE: {baseline_mae:.2f}")
    print(f"Verdict: {verdict}")

    # Machine‑checked answer
    preds_json = {str(k): v for k, v in predictions.items()}
    ans = {
        "predictions": preds_json,
        "baseline": round(baseline_mean, 2) if baseline_mean is not None else None,
        "mae_model": round(model_mae, 2) if model_mae is not None else None,
        "mae_baseline": round(baseline_mae, 2) if baseline_mae is not None else None,
        "verdict": verdict
    }
    print(f"ANSWER NASA-19 {json.dumps(ans, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# NASA-20 implementation
# ----------------------------------------------------------------------
def compute_goal_stats(fieldnames, data):
    """Return min and max for each goal column across all rows with non‑missing values."""
    mins = {}
    maxs = {}
    for col in fieldnames:
        if not is_goal(col):
            continue
        values = [parse_float(row[col]) for _, row in data if parse_float(row[col]) is not None]
        if values:
            mins[col] = min(values)
            maxs[col] = max(values)
    return mins, maxs

def score_row(row, mins, maxs):
    """Compute combined score for a row (higher is better)."""
    score = 0.0
    count = 0
    for col, mn, mx in zip(mins.keys(), mins.values(), maxs.values()):
        val = parse_float(row[col])
        if val is None:
            continue
        if mn == mx:
            norm = 0.5  # all equal, neutral
        else:
            if col.endswith('+'):
                norm = (val - mn) / (mx - mn)
            else:
                norm = (mx - val) / (mx - mn)
        score += norm
        count += 1
    return score / count if count else 0.0

def nasa_20(data, fieldnames):
    print("\n== NASA-20 ==")

    budget = min(30, len(data) // 5)
    repeats = 20
    random.seed(0)  # deterministic

    mins, maxs = compute_goal_stats(fieldnames, data)
    all_scores = {}
    for idx, row in data:
        all_scores[idx] = score_row(row, mins, maxs)

    # Our strategy: pick top budget rows by score
    sorted_rows = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    ours_selection = [idx for idx, _ in sorted_rows[:budget]]
    ours_best = sorted_rows[0][0]  # best overall

    # Random strategies
    random_best_list = []
    for _ in range(repeats):
        sample = random.sample([idx for idx, _ in data], budget)
        best = max(sample, key=lambda i: all_scores[i])
        random_best_list.append(best)

    # Compute average score of random bests
    random_best_scores = [all_scores[best] for best in random_best_list]
    avg_random_score = sum(random_best_scores) / repeats

    ours_score = all_scores[ours_best]
    if abs(ours_score - avg_random_score) < 1e-6:
        verdict = "no clear difference"
    elif ours_score > avg_random_score:
        verdict = "ours better"
    else:
        verdict = "random better"

    # Print summary
    print(f"Budget: {budget}")
    print(f"Repeats: {repeats}")
    print(f"Ours best row: {ours_best} (score {ours_score:.4f})")
    print(f"Random best rows: {random_best_list[:5]} ...")
    print(f"Average random best score: {avg_random_score:.4f}")
    print(f"Verdict: {verdict}")

    # Machine‑checked answer
    ans = {
        "budget": budget,
        "repeats": repeats,
        "ours": [ours_best] * repeats,
        "random": random_best_list,
        "verdict": verdict
    }
    print(f"ANSWER NASA-20 {json.dumps(ans, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv path>")
        sys.exit(1)
    path = sys.argv[1]
    data, fieldnames = read_csv(path)
    nasa_19(data, fieldnames)
    nasa_20(data, fieldnames)

if __name__ == "__main__":
    main()
