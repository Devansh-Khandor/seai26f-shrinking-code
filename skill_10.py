#!/usr/bin/env python3
"""
skill_10.py <csv_path>

Implements NASA-19 and NASA-20.

NASA-19: Compare a regression‑based effort estimator against a simple
historical‑average baseline and report whether it gives better
prediction quality on the held‑out rows.

NASA-20: Compare our composite‑score based project selection to
random selection under a fixed evaluation budget, and report
whether our method tends to pick better projects.
"""

import sys
import csv
import json
import math
import random
from statistics import mean

# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------
def parse_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def to_float(val):
    if val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

# ------------------------------------------------------------------
# Regression helpers (single predictor Kloc+)
# ------------------------------------------------------------------
def build_single_regression(rows):
    """Return intercept and slope for y = a + b * x on non‑held‑out rows."""
    X = []
    Y = []
    for r in rows:
        k = to_float(r["Kloc+"])
        e = to_float(r["Effort-"])
        if k is None or e is None:
            continue
        X.append(k)
        Y.append(e)
    n = len(X)
    if n == 0:
        return 0.0, 0.0
    x_mean = sum(X) / n
    y_mean = sum(Y) / n
    num = sum((X[i] - x_mean) * (Y[i] - y_mean) for i in range(n))
    den = sum((X[i] - x_mean) ** 2 for i in range(n))
    b = num / den if den != 0 else 0.0
    a = y_mean - b * x_mean
    return a, b

def predict_effort(a, b, k):
    if k is None:
        return None
    return a + b * k

# ------------------------------------------------------------------
# NASA‑19
# ------------------------------------------------------------------
def nasa_19(rows):
    total = len(rows)
    held_out = {i for i in range(1, total + 1) if i % 5 == 0}
    non_held = [r for i, r in enumerate(rows, start=1) if i not in held_out]

    # Model
    a, b = build_single_regression(non_held)

    # Predictions on held‑out rows
    predictions = {}
    mae_model = 0.0
    mae_baseline = 0.0
    baseline_vals = []
    for idx, r in enumerate(rows, start=1):
        e_true = to_float(r["Effort-"])
        if e_true is None:
            continue
        if idx in held_out:
            k = to_float(r["Kloc+"])
            pred = predict_effort(a, b, k)
            predictions[str(idx)] = pred if pred is not None else None
            mae_model += abs(e_true - pred) if pred is not None else 0.0
            baseline_vals.append(e_true)
        else:
            baseline_vals.append(e_true)

    n_hold = len([i for i in range(1, total + 1) if i % 5 == 0])
    if n_hold:
        mae_model /= n_hold
    else:
        mae_model = float('nan')
    if baseline_vals:
        baseline_mean = mean(baseline_vals)
    else:
        baseline_mean = float('nan')
    if n_hold:
        mae_baseline = sum(abs(to_float(rows[i-1]["Effort-"]) - baseline_mean)
                           for i in range(1, total + 1) if i % 5 == 0) / n_hold
    else:
        mae_baseline = float('nan')

    # Verdict
    if mae_model < mae_baseline:
        verdict = "better"
    elif mae_model > mae_baseline:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    # Output
    print("== NASA-19 ==")
    print(f"Model MAE: {mae_model:.4f}, Baseline MAE: {mae_baseline:.4f}")
    print(f"Verdict: {verdict}")
    ans = {
        "predictions": predictions,
        "baseline": baseline_mean,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict
    }
    print("ANSWER NASA-19", json.dumps(ans, separators=(',', ':')))

# ------------------------------------------------------------------
# Composite score helpers
# ------------------------------------------------------------------
def composite_score(r):
    """Return Kloc+ - Effort- - Defects-; rows with missing goal values
    get a very negative score so they are unlikely to be selected."""
    k = to_float(r["Kloc+"])
    e = to_float(r["Effort-"])
    d = to_float(r["Defects-"])
    if None in (k, e, d):
        return -float('inf')
    return k - e - d

def best_row_from_set(rows, indices):
    """Return index of best row among given indices using Effort- then Defects-."""
    best = None
    best_eff = float('inf')
    best_def = float('inf')
    for idx in indices:
        r = rows[idx-1]
        e = to_float(r["Effort-"])
        d = to_float(r["Defects-"])
        if e is None or d is None:
            continue
        # lower is better
        if e < best_eff or (e == best_eff and d < best_def):
            best = idx
            best_eff = e
            best_def = d
    return best

# ------------------------------------------------------------------
# NASA‑20
# ------------------------------------------------------------------
def nasa_20(rows):
    total = len(rows)
    budget = min(30, total // 5)
    if budget <= 0:
        print("== NASA-20 ==")
        print("Insufficient rows for evaluation budget.")
        ans = {"budget": budget, "repeats": 0, "ours": [], "random": [], "verdict": "no clear difference"}
        print("ANSWER NASA-20", json.dumps(ans, separators=(',', ':')))
        return

    random.seed(42)  # deterministic
    repeats = 20
    ours_best = []
    rand_best = []

    all_indices = list(range(1, total + 1))
    for _ in range(repeats):
        # Our selection: top budget by composite score
        scored = [(composite_score(r), idx) for idx, r in enumerate(rows, start=1)]
        scored.sort(key=lambda x: (-x[0], x[1]))
        ours = [idx for _, idx in scored[:budget]]
        best_ours = best_row_from_set(rows, ours) or ours[0]
        ours_best.append(best_ours)

        # Random selection
        rand = random.sample(all_indices, budget)
        best_rand = best_row_from_set(rows, rand) or rand[0]
        rand_best.append(best_rand)

    # Verdict
    ours_wins = sum(1 for a, b in zip(ours_best, rand_best) if a != None and b != None and a < b)
    rand_wins = sum(1 for a, b in zip(ours_best, rand_best) if a != None and b != None and a > b)
    if ours_wins > rand_wins:
        verdict = "ours better"
    elif rand_wins > ours_wins:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    # Output
    print("== NASA-20 ==")
    print(f"Budget: {budget}, Repeats: {repeats}")
    print(f"Ours best rows: {ours_best}")
    print(f"Random best rows: {rand_best}")
    print(f"Verdict: {verdict}")
    ans = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best,
        "random": rand_best,
        "verdict": verdict
    }
    print("ANSWER NASA-20", json.dumps(ans, separators=(',', ':')))

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_10.py <csv_path>\n")
        sys.exit(1)
    path = sys.argv[1]
    rows = parse_csv(path)
    nasa_19(rows)
    print()
    nasa_20(rows)

if __name__ == "__main__":
    main()
