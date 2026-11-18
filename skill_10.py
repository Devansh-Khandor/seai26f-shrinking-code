#!/usr/bin/env python3
"""
skill_10.py - CAR-19 and CAR-20: selection comparison and MPG prediction

Usage:
    python skill_10.py <csv_path>
"""

import sys
import csv
import json
import math
import random
import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# CSV reading
# ----------------------------------------------------------------------
def read_csv(path):
    """Read CSV and return list of dicts with numeric conversion.
    Missing values are set to None."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Goal columns
            for col in ['Lbs-', 'Acc+', 'Mpg+']:
                val = row[col]
                if val == '?':
                    row[col] = None
                else:
                    try:
                        row[col] = float(val)
                    except ValueError:
                        row[col] = None
            # Feature columns
            for col in ['Clndrs', 'Volume', 'HpX', 'Model', 'origin']:
                val = row[col]
                if val == '?':
                    row[col] = None
                else:
                    try:
                        row[col] = float(val)
                    except ValueError:
                        row[col] = None
            rows.append(row)
    return rows

# ----------------------------------------------------------------------
# Composite score (same as skill_07)
# ----------------------------------------------------------------------
GOAL_COLUMNS = ['Lbs-', 'Acc+', 'Mpg+']

def compute_minmax(rows):
    mins = {}
    maxs = {}
    for col in GOAL_COLUMNS:
        vals = [r[col] for r in rows if r[col] is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
        else:
            mins[col] = 0.0
            maxs[col] = 0.0
    return mins, maxs

def normalize(val, v_min, v_max, reverse=False):
    if v_max == v_min:
        return 0.5
    score = (val - v_min) / (v_max - v_min)
    return 1.0 - score if reverse else score

def composite_score(row, mins, maxs):
    return (
        normalize(row['Lbs-'], mins['Lbs-'], maxs['Lbs-'], reverse=True) +
        normalize(row['Acc+'], mins['Acc+'], maxs['Acc+'], reverse=False) +
        normalize(row['Mpg+'], mins['Mpg+'], maxs['Mpg+'], reverse=False)
    )

# ----------------------------------------------------------------------
# CAR-19: Selection comparison
# ----------------------------------------------------------------------
def car19(rows, budget=30, repeats=20):
    # Use only rows that have complete goal data
    eval_rows = [r for r in rows if all(r[c] is not None for c in GOAL_COLUMNS)]
    if not eval_rows:
        return {"budget": budget, "repeats": repeats, "ours": [], "random": [], "verdict": "no data"}

    mins, maxs = compute_minmax(eval_rows)
    # Pre‑compute composite scores
    scores = {idx+1: composite_score(r, mins, maxs) for idx, r in enumerate(eval_rows)}

    # Our method: pick top budget rows by composite score
    top_budget = sorted(scores.items(), key=lambda x: -x[1])[:budget]
    our_best_idx = top_budget[0][0]
    our_best_score = top_budget[0][1]

    ours_list = [our_best_idx] * repeats  # deterministic

    # Random method
    random_best_indices = []
    random_best_scores = []
    for rep in range(repeats):
        rng = random.Random(12345 + rep)
        chosen = rng.sample(list(scores.keys()), min(budget, len(scores)))
        chosen_scores = [(idx, scores[idx]) for idx in chosen]
        best_idx, best_score = max(chosen_scores, key=lambda x: x[1])
        random_best_indices.append(best_idx)
        random_best_scores.append(best_score)

    # Compute verdict
    ours_avg = our_best_score  # constant
    random_avg = sum(random_best_scores) / len(random_best_scores) if random_best_scores else 0.0
    eps = 1e-6
    if ours_avg > random_avg + eps:
        verdict = "ours better"
    elif random_avg > ours_avg + eps:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    result = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_list,
        "random": random_best_indices,
        "verdict": verdict
    }
    return result

# ----------------------------------------------------------------------
# CAR-20: MPG prediction vs baseline
# ----------------------------------------------------------------------
def car20(rows):
    # Identify held‑out rows: divisible by 5
    held_out_indices = [i+1 for i in range(len(rows)) if (i+1) % 5 == 0]
    # Rows for training: all but held-out
    train_rows = [r for i, r in enumerate(rows) if (i+1) not in held_out_indices and r['Mpg+'] is not None]

    if not train_rows:
        return {"predictions": {}, "baseline": None, "mae_model": None, "mae_baseline": None, "verdict": "no data"}

    # Train linear regression
    X_train = np.array([[r['Clndrs'], r['Volume'], r['HpX'], r['Model'], r['origin']] for r in train_rows])
    y_train = np.array([r['Mpg+'] for r in train_rows])

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Baseline: mean MPG of training data
    baseline = float(np.mean(y_train))

    # Predictions and MAEs
    predictions = {}
    model_errors = []
    baseline_errors = []

    for idx, r in enumerate(rows, start=1):
        if idx in held_out_indices and r['Mpg+'] is not None:
            X = np.array([[r['Clndrs'], r['Volume'], r['HpX'], r['Model'], r['origin']]])
            pred = float(model.predict(X)[0])
            predictions[str(idx)] = round(pred, 6)
            model_errors.append(abs(pred - r['Mpg+']))
            baseline_errors.append(abs(baseline - r['Mpg+']))

    mae_model = float(np.mean(model_errors)) if model_errors else None
    mae_baseline = float(np.mean(baseline_errors)) if baseline_errors else None

    # Verdict
    eps = 1e-6
    if mae_model is None or mae_baseline is None:
        verdict = "no data"
    elif mae_model < mae_baseline - eps:
        verdict = "better"
    elif mae_baseline < mae_model - eps:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    result = {
        "predictions": predictions,
        "baseline": baseline,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict
    }
    return result

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_csv(csv_path)

    # CAR-19
    print("== CAR-19 ==")
    car19_res = car19(rows, budget=30, repeats=20)
    # Human friendly lines
    print(f"Budget: {car19_res['budget']} rows")
    print(f"Best row by our system: {car19_res['ours'][0]} (score computed internally)")
    print(f"Random best rows over repeats: {car19_res['random']}")
    print(f"Verdict: {car19_res['verdict']}")
    print("ANSWER CAR-19", json.dumps(car19_res, separators=(',', ':')))

    # CAR-20
    print("\n== CAR-20 ==")
    car20_res = car20(rows)
    print(f"Baseline mean Mpg+: {car20_res['baseline']}")
    print(f"MAE model: {car20_res['mae_model']}")
    print(f"MAE baseline: {car20_res['mae_baseline']}")
    print(f"Verdict: {car20_res['verdict']}")
    print("ANSWER CAR-20", json.dumps(car20_res, separators=(',', ':')))

if __name__ == "__main__":
    main()
