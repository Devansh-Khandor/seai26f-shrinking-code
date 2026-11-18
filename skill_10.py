#!/usr/bin/env python3
"""
Skill 10: Comparative evaluation of vehicle‑selection and forecasting.

This module implements two CAR tasks for the used‑car data:

  * CAR‑19 – Compare our deterministic selection method with random selection
    when both have the same evaluation budget.
  * CAR‑20 – Predict fuel‑efficiency (Mpg+) for held‑out rows using a linear
    regression trained on the remaining data, and compare against the
    historical‑average baseline.

The script expects a single CSV file as its command‑line argument and prints
human‑readable diagnostics followed by a machine‑checked JSON answer for each
CAR.

Author: OpenAI ChatGPT
"""

import csv
import json
import math
import random
import sys
import statistics

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def read_csv(path):
    """Read CSV, return list of (row_index, dict_of_values)."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, line in enumerate(reader, start=1):
            parsed = {}
            for k, v in line.items():
                if v == '?' or v is None:
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = None
            rows.append((idx, parsed))
    return rows

def is_goal(col):
    return col.endswith('+') or col.endswith('-')

def is_input(col):
    return not is_goal(col) and not col.endswith('X')

def composite_score(rec, mins, maxs):
    """Composite score for a single record (None if any goal missing)."""
    if any(rec[col] is None for col in mins):
        return None
    score = 0.0
    for col in mins:
        denom = maxs[col] - mins[col]
        if denom == 0:
            continue
        if col.endswith('-'):          # minimize
            score += (maxs[col] - rec[col]) / denom
        else:                          # maximize
            score += (rec[col] - mins[col]) / denom
    return score

# ----------------------------------------------------------------------
# Linear algebra for regression (no numpy)
# ----------------------------------------------------------------------
def mat_transpose(A):
    return [[row[i] for row in A] for i in range(len(A[0]))]

def mat_mul(A, B):
    return [[sum(a*b for a,b in zip(row, col)) for col in zip(*B)] for row in A]

def mat_vec_mul(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def mat_identity(n):
    return [[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]

def mat_inv(A):
    n = len(A)
    AM = [row[:] for row in A]
    IM = mat_identity(n)
    for fd in range(n):
        pivot = AM[fd][fd]
        if abs(pivot) < 1e-12:
            swap = None
            for r in range(fd+1, n):
                if abs(AM[r][fd]) > 1e-12:
                    swap = r
                    break
            if swap is None:
                raise ValueError("Singular matrix")
            AM[fd], AM[swap] = AM[swap], AM[fd]
            IM[fd], IM[swap] = IM[swap], IM[fd]
            pivot = AM[fd][fd]
        for j in range(n):
            AM[fd][j] /= pivot
            IM[fd][j] /= pivot
        for i in range(n):
            if i == fd:
                continue
            factor = AM[i][fd]
            for j in range(n):
                AM[i][j] -= factor * AM[fd][j]
                IM[i][j] -= factor * IM[fd][j]
    return IM

# ----------------------------------------------------------------------
# Regression helpers
# ----------------------------------------------------------------------
def fit_linear_regression(X, y):
    """Return weights, inverse XTX, residual variance."""
    n = len(X)
    d = len(X[0])
    XT = mat_transpose(X)
    XtX = mat_mul(XT, X)
    inv_XtX = mat_inv(XtX)
    XtY = [sum(XT[i][j] * y[j] for j in range(n)) for i in range(d)]
    w = mat_vec_mul(inv_XtX, XtY)
    # predictions
    y_pred = [sum(X[i][j] * w[j] for j in range(d)) for i in range(n)]
    residuals = [y[i] - y_pred[i] for i in range(n)]
    sigma2 = sum(r*r for r in residuals) / max(1, n - d)
    return w, inv_XtX, sigma2

def predict_linear_regression(row, w, inv_XtX, sigma2):
    """Predict target for a single feature vector (incl intercept)."""
    y_hat = sum(row[i] * w[i] for i in range(len(w)))
    temp = mat_vec_mul(inv_XtX, row)
    scalar = sum(row[i] * temp[i] for i in range(len(row)))
    se = math.sqrt(sigma2 * (1 + scalar))
    # 95% CI not needed here
    return y_hat

# ----------------------------------------------------------------------
# CAR‑19 implementation
# ----------------------------------------------------------------------
def car19(rows, goal_cols, budget=30, repeats=20):
    # global mins/maxs for all rows
    mins = {}
    maxs = {}
    for col in goal_cols:
        vals = [rec[col] for _, rec in rows if rec[col] is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
        else:
            mins[col] = maxs[col] = 0.0

    ours_best_indices = []
    random_best_indices = []

    # Deterministic selection: first `budget` rows
    our_indices = [idx for idx, _ in rows[:budget]]
    our_scores = [composite_score(rec, mins, maxs) for _, rec in rows[:budget]]
    ours_best_indices = [our_indices[i] for i, s in enumerate(our_scores) if s == max(filter(None, our_scores))]

    # Random selection repeated
    rng = random.Random(42)  # fixed seed for reproducibility
    for _ in range(repeats):
        rand_indices = rng.sample([idx for idx, _ in rows], k=budget)
        rand_scores = []
        for idx in rand_indices:
            rec = next(rec for r, rec in rows if r == idx)
            rand_scores.append(composite_score(rec, mins, maxs))
        best_score = max(filter(None, rand_scores))
        best_idx = rand_indices[rand_scores.index(best_score)]
        random_best_indices.append(best_idx)

    # Determine verdict
    ours_score = max(filter(None, ours_best_indices))
    # Compute how many times ours better
    ours_better = 0
    random_better = 0
    for rb in random_best_indices:
        rec_rb = next(rec for r, rec in rows if r == rb)
        score_rb = composite_score(rec_rb, mins, maxs)
        if ours_score > score_rb:
            ours_better += 1
        elif score_rb > ours_score:
            random_better += 1
    if ours_better > random_better:
        verdict = "ours better"
    elif random_better > ours_better:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    return {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best_indices,
        "random": random_best_indices,
        "verdict": verdict
    }

# ----------------------------------------------------------------------
# CAR‑20 implementation
# ----------------------------------------------------------------------
def car20(rows, input_cols, goal_cols, heldout_pred="Mpg+"):
    # Identify training and held‑out rows
    train_rows = []
    heldout_rows = []
    for idx, rec in rows:
        if idx % 5 == 0:
            heldout_rows.append((idx, rec))
        else:
            train_rows.append((idx, rec))

    # Build training data
    X_train = []
    y_train = []
    for _, rec in train_rows:
        if any(rec[col] is None for col in input_cols + [heldout_pred]):
            continue
        X_train.append([1.0] + [rec[col] for col in input_cols])
        y_train.append(rec[heldout_pred])

    if not X_train:
        # No training data
        predictions = {}
        mae_model = None
        mae_baseline = None
        baseline = None
        verdict = "no clear difference"
        return {
            "predictions": predictions,
            "baseline": baseline,
            "mae_model": mae_model,
            "mae_baseline": mae_baseline,
            "verdict": verdict
        }

    w, inv_XtX, sigma2 = fit_linear_regression(X_train, y_train)

    # Baseline mean on training set
    baseline = statistics.mean(y_train)

    # Predictions for held‑out rows
    predictions = {}
    errors_model = []
    errors_baseline = []
    for idx, rec in heldout_rows:
        if any(rec[col] is None for col in input_cols):
            continue
        X_row = [1.0] + [rec[col] for col in input_cols]
        pred = predict_linear_regression(X_row, w, inv_XtX, sigma2)
        predictions[str(idx)] = round(pred, 4)
        actual = rec[heldout_pred]
        errors_model.append(abs(pred - actual))
        errors_baseline.append(abs(baseline - actual))

    mae_model = round(sum(errors_model) / len(errors_model), 4) if errors_model else None
    mae_baseline = round(sum(errors_baseline) / len(errors_baseline), 4) if errors_baseline else None

    if mae_model is None or mae_baseline is None:
        verdict = "no clear difference"
    elif mae_model < mae_baseline:
        verdict = "better"
    elif mae_model > mae_baseline:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    return {
        "predictions": predictions,
        "baseline": round(baseline, 4),
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict
    }

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path):
    rows = read_csv(csv_path)
    if not rows:
        print("No data rows found.")
        sys.exit(1)

    # Identify goal and input columns
    _, first_row = rows[0]
    goal_cols = [c for c in first_row if is_goal(c)]
    input_cols = [c for c in first_row if is_input(c)]

    # CAR-19
    print("== CAR-19 ==")
    car19_res = car19(rows, goal_cols, budget=30, repeats=20)
    # Human readable diagnostics
    print(f"Budget per approach: {car19_res['budget']}")
    print(f"Repeats performed: {car19_res['repeats']}")
    print(f"Best row from deterministic selection: {car19_res['ours'][0]}")
    print(f"Best rows from random selections: {car19_res['random'][:5]} ...")
    print(f"Verdict: {car19_res['verdict']}")
    print("ANSWER CAR-19 " + json.dumps(car19_res, separators=(',', ':')))

    # CAR-20
    print("\n== CAR-20 ==")
    car20_res = car20(rows, input_cols, goal_cols, heldout_pred="Mpg+")
    print(f"Baseline mean Mpg+: {car20_res['baseline']}")
    print(f"MAE (model): {car20_res['mae_model']}")
    print(f"MAE (baseline): {car20_res['mae_baseline']}")
    print(f"Verdict: {car20_res['verdict']}")
    print("ANSWER CAR-20 " + json.dumps(car20_res, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_10.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])
