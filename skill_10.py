#!/usr/bin/env python3
"""
skill_10.py

Implements APACHE-19 and APACHE-20 analyses.

Usage:
    python skill_10.py <csv_path>
"""

import sys
import csv
import json
import random
import math
from collections import defaultdict

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, reader.fieldnames

def to_float(val):
    if val is None or val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def is_goal(col):
    return col.endswith('+') or col.endswith('-')

def is_setting(col, goal_cols):
    return col not in goal_cols and not col.endswith('X') and not is_goal(col)

def solve_linear(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        # pivot
        max_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[max_row][i]) < 1e-12:
            return None
        M[i], M[max_row] = M[max_row], M[i]
        pivot = M[i][i]
        for j in range(i, n+1):
            M[i][j] /= pivot
        for r in range(i+1, n):
            factor = M[r][i]
            for j in range(i, n+1):
                M[r][j] -= factor * M[i][j]
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        x[i] = M[i][n]
        for j in range(i+1, n):
            x[i] -= M[i][j] * x[j]
    return x

def fit_linear_regression(rows, goal_col, settings):
    X = []
    y = []
    for row in rows:
        g = to_float(row[goal_col])
        if g is None:
            continue
        vals = [1.0]  # intercept
        ok = True
        for s in settings:
            v = to_float(row[s])
            if v is None:
                ok = False
                break
            vals.append(v)
        if not ok:
            continue
        X.append(vals)
        y.append(g)
    if not X:
        return None, None
    p = len(X[0])
    XtX = [[0.0]*p for _ in range(p)]
    Xty = [0.0]*p
    for i in range(len(X)):
        xi = X[i]
        yi = y[i]
        for a in range(p):
            Xty[a] += xi[a] * yi
            for b in range(a, p):
                XtX[a][b] += xi[a] * xi[b]
    for a in range(p):
        for b in range(a+1, p):
            XtX[b][a] = XtX[a][b]
    coeffs = solve_linear(XtX, Xty)
    if coeffs is None:
        return None, None
    return coeffs[0], coeffs[1:]  # intercept and betas

def predict_row(row, goal_col, settings, intercept, betas):
    if intercept is None or betas is None:
        return None
    val = intercept
    for i, s in enumerate(settings):
        v = to_float(row[s])
        if v is None:
            return None
        val += betas[i] * v
    return val

# ----------------------------------------------------------------------
# Ranking helpers for multi-goal comparison
# ----------------------------------------------------------------------
def compute_ranks(rows, goal_cols):
    ranks = {col: {} for col in goal_cols}
    for col in goal_cols:
        vals = [(to_float(r[col]), idx+1) for idx, r in enumerate(rows) if to_float(r[col]) is not None]
        if not vals:
            continue
        # Determine sort order
        minimize = col.endswith('-')
        vals.sort(key=lambda x: x[0], reverse=not minimize)
        # Assign ranks
        rank = 1
        last_val = None
        for value, idx in vals:
            if last_val is not None and value != last_val:
                rank += 1
            ranks[col][idx] = rank
            last_val = value
    return ranks

def composite_score(row_idx, ranks, goal_cols):
    total = 0
    for col in goal_cols:
        total += ranks.get(col, {}).get(row_idx, 0)
    return total

def best_row_by_composite(rows, goal_cols, ranks):
    best_idx = None
    best_score = None
    for idx in range(1, len(rows)+1):
        score = composite_score(idx, ranks, goal_cols)
        if best_score is None or score < best_score:
            best_score = score
            best_idx = idx
    return best_idx

# ----------------------------------------------------------------------
# APACHE-19 implementation
# ----------------------------------------------------------------------
def apache_19(rows, goal_cols, setting_cols):
    n_rows = len(rows)
    budget = min(30, n_rows // 5)
    repeats = 20
    random.seed(42)  # deterministic

    # Precompute ranks for all rows
    ranks = compute_ranks(rows, goal_cols)

    ours_best = []
    random_best = []

    for _ in range(repeats):
        # Our method
        evaluated = []
        # Start with first 5 rows that have complete data
        init = 0
        for i in range(1, n_rows+1):
            if len(evaluated) >= 5:
                break
            row = rows[i-1]
            if to_float(row[goal_cols[0]]) is None:
                continue
            # ensure all settings numeric for regression later
            ok = True
            for s in setting_cols:
                if to_float(row[s]) is None:
                    ok = False
                    break
            if not ok:
                continue
            evaluated.append(i)

        while len(evaluated) < budget:
            # fit regression on evaluated rows
            coeffs = fit_linear_regression([rows[idx-1] for idx in evaluated], goal_cols[0], setting_cols)
            intercept, betas = coeffs
            # pick next row with lowest predicted value
            best_pred = None
            best_idx = None
            for i in range(1, n_rows+1):
                if i in evaluated:
                    continue
                row = rows[i-1]
                if to_float(row[goal_cols[0]]) is None:
                    continue
                pred = predict_row(row, goal_cols[0], setting_cols, intercept, betas) if intercept is not None else None
                if pred is None:
                    continue
                if best_pred is None or pred < best_pred:
                    best_pred = pred
                    best_idx = i
            if best_idx is None:
                # fallback: choose first un-evaluated with goal
                for i in range(1, n_rows+1):
                    if i in evaluated:
                        continue
                    if to_float(rows[i-1][goal_cols[0]]) is not None:
                        best_idx = i
                        break
                if best_idx is None:
                    break
            evaluated.append(best_idx)

        # find best row among evaluated
        best_idx = best_row_by_composite([rows[i-1] for i in evaluated], goal_cols, {col: {i: ranks[col][i] for i in evaluated} if col in ranks else {} for col in goal_cols})
        ours_best.append(best_idx)

        # Random method
        candidates = [i for i in range(1, n_rows+1) if to_float(rows[i-1][goal_cols[0]]) is not None]
        random_samples = random.sample(candidates, min(budget, len(candidates)))
        best_idx_rand = best_row_by_composite([rows[i-1] for i in random_samples], goal_cols, {col: {i: ranks[col][i] for i in random_samples} if col in ranks else {} for col in goal_cols})
        random_best.append(best_idx_rand)

    # Verdict
    ours_wins = sum(1 for o, r in zip(ours_best, random_best) if o < r)
    random_wins = sum(1 for o, r in zip(ours_best, random_best) if o > r)
    if ours_wins > random_wins:
        verdict = "ours better"
    elif random_wins > ours_wins:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    answer = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best,
        "random": random_best,
        "verdict": verdict
    }
    return answer

# ----------------------------------------------------------------------
# APACHE-20 implementation
# ----------------------------------------------------------------------
def apache_20(rows, goal_col, setting_cols):
    # Identify held-out rows: divisible by 5
    held_out = [i+1 for i in range(len(rows)) if (i+1) % 5 == 0]
    # Non held-out rows
    train_rows = [rows[i] for i in range(len(rows)) if (i+1) not in held_out]
    # Baseline mean
    train_vals = [to_float(r[goal_col]) for r in train_rows if to_float(r[goal_col]) is not None]
    baseline = sum(train_vals) / len(train_vals) if train_vals else 0.0

    # Fit model on train rows
    intercept, betas = fit_linear_regression(train_rows, goal_col, setting_cols)

    predictions = {}
    mae_model = 0.0
    mae_baseline = 0.0
    count = 0
    for idx in held_out:
        row = rows[idx-1]
        true_val = to_float(row[goal_col])
        if true_val is None:
            continue
        pred = predict_row(row, goal_col, setting_cols, intercept, betas)
        if pred is None:
            pred = baseline
        predictions[str(idx)] = round(pred, 4)
        mae_model += abs(pred - true_val)
        mae_baseline += abs(baseline - true_val)
        count += 1
    mae_model = mae_model / count if count else 0.0
    mae_baseline = mae_baseline / count if count else 0.0

    if mae_model < mae_baseline:
        verdict = "better"
    elif mae_model > mae_baseline:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    answer = {
        "predictions": predictions,
        "baseline": round(baseline, 4),
        "mae_model": round(mae_model, 4),
        "mae_baseline": round(mae_baseline, 4),
        "verdict": verdict
    }
    return answer

# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main(csv_path):
    rows, columns = read_csv(csv_path)
    if not rows:
        print("No data rows found.")
        sys.exit(1)

    goal_cols = [c for c in columns if is_goal(c)]
    if not goal_cols:
        print("No goal column found.")
        sys.exit(1)
    setting_cols = [c for c in columns if is_setting(c, goal_cols)]

    # APACHE-19
    print("== APACHE-19 ==")
    answer19 = apache_19(rows, goal_cols, setting_cols)
    print(f"Budget: {answer19['budget']}")
    print(f"Repeats: {answer19['repeats']}")
    print(f"Our best rows: {answer19['ours']}")
    print(f"Random best rows: {answer19['random']}")
    print(f"Verdict: {answer19['verdict']}")
    print("ANSWER APACHE-19 " + json.dumps(answer19, separators=(',', ':')))

    # APACHE-20
    print("\n== APACHE-20 ==")
    answer20 = apache_20(rows, goal_cols[0], setting_cols)
    print(f"Baseline mean: {answer20['baseline']}")
    print(f"MAE model: {answer20['mae_model']}")
    print(f"MAE baseline: {answer20['mae_baseline']}")
    print(f"Verdict: {answer20['verdict']}")
    print("ANSWER APACHE-20 " + json.dumps(answer20, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
