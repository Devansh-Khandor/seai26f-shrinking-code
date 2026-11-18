#!/usr/bin/env python3
"""
skill_10.py

Implements APACHE-19 and APACHE-20:
- Compare our configuration-selection system vs random under the same budget.
- Compare regression predictions vs historical‑average baseline on held‑out rows.
"""

import csv
import json
import random
import math
import sys

# ----------------------------------------------------------------------
# Helper imports from previous skills
# ----------------------------------------------------------------------
try:
    from skill_07 import build_regression, predict, to_float
except Exception:
    # Minimal fall‑back if skill_07 not importable
    def load_csv(path):
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            data = []
            for row in reader:
                cleaned = {k: (None if v == '?' else v) for k, v in row.items()}
                data.append(cleaned)
        return data, headers

    def to_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    # Simple linear regression
    def _transpose(m):
        return [list(r) for r in zip(*m)]

    def _matmul(A, B):
        m, n = len(A), len(A[0])
        p = len(B[0])
        res = [[0.0] * p for _ in range(m)]
        for i in range(m):
            for k in range(n):
                aik = A[i][k]
                if aik == 0: continue
                for j in range(p):
                    res[i][j] += aik * B[k][j]
        return res

    def _solve(A, b):
        n = len(A)
        M = [row[:] + [b[i]] for i, row in enumerate(A)]
        for col in range(n):
            pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
            if abs(M[pivot][col]) < 1e-12:
                raise ValueError
            if pivot != col:
                M[col], M[pivot] = M[pivot], M[col]
            piv = M[col][col]
            for j in range(col, n+1):
                M[col][j] /= piv
            for r in range(n):
                if r == col: continue
                fac = M[r][col]
                if fac == 0: continue
                for j in range(col, n+1):
                    M[r][j] -= fac * M[col][j]
        return [M[i][-1] for i in range(n)]

    def build_regression(data, setting_cols, goal_col):
        X, Y = [], []
        for row in data:
            val = to_float(row.get(goal_col))
            if val is None: continue
            vlist = []
            skip = False
            for c in setting_cols:
                v = row.get(c)
                if v is None: skip=True; break
                if v not in ('0','1'):
                    try:
                        int(v)
                    except ValueError: skip=True; break
                vlist.append(1 if v=='1' else 0)
            if skip: continue
            X.append([1]+vlist)
            Y.append([val])
        if not X: return None
        Xt = _transpose(X)
        XtX = _matmul(Xt, X)
        XtY = _matmul(Xt, Y)
        try:
            coeffs = _solve(XtX, [r[0] for r in XtY])
        except Exception:
            return None
        return coeffs

    def predict(row, coeffs, setting_cols):
        feat = [1]
        for c in setting_cols:
            v = row.get(c)
            feat.append(1 if v=='1' else 0)
        return sum(c*f for c,f in zip(coeffs, feat))

# ----------------------------------------------------------------------
# Load CSV
# ----------------------------------------------------------------------
def load_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        data = []
        for row in reader:
            cleaned = {k: (None if v == '?' else v) for k, v in row.items()}
            data.append(cleaned)
    return data, headers

# ----------------------------------------------------------------------
# Utility: compute overall score for a row across all goal columns
# ----------------------------------------------------------------------
def compute_score(row, goal_cols):
    score = 0.0
    for col in goal_cols:
        val = row.get(col)
        if val is None:
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        if col.endswith('+'):
            score += v
        else:  # '-'
            score -= v
    return score

# ----------------------------------------------------------------------
# APACHE-19: compare selection system vs random
# ----------------------------------------------------------------------
def apache_19(data, headers, budget=30, repeats=20):
    goal_cols = [h for h in headers if h.endswith('+') or h.endswith('-')]
    setting_cols = [h for h in headers if h not in goal_cols and not h.endswith('X')]

    ours_best_rows = []
    random_best_rows = []

    for r in range(repeats):
        random.seed(42 + r)

        # --- selection system (greedy regression) ---
        # Start with empty evaluated set
        evaluated = []
        evaluated_set = set()
        # Use same logic as skill_07 APACHE-13 but with budget rows
        while len(evaluated) < budget:
            if not evaluated:
                next_idx = 0
            else:
                train_rows = [data[i] for i in evaluated]
                coeffs = build_regression(train_rows, setting_cols, goal_cols[0])
                if coeffs is None:
                    remaining = [i for i in range(len(data)) if i not in evaluated_set]
                    next_idx = remaining[0] if remaining else None
                else:
                    remaining = [i for i in range(len(data)) if i not in evaluated_set]
                    preds = [(i, predict(data[i], coeffs, setting_cols)) for i in remaining]
                    next_idx = min(preds, key=lambda x: x[1])[0]
            if next_idx is None:
                break
            evaluated.append(next_idx)
            evaluated_set.add(next_idx)

        # best row among all rows according to all goals
        best_ours_idx = max(range(len(data)), key=lambda i: compute_score(data[i], goal_cols))
        ours_best_rows.append(best_ours_idx + 1)

        # --- random selection ---
        sample_indices = random.sample(range(len(data)), budget)
        best_rand_idx = max(sample_indices, key=lambda i: compute_score(data[i], goal_cols))
        random_best_rows.append(best_rand_idx + 1)

    # Determine verdict
    ours_wins = 0
    random_wins = 0
    for i in range(repeats):
        ours_score = compute_score(data[ours_best_rows[i]-1], goal_cols)
        rand_score = compute_score(data[random_best_rows[i]-1], goal_cols)
        if ours_score > rand_score:
            ours_wins += 1
        elif ours_score < rand_score:
            random_wins += 1
    if ours_wins > random_wins:
        verdict = "ours better"
    elif ours_wins < random_wins:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    # Output
    print("== APACHE-19 ==")
    print(f"Budget: {budget}")
    print(f"Repeats: {repeats}")
    print(f"Best rows (ours): {ours_best_rows[:3]} ...")
    print(f"Best rows (random): {random_best_rows[:3]} ...")

    answer = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best_rows,
        "random": random_best_rows,
        "verdict": verdict
    }
    print("ANSWER APACHE-19 " + json.dumps(answer, separators=(',', ':')))

# ----------------------------------------------------------------------
# APACHE-20: predictions vs baseline
# ----------------------------------------------------------------------
def apache_20(data, headers):
    goal_cols = [h for h in headers if h.endswith('+') or h.endswith('-')]
    perf_col = goal_cols[0]
    setting_cols = [h for h in headers if h not in goal_cols and not h.endswith('X')]

    # Identify held‑out rows (divisible by 5)
    held_out_indices = [i for i in range(len(data)) if (i+1) % 5 == 0]
    non_held_indices = [i for i in range(len(data)) if i not in held_out_indices]

    # Build regression on non‑held rows
    training_rows = [data[i] for i in non_held_indices]
    coeffs = build_regression(training_rows, setting_cols, perf_col)

    # Baseline mean of non‑held performance
    baseline_vals = [to_float(data[i].get(perf_col)) for i in non_held_indices]
    baseline_vals = [v for v in baseline_vals if v is not None]
    if baseline_vals:
        baseline_mean = sum(baseline_vals) / len(baseline_vals)
    else:
        baseline_mean = None

    predictions = {}
    mae_model = 0.0
    mae_baseline = 0.0
    count = 0

    for idx in held_out_indices:
        actual = to_float(data[idx].get(perf_col))
        if actual is None:
            continue
        pred = predict(data[idx], coeffs, setting_cols) if coeffs is not None else baseline_mean
        predictions[str(idx+1)] = round(pred, 2) if pred is not None else None
        mae_model += abs(pred - actual) if pred is not None else abs(baseline_mean - actual)
        mae_baseline += abs(baseline_mean - actual)
        count += 1

    if count:
        mae_model /= count
        mae_baseline /= count
    else:
        mae_model = None
        mae_baseline = None

    if mae_model is not None and mae_baseline is not None:
        if mae_model < mae_baseline:
            verdict = "better"
        elif mae_model > mae_baseline:
            verdict = "worse"
        else:
            verdict = "no clear difference"
    else:
        verdict = "no clear difference"

    # Output
    print("== APACHE-20 ==")
    print(f"Baseline mean: {baseline_mean:.2f}" if baseline_mean is not None else "Baseline mean: None")
    print(f"MAE model: {mae_model:.2f}" if mae_model is not None else "MAE model: None")
    print(f"MAE baseline: {mae_baseline:.2f}" if mae_baseline is not None else "MAE baseline: None")
    print(f"First 3 predictions: {list(predictions.items())[:3]}")

    answer = {
        "predictions": predictions,
        "baseline": baseline_mean,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict
    }
    print("ANSWER APACHE-20 " + json.dumps(answer, separators=(',', ':')))

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    data, headers = load_csv(csv_path)

    apache_19(data, headers)
    apache_20(data, headers)

if __name__ == "__main__":
    main()
