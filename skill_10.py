#!/usr/bin/env python3
"""
skill_10.py

Implements APACHE-19 and APACHE-20 analyses for the Apache performance dataset.

Usage:
    python skill_10.py <csv_path>
"""

import csv
import json
import math
import random
import sys

# --------------------------------------------------------------------------- #
# Utility functions for CSV and data

def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    return header, rows

def parse_int(val):
    if val is None or val == '?':
        return None
    try:
        return int(val)
    except ValueError:
        return None

def is_goal(col):
    return col.endswith('+') or col.endswith('-')

def is_input(col):
    return not col.endswith('X') and not is_goal(col)

# --------------------------------------------------------------------------- #
# Basic matrix helpers (no external libraries)

def transpose(mat):
    return list(map(list, zip(*mat)))

def mat_mult(A, B):
    """Multiply matrices A (n x m) and B (m x p) -> (n x p)."""
    result = [[0.0] * len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for k in range(len(B)):
            aik = A[i][k]
            if aik == 0:
                continue
            for j in range(len(B[0])):
                result[i][j] += aik * B[k][j]
    return result

def solve_linear_system(A, b):
    """
    Solve Ax = b for x using Gaussian elimination with partial pivoting.
    A is a square matrix, b is a column vector.
    Returns list of solutions.
    """
    n = len(A)
    # Create augmented matrix
    M = [A[i][:] + [b[i][0]] for i in range(n)]

    for k in range(n):
        # Pivot
        piv = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[piv][k]) < 1e-12:
            M[piv][k] += 1e-12
        if piv != k:
            M[k], M[piv] = M[piv], M[k]

        # Normalize pivot row
        div = M[k][k]
        for j in range(k, n + 1):
            M[k][j] /= div

        # Eliminate below
        for i in range(n):
            if i == k:
                continue
            factor = M[i][k]
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]

    # Extract solutions
    x = [row[-1] for row in M]
    return x

# --------------------------------------------------------------------------- #
# APACHE-19: configuration selection vs random

def score_row(row, goal_cols):
    """Compute a scalar score for a row based on all goal columns."""
    score = 0.0
    for c in goal_cols:
        v = parse_int(row[c])
        if v is None:
            continue
        if c.endswith('+'):
            score += v
        else:
            score -= v
    return score

def apache_19(header, rows):
    goal_cols = [c for c in header if is_goal(c)]
    n_rows = len(rows)
    budget = min(30, n_rows // 5)

    # Our selection: 30 best-scoring rows
    scored = [(idx+1, score_row(r, goal_cols)) for idx, r in enumerate(rows)]
    scored.sort(key=lambda x: (-x[1], x[0]))
    our_selected = [idx for idx, _ in scored[:budget]]
    our_best = max(our_selected, key=lambda idx: scored[idx-1][1]) if our_selected else None

    # Random repeats
    repeats = 20
    our_bests = []
    random_bests = []

    for _ in range(repeats):
        # Our best in this repeat is deterministic (always same)
        our_bests.append(our_best)

        rand_indices = random.sample(range(1, n_rows+1), budget)
        rand_scores = [(idx, score_row(rows[idx-1], goal_cols)) for idx in rand_indices]
        rand_best = max(rand_scores, key=lambda x: (x[1], -x[0]))[0]
        random_bests.append(rand_best)

    # Verdict
    ours_wins = sum(1 for i, r in enumerate(random_bests) if score_row(rows[ours_bests[i]-1], goal_cols) > score_row(rows[r-1], goal_cols))
    random_wins = sum(1 for i, r in enumerate(random_bests) if score_row(rows[ours_bests[i]-1], goal_cols) < score_row(rows[r-1], goal_cols))
    if ours_wins > random_wins:
        verdict = "ours better"
    elif random_wins > ours_wins:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    # Output
    print("== APACHE-19 ==")
    print(f"Budget: {budget}")
    print(f"Repeats: {repeats}")
    print(f"Best row by our method (first repeat): {our_bests[0]}")
    print(f"Best row by random (first repeat): {random_bests[0]}")
    # Machine-checked answer
    ans = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_bests,
        "random": random_bests,
        "verdict": verdict
    }
    print("ANSWER APACHE-19", json.dumps(ans))

# --------------------------------------------------------------------------- #
# APACHE-20: predictions vs baseline

def compute_impute_means(rows, input_cols):
    means = {}
    for c in input_cols:
        total, count = 0, 0
        for r in rows:
            v = parse_int(r[c])
            if v is not None:
                total += v
                count += 1
        means[c] = (total / count) if count else 0
    return means

def build_regression(rows, feature_cols, target_col, impute_means):
    X = []
    y = []
    for r in rows:
        tgt = parse_int(r[target_col])
        if tgt is None:
            continue
        feats = [1]  # intercept
        for c in feature_cols:
            v = parse_int(r[c])
            if v is None:
                v = impute_means[c]
            feats.append(v)
        X.append(feats)
        y.append([tgt])
    X_T = transpose(X)
    XtX = mat_mult(X_T, X)
    XtY = mat_mult(X_T, y)
    beta = solve_linear_system(XtX, XtY)
    return beta

def predict(beta, impute_means, row, feature_cols):
    feat = [1]
    for c in feature_cols:
        v = parse_int(row[c])
        if v is None:
            v = impute_means[c]
        feat.append(v)
    val = sum(f * b for f, b in zip(feat, beta))
    return val

def apache_20(header, rows):
    goal_cols = [c for c in header if is_goal(c)]
    # we only care about the first minimizing goal (Performance-)
    perf_col = next((c for c in goal_cols if c.endswith('-')), None)
    if perf_col is None:
        print("No minimizing goal column found.")
        return

    input_cols = [c for c in header if is_input(c)]

    # Identify held-out rows (index divisible by 5)
    held_out_indices = [i for i, r in enumerate(rows) if (i+1) % 5 == 0]
    train_rows = [r for i, r in enumerate(rows) if i not in held_out_indices]
    test_rows = [rows[i] for i in held_out_indices]

    impute_means = compute_impute_means(train_rows, input_cols)

    beta = build_regression(train_rows, input_cols, perf_col, impute_means)

    predictions = {}
    abs_errors = []

    for idx, r in zip(held_out_indices, test_rows):
        pred_val = predict(beta, impute_means, r, input_cols)
        pred_int = int(round(pred_val))
        predictions[str(idx+1)] = pred_int
        actual = parse_int(r[perf_col])
        if actual is not None:
            abs_errors.append(abs(pred_int - actual))

    mae_model = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0

    # Baseline: mean of non-held-out performances
    baseline_vals = [parse_int(r[perf_col]) for r in train_rows]
    baseline_vals = [v for v in baseline_vals if v is not None]
    baseline_mean = sum(baseline_vals) / len(baseline_vals) if baseline_vals else 0.0
    mae_baseline = sum(abs(baseline_mean - v) for v in baseline_vals) / len(baseline_vals) if baseline_vals else 0.0

    verdict = "better" if mae_model < mae_baseline else ("worse" if mae_model > mae_baseline else "no clear difference")

    # Output
    print("== APACHE-20 ==")
    for row_num, pred in sorted(predictions.items(), key=lambda x: int(x[0])):
        print(f"Row {row_num} prediction: {pred}")
    print(f"Baseline mean performance: {baseline_mean:.2f}")
    print(f"MAE model: {mae_model:.2f}")
    print(f"MAE baseline: {mae_baseline:.2f}")
    # Machine-checked answer
    ans = {
        "predictions": predictions,
        "baseline": round(baseline_mean, 3),
        "mae_model": round(mae_model, 3),
        "mae_baseline": round(mae_baseline, 3),
        "verdict": verdict
    }
    print("ANSWER APACHE-20", json.dumps(ans))

# --------------------------------------------------------------------------- #
# Main

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    apache_19(header, rows)
    print()
    apache_20(header, rows)

if __name__ == "__main__":
    main()
