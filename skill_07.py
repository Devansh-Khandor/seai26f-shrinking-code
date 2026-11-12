#!/usr/bin/env python3
"""
skill_07.py

Usage:
    python skill_07.py <path_to_Apache_AllMeasurements.csv>
"""

import csv
import sys
import math
from collections import defaultdict

# ----------------------------------------------------------------------
# CSV helpers
# ----------------------------------------------------------------------
def read_csv(path):
    """Read the CSV file and return a list of dicts with integer values."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            new_row = {}
            for key, value in row.items():
                try:
                    new_row[key] = int(value)
                except ValueError:
                    # Skip malformed rows
                    break
            else:
                rows.append(new_row)
        return rows, reader.fieldnames

# ----------------------------------------------------------------------
# Correlation helpers
# ----------------------------------------------------------------------
def compute_correlations(rows, perf_key):
    """Return Pearson correlation for each binary setting with performance."""
    n = len(rows)
    perf_vals = [row[perf_key] for row in rows]
    mean_perf = sum(perf_vals) / n
    std_perf = math.sqrt(sum((x - mean_perf) ** 2 for x in perf_vals) / n)

    corr = {}
    for key in rows[0].keys():
        if key == perf_key:
            continue
        vals = [row[key] for row in rows]
        mean_val = sum(vals) / n
        std_val = math.sqrt(sum((x - mean_val) ** 2 for x in vals) / n)
        if std_val == 0 or std_perf == 0:
            continue
        cov = sum((vals[i] - mean_val) * (perf_vals[i] - mean_perf) for i in range(n)) / n
        corr[key] = cov / (std_val * std_perf)
    return corr

# ----------------------------------------------------------------------
# Heuristic for low performance discovery (APACHE-13)
# ----------------------------------------------------------------------
def heuristic_score(row, corr):
    """Score a configuration for likelihood of low performance."""
    score = 0.0
    for k, v in row.items():
        if k == 'Performance-':
            continue
        if k in corr and corr[k] < 0:
            # More negative correlation and value 1 increases score
            score += -corr[k] * v
    return score

def find_lowest_using_heuristic(rows, corr):
    """Simulate evaluating configurations ordered by heuristic until worst found."""
    scored = [(heuristic_score(r, corr), r) for r in rows]
    scored.sort(reverse=True)  # highest score first (most 'bad' settings)

    worst_perf = min(r['Performance-'] for r in rows)
    eval_count = 0
    lowest_found = None

    for score, r in scored:
        eval_count += 1
        if lowest_found is None or r['Performance-'] < lowest_found:
            lowest_found = r['Performance-']
            if lowest_found == worst_perf:
                break

    return eval_count, worst_perf, lowest_found, scored[0][1]

# ----------------------------------------------------------------------
# Linear regression helpers (APACHE-14)
# ----------------------------------------------------------------------
def gaussian_elimination(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for k in range(n):
        # Pivot
        max_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[max_row][k]) < 1e-12:
            raise ValueError("Matrix is singular or ill-conditioned")
        if max_row != k:
            M[k], M[max_row] = M[max_row], M[k]
        pivot = M[k][k]
        for j in range(k, n + 1):
            M[k][j] /= pivot
        for i in range(k + 1, n):
            factor = M[i][k]
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]

    x = [0] * n
    for i in reversed(range(n)):
        x[i] = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
    return x

def normal_equation_solve(X, y):
    n = len(X)
    p = len(X[0])
    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for i in range(n):
        for j in range(p):
            Xty[j] += X[i][j] * y[i]
            for k in range(p):
                XtX[j][k] += X[i][j] * X[i][k]
    beta = gaussian_elimination(XtX, Xty)
    return beta

def predict_performance(rows, feature_keys):
    """Return list of (index, predicted, actual, diff)."""
    X = []
    y = []
    for r in rows:
        row = [1]  # intercept
        for k in feature_keys:
            row.append(r[k])
        X.append(row)
        y.append(r['Performance-'])
    beta = normal_equation_solve(X, y)

    preds = []
    for idx, r in enumerate(rows, 1):
        pred = beta[0]
        for j, k in enumerate(feature_keys, start=1):
            pred += beta[j] * r[k]
        actual = r['Performance-']
        preds.append((idx, pred, actual, actual - pred))
    return preds

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path):
    rows, fieldnames = read_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        sys.exit(1)

    perf_key = None
    for k in fieldnames:
        if k.strip().lower() == 'performance-':
            perf_key = k
            break
    if perf_key is None:
        print("Error: No 'Performance-' column found.")
        sys.exit(1)

    feature_keys = [k for k in fieldnames if k.strip().lower() != 'performance-']

    # APACHE-13
    print("== APACHE-13 ==")
    corr = compute_correlations(rows, perf_key)
    eval_count, worst_perf, lowest_found, first_row = find_lowest_using_heuristic(rows, corr)
    print(f"Total configurations: {len(rows)}")
    print(f"Worst performance value in full dataset: {worst_perf}")
    print(f"Using heuristic, found worst after evaluating {eval_count} configurations.")
    print(f"First evaluated config performance: {first_row[perf_key]}")
    print(f"Lowest found during heuristic: {lowest_found}")
    print()

    # APACHE-14
    print("== APACHE-14 ==")
    preds = predict_performance(rows, feature_keys)
    preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
    print(f"Total configurations: {len(rows)}")
    print("Top 5 predicted high‑performance configurations:")
    for idx, pred, actual, diff in preds_sorted[:5]:
        print(f"  Config #{idx}: Pred={pred:.2f}, Actual={actual}, Diff={diff:.2f}")
    print()
    # Show actual best to illustrate early discovery
    best_actual = max(r[perf_key] for r in rows)
    best_index = next(i for i, r in enumerate(rows, 1) if r[perf_key] == best_actual)
    print(f"Actual best configuration is #{best_index} with performance {best_actual}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <path_to_csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
