#!/usr/bin/env python3
"""
skill_02.py

Usage:
    python skill_02.py <path_to_Apache_AllMeasurements.csv>
"""

import csv
import sys
import math
from collections import defaultdict
from itertools import combinations

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
                # Convert to int, keep 'Performance-' as int
                new_row[key] = int(value)
            rows.append(new_row)
        return rows, reader.fieldnames

# ----------------------------------------------------------------------
# Linear regression helpers
# ----------------------------------------------------------------------
def gaussian_elimination(A, b):
    """
    Solve Ax = b using Gaussian elimination with partial pivoting.
    A: list of lists, square matrix
    b: list, RHS vector
    Returns solution vector x.
    """
    n = len(A)
    # Augment A with b
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    # Forward elimination
    for k in range(n):
        # Pivot selection
        max_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[max_row][k]) < 1e-12:
            raise ValueError("Matrix is singular or ill-conditioned")
        if max_row != k:
            M[k], M[max_row] = M[max_row], M[k]

        # Normalize pivot row
        pivot = M[k][k]
        for j in range(k, n + 1):
            M[k][j] /= pivot

        # Eliminate below
        for i in range(k + 1, n):
            factor = M[i][k]
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]

    # Back substitution
    x = [0] * n
    for i in reversed(range(n)):
        x[i] = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
    return x

def build_design_matrix(rows, feature_keys):
    """Return X matrix (n x (p+1)) and y vector."""
    X = []
    y = []
    for r in rows:
        row = [1]  # intercept
        for k in feature_keys:
            row.append(r[k])
        X.append(row)
        y.append(r['Performance-'])
    return X, y

def normal_equation_solve(X, y):
    """Solve (X^T X) beta = X^T y."""
    n = len(X)
    p = len(X[0])  # includes intercept
    # Compute XtX and Xty
    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for i in range(n):
        for j in range(p):
            Xty[j] += X[i][j] * y[i]
            for k in range(p):
                XtX[j][k] += X[i][j] * X[i][k]
    beta = gaussian_elimination(XtX, Xty)
    return beta

# ----------------------------------------------------------------------
# APACHE-03: Predictive model
# ----------------------------------------------------------------------
def apache_03(rows, feature_keys):
    """Fit a linear model and provide simple predictions."""
    X, y = build_design_matrix(rows, feature_keys)
    beta = normal_equation_solve(X, y)  # beta[0] is intercept

    # Prediction for a configuration with all zero settings
    all_off_pred = beta[0]

    # Predictions on training data
    preds = []
    for r in rows:
        pred = beta[0]
        for idx, k in enumerate(feature_keys, start=1):
            pred += beta[idx] * r[k]
        preds.append(pred)

    # Compute mean absolute error
    mae = sum(abs(p - a) for p, a in zip(preds, y)) / len(y)

    # Compute correlation between predictions and actual
    mean_y = sum(y) / len(y)
    mean_p = sum(preds) / len(preds)
    cov = sum((p - mean_p) * (a - mean_y) for p, a in zip(preds, y)) / len(y)
    std_y = math.sqrt(sum((a - mean_y) ** 2 for a in y) / len(y))
    std_p = math.sqrt(sum((p - mean_p) ** 2 for p in preds) / len(preds))
    corr = cov / (std_y * std_p) if std_y * std_p != 0 else 0.0

    return {
        'all_off_pred': all_off_pred,
        'mae': mae,
        'corr': corr,
    }

# ----------------------------------------------------------------------
# APACHE-04: Single setting change analysis
# ----------------------------------------------------------------------
def apache_04(rows, feature_keys):
    """Analyze pairs that differ by exactly one setting."""
    diffs = []
    for r1, r2 in combinations(rows, 2):
        # Count differences in features
        diff_count = sum(1 for k in feature_keys if r1[k] != r2[k])
        if diff_count == 1:
            diff_perf = abs(r1['Performance-'] - r2['Performance-'])
            diffs.append(diff_perf)
    if not diffs:
        return {'count': 0, 'mean': None, 'std': None}
    mean_diff = sum(diffs) / len(diffs)
    std_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / len(diffs))
    return {
        'count': len(diffs),
        'mean': mean_diff,
        'std': std_diff,
    }

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path):
    rows, fieldnames = read_csv(csv_path)
    # Determine feature columns (all except 'Performance-')
    feature_keys = [k for k in fieldnames if k.strip().lower() != 'performance-']

    # APACHE-03
    print("== APACHE-03 ==")
    stats = apache_03(rows, feature_keys)
    print(f"Estimated performance for all‑off configuration: {stats['all_off_pred']:.2f}")
    print(f"Mean absolute error on training data: {stats['mae']:.2f}")
    print(f"Correlation between predictions and actual: {stats['corr']:.2f}")

    # APACHE-04
    print("\n== APACHE-04 ==")
    diff_stats = apache_04(rows, feature_keys)
    if diff_stats['count'] == 0:
        print("No pairs of configurations differ by exactly one setting.")
    else:
        print(f"Number of single‑setting change pairs: {diff_stats['count']}")
        print(f"Average performance difference: {diff_stats['mean']:.2f}")
        print(f"Standard deviation of these differences: {diff_stats['std']:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <path_to_csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
