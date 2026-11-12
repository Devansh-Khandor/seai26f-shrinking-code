#!/usr/bin/env python3
"""
skill_09.py

Implements:
  * APACHE-17: Identify configurations whose recorded performance values are unusually different
    from the performance predicted by a linear model fitted to the historical data.
  * APACHE-18: Flag configurations whose setting combinations are unusually distant from their
    nearest historically similar configuration.

Usage:
    python skill_09.py <path_to_Apache_AllMeasurements.csv>
"""

import csv
import sys
import math
from collections import defaultdict

# ----------------------------------------------------------------------
# CSV helpers
# ----------------------------------------------------------------------
def read_csv(path):
    """Read the CSV file and return a list of dicts with integer values and fieldnames."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            new_row = {}
            for k, v in row.items():
                try:
                    new_row[k] = int(v)
                except ValueError:
                    # skip malformed rows
                    break
            else:
                rows.append(new_row)
        return rows, reader.fieldnames

# ----------------------------------------------------------------------
# Linear regression helpers
# ----------------------------------------------------------------------
def gaussian_elimination(A, b):
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for k in range(n):
        # Pivot selection
        max_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[max_row][k]) < 1e-12:
            raise ValueError("Matrix is singular or ill‑conditioned")
        if max_row != k:
            M[k], M[max_row] = M[max_row], M[k]
        pivot = M[k][k]
        for j in range(k, n + 1):
            M[k][j] /= pivot
        for i in range(k + 1, n):
            factor = M[i][k]
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]

    # Back substitution
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
    return x

def normal_equation_solve(X, y):
    """Solve (X^T X) β = X^T y."""
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

def fit_regression(rows, feature_keys, perf_key):
    """Fit a linear model and return the coefficient vector β."""
    X = []
    y = []
    for r in rows:
        row = [1]  # intercept
        for k in feature_keys:
            row.append(r[k])
        X.append(row)
        y.append(r[perf_key])
    beta = normal_equation_solve(X, y)
    return beta

def predict(row, beta, feature_keys):
    """Predict performance for a single configuration."""
    pred = beta[0]
    for idx, k in enumerate(feature_keys, start=1):
        pred += beta[idx] * row[k]
    return pred

# ----------------------------------------------------------------------
# APACHE-17: Outlier detection based on residuals
# ----------------------------------------------------------------------
def apache_17(rows, feature_keys, perf_key, beta):
    """Identify configurations with residuals far from the model predictions."""
    results = []
    residuals = []
    for idx, r in enumerate(rows, 1):
        pred = predict(r, beta, feature_keys)
        res = r[perf_key] - pred
        residuals.append(res)
        results.append((idx, r[perf_key], pred, res))

    # Compute residual statistics
    n = len(residuals)
    if n == 0:
        return ["No data to analyze."]
    mean_res = sum(residuals) / n
    std_res = math.sqrt(sum((x - mean_res) ** 2 for x in residuals) / n)

    # Threshold: mean + 2*std (two‑sigma rule)
    threshold = mean_res + 2 * std_res
    # Also consider negative side: mean - 2*std
    threshold_neg = mean_res - 2 * std_res

    # Find configs with |residual| > 2*std
    flagged = [r for r in results if r[3] > threshold or r[3] < threshold_neg]

    if not flagged:
        return ["No unusually deviating configurations found (APACHE‑17)."]

    # Sort by absolute residual descending
    flagged.sort(key=lambda x: abs(x[3]), reverse=True)

    output = []
    output.append(f"Number of flagged configurations: {len(flagged)}")
    output.append(f"Residual mean: {mean_res:.2f}, std: {std_res:.2f}")
    output.append("Top deviating configurations:")
    for i, (idx, perf, pred, res) in enumerate(flagged[:5], 1):
        output.append(f"  #{idx:3d} Perf={perf:6d} Pred={pred:7.2f} Residual={res:7.2f}")
    return output

# ----------------------------------------------------------------------
# APACHE-18: Unusual setting combinations
# ----------------------------------------------------------------------
def hamming(a, b):
    """Return Hamming distance between two tuples of 0/1."""
    return sum(1 for x, y in zip(a, b) if x != y)

def apache_18(rows, feature_keys):
    """Flag configurations whose setting combination is unusually far from their nearest neighbour."""
    # Precompute config keys
    keys = [tuple(r[k] for k in feature_keys) for r in rows]
    n = len(keys)
    if n == 0:
        return ["No data to analyze."]

    nearest_dist = [None] * n
    nearest_idx = [None] * n

    # Compute nearest neighbor distance for each configuration
    for i in range(n):
        min_d = None
        min_j = None
        for j in range(n):
            if i == j:
                continue
            d = hamming(keys[i], keys[j])
            if min_d is None or d < min_d:
                min_d = d
                min_j = j
        nearest_dist[i] = min_d
        nearest_idx[i] = min_j

    # Statistics of distances
    dist_vals = [d for d in nearest_dist if d is not None]
    m = len(dist_vals)
    mean_d = sum(dist_vals) / m
    std_d = math.sqrt(sum((x - mean_d) ** 2 for x in dist_vals) / m)

    # Threshold: mean + 2*std
    threshold = mean_d + 2 * std_d

    flagged = [(i, nearest_dist[i], nearest_idx[i]) for i, d in enumerate(nearest_dist) if d > threshold]

    if not flagged:
        return ["No unusually distant configurations found (APACHE‑18)."]

    output = []
    output.append(f"Number of flagged configurations: {len(flagged)}")
    output.append(f"Nearest‑neighbour distance mean: {mean_d:.2f}, std: {std_d:.2f}")
    output.append("Flagged configurations (index, distance, nearest neighbour index):")
    for idx, dist, nn in flagged[:5]:
        output.append(f"  #{idx+1:3d} dist={dist:2d} nn={nn+1:3d}")
    return output

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path):
    rows, fieldnames = read_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        sys.exit(1)

    perf_key = next((k for k in fieldnames if k.strip().lower() == 'performance-'), None)
    if perf_key is None:
        print("Error: No 'Performance-' column found.")
        sys.exit(1)

    feature_keys = [k for k in fieldnames if k.strip().lower() != 'performance-']

    # Fit regression model once
    beta = fit_regression(rows, feature_keys, perf_key)

    # --------------------------------------------------------------
    # APACHE-17
    # --------------------------------------------------------------
    print("== APACHE-17 ==")
    for line in apache_17(rows, feature_keys, perf_key, beta):
        print(line)

    # --------------------------------------------------------------
    # APACHE-18
    # --------------------------------------------------------------
    print("\n== APACHE-18 ==")
    for line in apache_18(rows, feature_keys):
        print(line)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <path_to_Apache_AllMeasurements.csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
