#!/usr/bin/env python3
"""
skill_08.py

Implements:
  * APACHE-15: Recommend a minimal change to a poorly performing configuration that
    would lower its predicted performance, based on historically observed valid
    configurations.
  * APACHE-16: Find the smallest number of configuration-setting changes that
    bring a poorly performing configuration closer to the dataset's mean
    performance, while only considering valid combinations from the data.

Usage:
    python skill_08.py <path_to_Apache_AllMeasurements.csv>
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
    """Read the CSV file and return a list of dicts with integer values and fieldnames."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            new_row = {}
            for key, value in row.items():
                try:
                    new_row[key] = int(value)
                except ValueError:
                    # skip malformed rows
                    break
            else:
                rows.append(new_row)
        return rows, reader.fieldnames

# ----------------------------------------------------------------------
# Statistical helpers
# ----------------------------------------------------------------------
def mean(values):
    return sum(values) / len(values) if values else 0.0

def stddev(values, mean_val):
    return math.sqrt(sum((x - mean_val) ** 2 for x in values) / len(values)) if values else 0.0

# ----------------------------------------------------------------------
# Linear regression helpers (APACHE-14 logic reused)
# ----------------------------------------------------------------------
def gaussian_elimination(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for k in range(n):
        # pivot
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

def fit_regression(rows, feature_keys, perf_key):
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
    pred = beta[0]
    for idx, k in enumerate(feature_keys, start=1):
        pred += beta[idx] * row[k]
    return pred

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def config_key(row, keys):
    """Return a tuple of setting values for the given keys."""
    return tuple(row[k] for k in keys)

def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)

# ----------------------------------------------------------------------
# APACHE-15 logic
# ----------------------------------------------------------------------
def apache_15(rows, feature_keys, perf_key, beta):
    """
    For the configuration with lowest actual performance, suggest flipping
    a single setting that yields the lowest predicted performance among
    historically valid configurations.
    """
    # Identify the worst configuration
    worst_row = min(rows, key=lambda r: r[perf_key])
    worst_key = config_key(worst_row, feature_keys)

    # Build a lookup for existing configurations
    existing = {config_key(r, feature_keys): r for r in rows}

    best_candidate = None
    best_pred = None
    best_change = None

    for idx, key in enumerate(feature_keys):
        # Toggle the setting
        toggled = list(worst_row[key] for key in feature_keys)
        toggled[idx] ^= 1  # flip 0<->1
        toggled_key = tuple(toggled)
        if toggled_key in existing:
            candidate_row = existing[toggled_key]
            cand_pred = predict(candidate_row, beta, feature_keys)
            if best_pred is None or cand_pred < best_pred:
                best_pred = cand_pred
                best_candidate = candidate_row
                best_change = key

    # Prepare result
    result = []
    if best_candidate:
        result.append(f"Poorly performing configuration (Perf={worst_row[perf_key]})")
        result.append(f"Recommended setting change: flip '{best_change}'")
        result.append(f"Predicted performance after change: {best_pred:.2f}")
    else:
        result.append("Could not find a single‑setting change that improves prediction.")
    return result

# ----------------------------------------------------------------------
# APACHE-16 logic
# ----------------------------------------------------------------------
def apache_16(rows, feature_keys, perf_key, target_perf):
    """
    Find the minimal number of setting changes required to reach a configuration
    whose performance is at least the target performance. Only consider
    configurations that exist in the dataset.
    """
    # Map configuration to row
    config_to_row = {config_key(r, feature_keys): r for r in rows}

    # Identify the worst configuration
    worst_row = min(rows, key=lambda r: r[perf_key])
    worst_key = config_key(worst_row, feature_keys)

    # Filter candidates meeting target
    candidates = [r for r in rows if r[perf_key] >= target_perf]
    if not candidates:
        return ["No configuration meets the target performance."]

    # Compute minimal Hamming distance
    min_dist = None
    best_cands = []

    for cand in candidates:
        cand_key = config_key(cand, feature_keys)
        dist = hamming(worst_key, cand_key)
        if min_dist is None or dist < min_dist:
            min_dist = dist
            best_cands = [cand]
        elif dist == min_dist:
            best_cands.append(cand)

    # Choose the best (highest performance) among those with minimal distance
    best = max(best_cands, key=lambda r: r[perf_key])

    # Determine which settings differ
    changes = [(k, worst_row[k], best[k]) for k in feature_keys if worst_row[k] != best[k]]

    result = []
    result.append(f"Target performance threshold: {target_perf:.2f}")
    result.append(f"Minimal number of setting changes needed: {min_dist}")
    result.append("Changes required:")
    for k, old, new in changes:
        result.append(f"  {k}: {old} -> {new}")
    result.append(f"Resulting configuration performance: {best[perf_key]}")
    return result

# ----------------------------------------------------------------------
# Main entry point
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
    # APACHE-15
    # --------------------------------------------------------------
    print("== APACHE-15 ==")
    for line in apache_15(rows, feature_keys, perf_key, beta):
        print(line)

    # --------------------------------------------------------------
    # APACHE-16
    # --------------------------------------------------------------
    print("\n== APACHE-16 ==")
    # Use mean performance as the target threshold
    perf_values = [r[perf_key] for r in rows]
    target_perf = mean(perf_values)
    for line in apache_16(rows, feature_keys, perf_key, target_perf):
        print(line)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <path_to_Apache_AllMeasurements.csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
