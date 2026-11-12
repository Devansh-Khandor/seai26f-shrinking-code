#!/usr/bin/env python3
"""
Apache performance analysis: predictions and differences.

This script extends the original skill_01.py functionality by:
  - APACHE-03: Predicting performance for a configuration based on its
    historical benchmark records.
  - APACHE-04: Estimating the performance difference caused by toggling
    a single configuration setting, with an uncertainty estimate.

Usage:
    python skill_02.py <csv_path>
"""

import sys
import csv
import statistics
from typing import List, Dict, Any, Tuple
import skill_01


def load_csv(path: str) -> List[Dict[str, Any]]:
    """Delegate to skill_01's load_csv."""
    return skill_01.load_csv(path)


def solve_linear_regression(X: List[List[float]], y: List[float]) -> List[float]:
    """
    Solve the normal equations (X^T X) beta = X^T y for beta using
    Gaussian elimination with partial pivoting.
    """
    m = len(X)
    n = len(X[0])  # number of features (including bias)

    # Build A = X^T X and b = X^T y
    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    for i in range(m):
        xi = X[i]
        yi = y[i]
        for p in range(n):
            b[p] += xi[p] * yi
            for q in range(p, n):
                A[p][q] += xi[p] * xi[q]

    # Mirror the upper triangle to the lower triangle
    for p in range(n):
        for q in range(p + 1, n):
            A[q][p] = A[p][q]

    # Augment A with b to form [A | b]
    for i in range(n):
        A[i].append(b[i])

    # Forward elimination with partial pivoting
    for i in range(n):
        # Pivot selection
        pivot_row = i
        max_val = abs(A[i][i])
        for r in range(i + 1, n):
            if abs(A[r][i]) > max_val:
                max_val = abs(A[r][i])
                pivot_row = r
        if max_val < 1e-12:
            continue  # singular, skip
        if pivot_row != i:
            A[i], A[pivot_row] = A[pivot_row], A[i]

        # Normalize pivot row
        piv = A[i][i]
        for j in range(i, n + 1):
            A[i][j] /= piv

        # Eliminate other rows
        for r in range(n):
            if r == i:
                continue
            factor = A[r][i]
            if abs(factor) < 1e-12:
                continue
            for j in range(i, n + 1):
                A[r][j] -= factor * A[i][j]

    # Extract beta coefficients
    beta = [A[i][n] for i in range(n)]
    return beta


def train_regression(rows: List[Dict[str, Any]]) -> Tuple[List[float], List[str], List[float]]:
    """
    Train a linear regression model to predict 'Performance-'.
    Returns (beta, feature_names, predictions).
    """
    if not rows:
        return [], [], []

    feature_names = [k for k in rows[0] if k != 'Performance-']
    X = [[1] + [float(row[f]) for f in feature_names] for row in rows]
    y = [float(row['Performance-']) for row in rows]
    beta = solve_linear_regression(X, y)
    predictions = [sum(b * x for b, x in zip(beta, X_i)) for X_i in X]
    return beta, feature_names, predictions


def regression_metrics(rows: List[Dict[str, Any]],
                       predictions: List[float]) -> Tuple[float, float, float]:
    """
    Compute mean predicted performance, R^2, and mean absolute error.
    """
    if not rows:
        return 0.0, 0.0, 0.0

    actual = [float(row['Performance-']) for row in rows]
    mean_pred = statistics.mean(predictions)
    mean_actual = statistics.mean(actual)

    # R-squared
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predictions))
    ss_tot = sum((a - mean_actual) ** 2 for a in actual)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    # Mean absolute error
    mae = statistics.mean(abs(a - p) for a, p in zip(actual, predictions))

    return mean_pred, r2, mae


def compute_pair_differences(rows: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    For every pair of configurations that differ by exactly one setting,
    compute the performance difference and group by the differing feature.
    Returns a dict: feature -> list of differences.
    """
    if not rows:
        return {}

    feature_names = [k for k in rows[0] if k != 'Performance-']
    diff_dict: Dict[str, List[float]] = {f: [] for f in feature_names}

    n = len(rows)
    for i in range(n):
        row_i = rows[i]
        for j in range(i + 1, n):
            row_j = rows[j]
            diff_feats = [f for f in feature_names if row_i[f] != row_j[f]]
            if len(diff_feats) == 1:
                f = diff_feats[0]
                diff = float(row_j['Performance-']) - float(row_i['Performance-'])
                diff_dict[f].append(diff)

    # Remove entries with no differences recorded
    diff_dict = {f: diffs for f, diffs in diff_dict.items() if diffs}
    return diff_dict


def estimate_feature_effects(diff_dict: Dict[str, List[float]]) -> List[Tuple[str, float, float]]:
    """
    For each feature, compute mean difference and standard deviation.
    Return list sorted by absolute mean difference descending.
    """
    results = []
    for f, diffs in diff_dict.items():
        mean_diff = statistics.mean(diffs)
        std_dev = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        results.append((f, mean_diff, std_dev))
    # Sort by absolute mean difference descending
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    return results


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)

    # APACHE-03: Performance prediction
    beta, feature_names, predictions = train_regression(rows)
    mean_pred, r2, mae = regression_metrics(rows, predictions)

    print("== APACHE-03 ==")
    print(f"Mean predicted performance: {mean_pred:.2f}")
    print(f"R-squared: {r2:.4f}")
    print(f"Mean absolute error: {mae:.2f}")

    # APACHE-04: Estimated effect of single setting changes
    diff_dict = compute_pair_differences(rows)
    feature_effects = estimate_feature_effects(diff_dict)

    print("\n== APACHE-04 ==")
    if not feature_effects:
        print("No pairs of configurations differing by a single setting found.")
    else:
        top_n = min(3, len(feature_effects))
        print(f"Estimated performance difference per feature change (mean ± std dev):")
        for i in range(top_n):
            f, mean_diff, std_dev = feature_effects[i]
            print(f"- Changing '{f}': {mean_diff:.1f} ± {std_dev:.1f}")


if __name__ == "__main__":
    main()
