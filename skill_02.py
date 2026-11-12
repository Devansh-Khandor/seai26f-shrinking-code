#!/usr/bin/env python3
"""
Skill 02: Predictive performance estimation and uncertainty analysis.

Usage:
    python skill_02.py <csv_path>
"""

import sys
from pathlib import Path
import csv
import math
from collections import Counter, defaultdict

# ----------------------------------------------------------------------
# Re‑use helper functions from the previous skill
# ----------------------------------------------------------------------
from skill_01 import (
    read_csv,
    compute_statistics,
    best_worst_configs,
    top_percentile_threshold,
    find_distinguishing_settings,
    format_config,
)

# ----------------------------------------------------------------------
# Linear‑regression utilities
# ----------------------------------------------------------------------
def train_linear_model(rows, performance):
    """
    Train a simple linear regression model (least‑squares).
    Returns (coefficients, feature_names).
    The first coefficient is the intercept.
    """
    # All columns except Performance- are features
    feature_names = [k for k in rows[0].keys() if k != "Performance-"]
    n_samples = len(rows)
    n_features = len(feature_names) + 1  # +1 for intercept

    # Build design matrix X (with intercept column)
    X = []
    for r in rows:
        row = [1]  # intercept
        for col in feature_names:
            val = r[col]
            # Convert numeric string to int; default to 0 if not numeric
            try:
                row.append(int(val))
            except ValueError:
                row.append(0)
        X.append(row)

    y = performance

    # Compute XTX and XTy
    XTX = [[0.0] * n_features for _ in range(n_features)]
    XTy = [0.0] * n_features

    for i in range(n_samples):
        xi = X[i]
        yi = y[i]
        for j in range(n_features):
            XTy[j] += xi[j] * yi
            for k in range(j, n_features):
                XTX[j][k] += xi[j] * xi[k]

    # Symmetrize lower triangle
    for j in range(n_features):
        for k in range(j):
            XTX[j][k] = XTX[k][j]

    coeffs = solve_linear_system(XTX, XTy)
    return coeffs, feature_names


def solve_linear_system(A, b):
    """
    Solve linear system A x = b by Gaussian elimination with partial pivoting.
    A is a square matrix; b is a vector.
    Returns solution vector x.
    """
    n = len(A)
    # Make a working copy of A and b
    M = [row[:] for row in A]
    v = b[:]

    for i in range(n):
        # Partial pivot
        max_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[max_row][i]) < 1e-12:
            raise ValueError("Matrix is singular or near‑singular")
        if max_row != i:
            M[i], M[max_row] = M[max_row], M[i]
            v[i], v[max_row] = v[max_row], v[i]

        # Eliminate below
        for j in range(i + 1, n):
            factor = M[j][i] / M[i][i]
            for k in range(i, n):
                M[j][k] -= factor * M[i][k]
            v[j] -= factor * v[i]

    # Back substitution
    x = [0.0] * n
    for i in reversed(range(n)):
        s = v[i]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x


def predict(row, coeffs, feature_names):
    """
    Return predicted performance for a single row using the model.
    """
    intercept = coeffs[0]
    score = intercept
    for idx, col in enumerate(feature_names):
        try:
            val = int(row[col])
        except ValueError:
            val = 0
        score += coeffs[idx + 1] * val
    return score


def compute_prediction_errors(rows, coeffs, feature_names):
    """
    Return list of predicted, actual, and error (actual - predicted).
    """
    preds = []
    actuals = []
    errors = []
    for r in rows:
        pred = predict(r, coeffs, feature_names)
        actual = float(r["Performance-"])
        preds.append(pred)
        actuals.append(actual)
        errors.append(actual - pred)
    return preds, actuals, errors


def mean_absolute_error(errors):
    return sum(abs(e) for e in errors) / len(errors) if errors else 0.0


def mean_squared_error(errors):
    return sum(e * e for e in errors) / len(errors) if errors else 0.0


def standard_deviation(values):
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def differ_by_one(row1, row2, feature_names):
    diff_count = 0
    for col in feature_names:
        if row1[col] != row2[col]:
            diff_count += 1
            if diff_count > 1:
                return False
    return diff_count == 1


def compare_adjacent_configs(rows, coeffs, feature_names, performance):
    """
    Find all pairs of configurations that differ by exactly one setting.
    Return a list of tuples:
        (idx1, idx2, actual_diff, pred_diff, abs_error)
    """
    results = []
    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            if differ_by_one(rows[i], rows[j], feature_names):
                actual_diff = performance[j] - performance[i]
                pred_i = predict(rows[i], coeffs, feature_names)
                pred_j = predict(rows[j], coeffs, feature_names)
                pred_diff = pred_j - pred_i
                error = abs(actual_diff - pred_diff)
                results.append((i, j, actual_diff, pred_diff, error))
    return results


# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main(csv_path_str):
    csv_path = Path(csv_path_str)
    if not csv_path.is_file():
        print(f"Error: file '{csv_path}' not found.", file=sys.stderr)
        sys.exit(1)

    rows, performance = read_csv(csv_path)

    # Train linear model
    coeffs, feature_names = train_linear_model(rows, performance)

    # ==============================
    # APACHE-03
    # ==============================
    print("== APACHE-03 ==")

    preds, actuals, errors = compute_prediction_errors(rows, coeffs, feature_names)

    mae = mean_absolute_error(errors)
    mse = mean_squared_error(errors)
    std_err = standard_deviation(errors)

    print(f"Mean absolute error (MAE): {mae:.2f}")
    print(f"Mean squared error (MSE): {mse:.2f}")
    print(f"Standard deviation of errors: {std_err:.2f}")

    # Provide a few sample predictions for reference
    print("\nSample predictions (first 5 configs):")
    for i in range(min(5, len(rows))):
        print(f"  Config {i+1}: Pred={preds[i]:.2f}, Actual={actuals[i]:.2f}")

    print()

    # ==============================
    # APACHE-04
    # ==============================
    print("== APACHE-04 ==")

    comparison_results = compare_adjacent_configs(rows, coeffs, feature_names, performance)

    if not comparison_results:
        print("No configuration pairs differ by exactly one setting.")
    else:
        diff_errors = [res[4] for res in comparison_results]
        mean_diff_err = mean_absolute_error(diff_errors)
        std_diff_err = standard_deviation(diff_errors)

        print(f"Number of adjacent configuration pairs: {len(comparison_results)}")
        print(f"Mean absolute error of predicted performance differences: {mean_diff_err:.2f}")
        print(f"Standard deviation of difference errors (uncertainty): {std_diff_err:.2f}")

        # Show a few sample comparisons
        print("\nSample adjacent pair comparisons (first 3):")
        for i, res in enumerate(comparison_results[:3]):
            idx1, idx2, act_diff, pred_diff, err = res
            print(f"  Pair {i+1}: Config {idx1+1} vs Config {idx2+1}")
            print(f"    Actual diff={act_diff:.2f}, Predicted diff={pred_diff:.2f}, Abs error={err:.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
