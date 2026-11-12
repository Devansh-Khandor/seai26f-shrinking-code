#!/usr/bin/env python3
"""
skill_02.py

Usage:
    python skill_02.py <csv_path>

The script prints:
    * CAR-03: Estimations for missing fuel‑efficiency (mpg) values.
    * CAR-04: Estimated weight for newly acquired vehicles with uncertainty.
"""

import sys
import csv
import math
from pathlib import Path

# Import helper functions from skill_01
from skill_01 import read_csv, mean, format_number

# --------------------------------------------------------------------------- #
# Linear regression helpers
# --------------------------------------------------------------------------- #

def linear_regression_coefficients(data, target_key, feature_keys):
    """
    Compute regression coefficients (including intercept) for predicting
    `target_key` from `feature_keys`. Uses the normal equations.
    Returns a list of coefficients: [intercept, beta1, beta2, ...]
    """
    # Build X matrix with bias term
    X = []
    y = []
    for v in data:
        # Skip rows where target is missing or zero
        target_val = v.get(target_key)
        if target_val is None or target_val == '' or target_val <= 0:
            continue
        row = [1.0]  # intercept
        for key in feature_keys:
            row.append(float(v[key]))
        X.append(row)
        y.append(float(target_val))

    n = len(X)
    if n == 0:
        raise ValueError("No data available to train regression.")

    k = len(feature_keys)  # number of predictors

    # Compute A = X^T X and B = X^T y
    A = [[0.0] * (k + 1) for _ in range(k + 1)]
    B = [0.0] * (k + 1)

    for i in range(n):
        xi = X[i]
        yi = y[i]
        for p in range(k + 1):
            B[p] += xi[p] * yi
            for q in range(p, k + 1):
                A[p][q] += xi[p] * xi[q]
                if p != q:
                    A[q][p] += xi[p] * xi[q]

    # Solve A * beta = B using Gaussian elimination
    beta = solve_linear_system(A, B)
    return beta

def solve_linear_system(A, B):
    """Solve linear system A x = B for x using Gaussian elimination with partial pivoting."""
    n = len(B)
    # Augmented matrix
    M = [row[:] + [B[i]] for i, row in enumerate(A)]

    # Forward elimination
    for i in range(n):
        # Pivot
        max_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[max_row][i]) < 1e-12:
            raise ValueError("Matrix is singular or ill-conditioned.")
        if max_row != i:
            M[i], M[max_row] = M[max_row], M[i]

        pivot = M[i][i]
        # Normalize pivot row
        for j in range(i, n + 1):
            M[i][j] /= pivot

        # Eliminate below
        for r in range(i + 1, n):
            factor = M[r][i]
            for c in range(i, n + 1):
                M[r][c] -= factor * M[i][c]

    # Back substitution
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
    return x

def predict_using_coefficients(beta, row, feature_keys):
    """Predict target value for a single row using coefficients beta."""
    val = beta[0]  # intercept
    for i, key in enumerate(feature_keys):
        val += beta[i + 1] * float(row[key])
    return val

def regression_statistics(data, target_key, feature_keys, beta):
    """
    Compute standard error of the estimate for the regression model.
    Returns sigma (standard deviation of residuals).
    """
    n = 0
    sse = 0.0
    for v in data:
        target_val = v.get(target_key)
        if target_val is None or target_val == '' or target_val <= 0:
            continue
        pred = predict_using_coefficients(beta, v, feature_keys)
        n += 1
        sse += (float(target_val) - pred) ** 2
    k = len(feature_keys)
    if n - k - 1 > 0:
        sigma = math.sqrt(sse / (n - k - 1))
    else:
        sigma = float('nan')
    return sigma

# --------------------------------------------------------------------------- #
# CAR-03: Estimate missing MPG
# --------------------------------------------------------------------------- #

def print_car_03_estimates(data):
    # Regression model: MPG ~ Lbs- + Acc+ + Volume + HpX
    target_key = 'Mpg+'
    feature_keys = ['Lbs-', 'Acc+', 'Volume', 'HpX']
    try:
        beta = linear_regression_coefficients(data, target_key, feature_keys)
    except ValueError as exc:
        print("\n== CAR-03 ==")
        print(f"Could not train regression model: {exc}")
        return

    sigma = regression_statistics(data, target_key, feature_keys, beta)

    predictions = []
    for v in data:
        mpg_val = v.get(target_key)
        if mpg_val is None or mpg_val == '' or mpg_val <= 0:
            pred = predict_using_coefficients(beta, v, feature_keys)
            predictions.append((v, pred, sigma))
    # If no missing MPG values, still display predictions for the first 3 vehicles
    if not predictions:
        for v in data[:3]:
            pred = predict_using_coefficients(beta, v, feature_keys)
            predictions.append((v, pred, sigma))

    print("\n== CAR-03 ==")
    print("Estimated MPG for vehicles with missing fuel‑efficiency (or first 3 vehicles):")
    for i, (v, pred, sigma) in enumerate(predictions, start=1):
        # Show available data that influences prediction
        print(f"{i}. Clndrs={v['Clndrs']}, Volume={v['Volume']}, "
              f"HpX={v['HpX']}, Origin={v['origin']}, "
              f"Predicted MPG: {format_number(pred)} "
              f"+/- {format_number(sigma)}")

# --------------------------------------------------------------------------- #
# CAR-04: Estimate weight
# --------------------------------------------------------------------------- #

def print_car_04_estimates(data):
    # Regression model: Weight ~ MPG + Acc + Volume + HpX
    target_key = 'Lbs-'
    feature_keys = ['Mpg+', 'Acc+', 'Volume', 'HpX']
    try:
        beta = linear_regression_coefficients(data, target_key, feature_keys)
    except ValueError as exc:
        print("\n== CAR-04 ==")
        print(f"Could not train regression model: {exc}")
        return

    sigma = regression_statistics(data, target_key, feature_keys, beta)

    print("\n== CAR-04 ==")
    print("Estimated weight for newly acquired vehicles (with uncertainty):")
    # Show estimates for first 3 vehicles
    for i, v in enumerate(data[:3], start=1):
        pred = predict_using_coefficients(beta, v, feature_keys)
        print(f"{i}. Clndrs={v['Clndrs']}, Volume={v['Volume']}, "
              f"HpX={v['HpX']}, Origin={v['origin']}, "
              f"Predicted Weight: {format_number(pred)} kg "
              f"+/- {format_number(sigma)} kg")

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    data = read_csv(csv_path)
    if not data:
        print("No data found in CSV.", file=sys.stderr)
        sys.exit(1)

    print_car_03_estimates(data)
    print_car_04_estimates(data)

if __name__ == "__main__":
    main()
