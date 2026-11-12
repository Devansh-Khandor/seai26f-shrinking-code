#!/usr/bin/env python3
"""
Skill 02: Fuel‑efficiency and weight estimation for used‑car dealership.

Usage:
    python skill_02.py <csv_path>
"""

import csv
import sys
import math

# ----------------------------------------------------------------------
# Data handling helpers
# ----------------------------------------------------------------------
def read_data(csv_path):
    """Read CSV and return a list of vehicle records with numeric values converted.
    Missing numeric values are stored as None."""
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # Convert numeric columns; leave None if conversion fails
            for key in ['Clndrs', 'Volume', 'HpX', 'Model', 'origin',
                        'Lbs-', 'Acc+', 'Mpg+']:
                try:
                    row[key] = float(row[key]) if row[key] != '' else None
                except (ValueError, KeyError):
                    row[key] = None
            rows.append(row)
    return rows

# ----------------------------------------------------------------------
# Linear algebra utilities (no external dependencies)
# ----------------------------------------------------------------------
def transpose(A):
    return [list(col) for col in zip(*A)]

def mat_mult(A, B):
    """Multiply matrices A (m x n) and B (n x p)."""
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            s = 0.0
            for k in range(len(A[0])):
                s += A[i][k] * B[k][j]
            row.append(s)
        result.append(row)
    return result

def mat_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def mat_inv(A):
    """Inverse of a square matrix using Gaussian elimination."""
    n = len(A)
    # Augment with identity
    AM = [A[i][:] + [float(i == j) for j in range(n)] for i in range(n)]
    # Forward elimination
    for i in range(n):
        # Find pivot
        pivot = AM[i][i]
        if abs(pivot) < 1e-12:
            # Search for non-zero pivot in rows below
            for r in range(i+1, n):
                if abs(AM[r][i]) > 1e-12:
                    AM[i], AM[r] = AM[r], AM[i]
                    pivot = AM[i][i]
                    break
            else:
                raise ValueError("Matrix not invertible")
        # Normalize pivot row
        for j in range(2*n):
            AM[i][j] /= pivot
        # Eliminate other rows
        for r in range(n):
            if r == i:
                continue
            factor = AM[r][i]
            for j in range(2*n):
                AM[r][j] -= factor * AM[i][j]
    # Extract inverse
    inv = [row[n:] for row in AM]
    return inv

def mat_vec_mult(A, v):
    """Multiply matrix A (m x n) with vector v (n)."""
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def solve_linear_system(A, b):
    """Solve A x = b for x, where A is square."""
    return mat_vec_mult(mat_inv(A), b)

# ----------------------------------------------------------------------
# Regression helpers
# ----------------------------------------------------------------------
def linear_regression(X, y):
    """Compute coefficients β for Xβ = y using normal equations.
    X: list of rows (including bias column as last column).
    y: list of target values."""
    XTX = mat_mult(transpose(X), X)
    XTy = mat_vec_mult(transpose(X), y)
    beta = solve_linear_system(XTX, XTy)
    return beta, XTX

def predict_row(beta, features):
    """Predict target from beta and features (list, no bias)."""
    return sum(beta[i] * features[i] for i in range(len(features))) + beta[-1]

def prediction_variance(features, beta, XTX_inv, sigma2):
    """Compute variance of predicted value: sigma2 * (1 + fᵀ XTX_inv f)"""
    f = features + [1.0]  # add bias
    # Compute fᵀ XTX_inv f
    tmp = mat_vec_mult(XTX_inv, f)
    val = sum(tmp[i] * f[i] for i in range(len(f)))
    return sigma2 * (1.0 + val)

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def estimate_mpg(rows, n_est=3):
    # Build training data where mpg and predictors exist
    X = []
    y = []
    for r in rows:
        if r['Mpg+'] is not None and r['Lbs-'] is not None and r['Acc+'] is not None:
            X.append([r['Lbs-'], r['Acc+']])
            y.append(r['Mpg+'])
    if len(X) < 2:
        return []

    # Add bias column
    X_bias = [row + [1.0] for row in X]
    beta, XTX = linear_regression(X_bias, y)

    # Compute residual variance σ²
    predictions = [predict_row(beta, row) for row in X]
    residuals = [predictions[i] - y[i] for i in range(len(y))]
    sigma2 = sum(r**2 for r in residuals) / (len(y) - len(beta))

    # Inverse of XTX for prediction variance
    XTX_inv = mat_inv(XTX)

    # Find rows needing mpg estimate
    estimates = []
    for r in rows:
        if r['Mpg+'] is None and r['Lbs-'] is not None and r['Acc+'] is not None:
            est = predict_row(beta, [r['Lbs-'], r['Acc+']])
            se = math.sqrt(prediction_variance([r['Lbs-'], r['Acc+']], beta, XTX_inv, sigma2))
            estimates.append((est, se))
            if len(estimates) >= n_est:
                break
    return estimates

def estimate_weight(rows, n_est=3):
    # Build training data where weight and predictors exist
    X = []
    y = []
    for r in rows:
        if r['Lbs-'] is not None and r['Mpg+'] is not None and r['Acc+'] is not None:
            X.append([r['Mpg+'], r['Acc+']])
            y.append(r['Lbs-'])
    if len(X) < 2:
        return []

    # Add bias column
    X_bias = [row + [1.0] for row in X]
    beta, XTX = linear_regression(X_bias, y)

    # Compute residual variance σ²
    predictions = [predict_row(beta, row) for row in X]
    residuals = [predictions[i] - y[i] for i in range(len(y))]
    sigma2 = sum(r**2 for r in residuals) / (len(y) - len(beta))

    # Inverse of XTX for prediction variance
    XTX_inv = mat_inv(XTX)

    # Find rows needing weight estimate
    estimates = []
    for r in rows:
        if r['Lbs-'] is None and r['Mpg+'] is not None and r['Acc+'] is not None:
            est = predict_row(beta, [r['Mpg+'], r['Acc+']])
            se = math.sqrt(prediction_variance([r['Mpg+'], r['Acc+']], beta, XTX_inv, sigma2))
            estimates.append((est, se))
            if len(estimates) >= n_est:
                break
    return estimates

# ----------------------------------------------------------------------
def main(csv_path):
    rows = read_data(csv_path)
    if not rows:
        print("No data found.")
        return

    # CAR-03: fuel efficiency estimation
    mpg_estimates = estimate_mpg(rows, n_est=3)
    print("== CAR-03 ==")
    if mpg_estimates:
        for i, (est, se) in enumerate(mpg_estimates, start=1):
            print(f"{i}. Estimated Mpg+ = {est:.2f} ± {se:.2f} (SE)")
    else:
        print("No missing MPG data or insufficient training data.")

    # CAR-04: weight estimation
    weight_estimates = estimate_weight(rows, n_est=3)
    print("\n== CAR-04 ==")
    if weight_estimates:
        for i, (est, se) in enumerate(weight_estimates, start=1):
            print(f"{i}. Estimated Lbs- = {est:.1f} ± {se:.1f} (SE)")
    else:
        print("No missing weight data or insufficient training data.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
