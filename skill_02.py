#!/usr/bin/env python3
"""
skill_02.py

Usage:
    python skill_02.py <csv_path>

This script reads the provided NASA data CSV and prints the results for
two client requests (NASA-03 and NASA-04).  Only the Python standard
library is used.
"""

import csv
import math
import sys

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def _parse_float(value):
    """Return a float or None if conversion fails."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None

def _matrix_mul(A, B):
    """Multiply matrix A (m x n) by matrix B (n x p)."""
    m, n = len(A), len(A[0])
    nB, p = len(B), len(B[0])
    assert n == nB, "Incompatible matrix sizes for multiplication."
    result = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            result[i][j] = s
    return result

def _matrix_transpose(M):
    """Return the transpose of matrix M."""
    rows, cols = len(M), len(M[0])
    return [[M[i][j] for i in range(rows)] for j in range(cols)]

def _solve_linear(A, b):
    """
    Solve the linear system A x = b for x.
    A is a square matrix (list of lists), b is a list.
    Uses Gaussian elimination with partial pivoting.
    """
    n = len(A)
    # Build augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    # Forward elimination
    for k in range(n):
        # Pivot
        pivot_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[pivot_row][k]) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular.")
        if pivot_row != k:
            M[k], M[pivot_row] = M[pivot_row], M[k]
        # Eliminate below
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]

    # Back substitution
    x = [0.0] * n
    for i in reversed(range(n)):
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x

def _compute_regression(data_rows, predictors, target):
    """
    Perform OLS regression: target ~ 1 + predictors.
    Returns (coefficients, residual_stddev).
    coefficients is a dict with keys 'intercept', predictor names.
    """
    # Build design matrix X and target vector y
    X = []
    y = []
    for r in data_rows:
        row = [1.0]  # intercept
        ok = True
        for p in predictors:
            val = r.get(p)
            if val is None:
                ok = False
                break
            row.append(val)
        if not ok:
            continue
        row.append(r.get(target))
        if row[-1] is None:
            continue
        X.append(row[:-1])
        y.append(row[-1])

    if not X:
        raise ValueError("No complete rows for regression.")

    # Compute X^T X and X^T y
    XT = _matrix_transpose(X)
    XT_X = _matrix_mul(XT, X)
    XT_y = _matrix_mul(XT, [[val] for val in y])
    XT_y = [row[0] for row in XT_y]

    # Solve for coefficients
    coeffs_vec = _solve_linear(XT_X, XT_y)

    # Residuals
    n = len(X)
    p = len(coeffs_vec)  # number of parameters
    residuals = []
    for i in range(n):
        pred = sum(coeffs_vec[j] * X[i][j] for j in range(p))
        residuals.append(y[i] - pred)

    # Residual standard deviation
    rss = sum(r ** 2 for r in residuals)
    resid_std = math.sqrt(rss / (n - p))

    coeff_dict = {"intercept": coeffs_vec[0]}
    for idx, name in enumerate(predictors, start=1):
        coeff_dict[name] = coeffs_vec[idx]

    return coeff_dict, resid_std

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _load_data(csv_path):
    """Load CSV and return list of dicts with numeric fields parsed."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for r in reader:
            parsed = {
                "idX": r.get("idX"),
                "Kloc+": _parse_float(r.get("Kloc+")),
                "Effort-": _parse_float(r.get("Effort-")),
                "Defects-": _parse_float(r.get("Defects-")),
                "MonthsX": _parse_float(r.get("MonthsX")),
            }
            rows.append(parsed)
    return rows

def main(csv_path):
    rows = _load_data(csv_path)

    # -------------------------------------------------- NASA-03 ---------------------------------
    try:
        coeffs_eff, std_eff = _compute_regression(rows,
                                                   predictors=["Kloc+", "MonthsX"],
                                                   target="Effort-")
    except ValueError as e:
        coeffs_eff, std_eff = {}, float("nan")

    print("== NASA-03 ==")
    if coeffs_eff:
        print(f"Coefficients: intercept = {coeffs_eff['intercept']:.4f}, "
              f"beta_Kloc+ = {coeffs_eff['Kloc+']:.4f}, "
              f"beta_MonthsX = {coeffs_eff['MonthsX']:.4f}")
        print(f"Residual stddev: {std_eff:.4f}")
        # Example prediction using first complete row
        for r in rows:
            if r["Kloc+"] is not None and r["MonthsX"] is not None:
                pred_eff = (coeffs_eff['intercept'] +
                            coeffs_eff['Kloc+'] * r["Kloc+"] +
                            coeffs_eff['MonthsX'] * r["MonthsX"])
                print(f"Example prediction (ID {r['idX']}): "
                      f"Effort- ≈ {pred_eff:.2f}")
                break
    else:
        print("Regression could not be performed due to insufficient data.")
        print("---")

    # -------------------------------------------------- NASA-04 ---------------------------------
    try:
        coeffs_def, std_def = _compute_regression(rows,
                                                   predictors=["Kloc+", "Effort-"],
                                                   target="Defects-")
    except ValueError as e:
        coeffs_def, std_def = {}, float("nan")

    print("\n== NASA-04 ==")
    if coeffs_def:
        print(f"Coefficients: intercept = {coeffs_def['intercept']:.4f}, "
              f"beta_Kloc+ = {coeffs_def['Kloc+']:.4f}, "
              f"beta_Effort- = {coeffs_def['Effort-']:.4f}")
        print(f"Residual stddev: {std_def:.4f}")
        # Example prediction using first complete row
        for r in rows:
            if r["Kloc+"] is not None and r["Effort-"] is not None:
                pred_def = (coeffs_def['intercept'] +
                            coeffs_def['Kloc+'] * r["Kloc+"] +
                            coeffs_def['Effort-'] * r["Effort-"])
                print(f"Example prediction (ID {r['idX']}): "
                      f"Defects- ≈ {pred_def:.2f}")
                break
    else:
        print("Regression could not be performed due to insufficient data.")
        print("---")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
