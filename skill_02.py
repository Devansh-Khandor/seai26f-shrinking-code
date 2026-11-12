#!/usr/bin/env python3
"""
Skill 02: Missing MPG and Weight estimation
===========================================

This script reads the Auto93 dataset and prints two reports:

* CAR-03 – Estimate missing MPG values using a linear model trained on
  the historical records.

* CAR-04 – Estimate vehicle weight from other available attributes and
  provide a standard‑deviation estimate of the prediction uncertainty.

Usage:
    python skill_02.py <csv_path>
"""

import csv
import sys
import math
from collections import defaultdict

# ----------------------------------------------------------------------
# Helper functions for linear algebra (no external libraries)
# ----------------------------------------------------------------------
def transpose(m):
    """Return the transpose of a matrix (list of lists)."""
    return [list(row) for row in zip(*m)]

def matmul(a, b):
    """Multiply two matrices a and b."""
    result = [[0.0] * len(b[0]) for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            s = 0.0
            for k in range(len(a[0])):
                s += a[i][k] * b[k][j]
            result[i][j] = s
    return result

def invert_matrix(m):
    """Return the inverse of a square matrix using Gauss‑Jordan."""
    n = len(m)
    # Augment with identity
    a = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(m)]
    # Forward elimination
    for i in range(n):
        # Find pivot
        pivot = i
        for r in range(i + 1, n):
            if abs(a[r][i]) > abs(a[pivot][i]):
                pivot = r
        if abs(a[pivot][i]) < 1e-12:
            raise ValueError("Matrix is singular and cannot be inverted.")
        # Swap rows if needed
        if pivot != i:
            a[i], a[pivot] = a[pivot], a[i]
        # Normalize pivot row
        pv = a[i][i]
        for j in range(2 * n):
            a[i][j] /= pv
        # Eliminate other rows
        for r in range(n):
            if r == i:
                continue
            factor = a[r][i]
            for j in range(2 * n):
                a[r][j] -= factor * a[i][j]
    # Extract right half
    inv = [row[n:] for row in a]
    return inv

# ----------------------------------------------------------------------
# Data handling
# ----------------------------------------------------------------------
def read_csv(csv_path):
    """
    Read the CSV file and return a list of dictionaries.  Numeric values
    are converted to floats; missing or non‑numeric entries become None.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            parsed = {}
            for key in ("Clndrs", "Volume", "HpX", "Model", "origin",
                        "Lbs-", "Acc+", "Mpg+"):
                try:
                    parsed[key] = float(r[key])
                except (KeyError, ValueError, TypeError):
                    parsed[key] = None
            rows.append(parsed)
    return rows

# ----------------------------------------------------------------------
# Regression utilities
# ----------------------------------------------------------------------
def build_design_matrix(rows, feature_keys, target_key=None):
    """
    Build X matrix and y vector for linear regression.

    :param rows: list of dicts
    :param feature_keys: list of keys to use as features
    :param target_key: key of the target variable; if None, y is omitted
    :return: X (list of lists), y (list) if target_key provided
    """
    X = []
    y = []
    for r in rows:
        if target_key is not None and r[target_key] is None:
            continue  # skip rows without target
        # For features that might be missing, treat None as 0
        row = [1.0]  # intercept
        for k in feature_keys:
            row.append(r.get(k, 0.0) or 0.0)
        X.append(row)
        if target_key is not None:
            y.append([r[target_key]])
    return X, y

def linear_regression(X, y):
    """
    Compute linear regression coefficients using OLS:
    beta = (XᵀX)⁻¹ Xᵀ y

    :param X: design matrix (n_samples x n_features)
    :param y: target vector (n_samples x 1)
    :return: beta as a list of floats
    """
    Xt = transpose(X)
    XtX = matmul(Xt, X)
    try:
        XtX_inv = invert_matrix(XtX)
    except ValueError:
        raise RuntimeError("Singular matrix encountered during regression.")
    Xty = matmul(Xt, y)
    beta_mat = matmul(XtX_inv, Xty)
    beta = [row[0] for row in beta_mat]
    return beta

def predict(X, beta):
    """Predict y values given design matrix X and coefficients beta."""
    return matmul(X, [[b] for b in beta])

def residuals(y_true, y_pred):
    """Return residuals (true - predicted)."""
    return [t[0] - p[0] for t, p in zip(y_true, y_pred)]

def std_dev(resid, df):
    """Calculate standard deviation of residuals given degrees of freedom."""
    s2 = sum(r ** 2 for r in resid) / df
    return math.sqrt(s2)

# ----------------------------------------------------------------------
# CAR-03: Estimate missing MPG
# ----------------------------------------------------------------------
def car_03(rows):
    # Features: Lbs-, HpX, Acc+
    feature_keys = ["Lbs-", "HpX", "Acc+"]
    X, y = build_design_matrix(rows, feature_keys, target_key="Mpg+")
    beta = linear_regression(X, y)

    # Find rows with missing Mpg+
    missing_rows = [r for r in rows if r["Mpg+"] is None]
    predictions = []
    if missing_rows:
        X_missing, _ = build_design_matrix(missing_rows, feature_keys)
        preds = predict(X_missing, beta)
        predictions = [(r, preds[i][0]) for i, r in enumerate(missing_rows)]

    # Prepare output
    out = []
    out.append(f"Number of vehicles with missing MPG: {len(missing_rows)}")
    if predictions:
        out.append("Estimated MPG for first 3 missing vehicles:")
        for r, mpg in predictions[:3]:
            out.append(f"  {mpg:.1f} mpg")
    else:
        out.append("No missing MPG values found.")
    out.append("Regression model: MPG = {:.3f}".format(beta[0]))
    for i, key in enumerate(feature_keys, start=1):
        out.append(f"          + ({beta[i]:+.3f}) * {key}")
    return out

# ----------------------------------------------------------------------
# CAR-04: Estimate weight and uncertainty
# ----------------------------------------------------------------------
def car_04(rows):
    # Use all rows to build regression: target = Lbs-
    feature_keys = ["HpX", "Acc+", "Volume", "Clndrs", "Model", "origin"]
    X, y = build_design_matrix(rows, feature_keys, target_key="Lbs-")
    beta = linear_regression(X, y)

    # Predict all weights
    preds = predict(X, beta)
    resid = residuals(y, preds)
    df = len(rows) - len(beta)  # degrees of freedom
    sigma = std_dev(resid, df)

    # Build predictions for each row
    X_all = build_design_matrix(rows, feature_keys)[0]
    preds_all = predict(X_all, beta)

    out = []
    out.append(f"Standard deviation of weight prediction errors: {sigma:.2f} lbs")
    out.append("First 3 vehicles: Predicted weight (±σ) ")
    for i in range(3):
        pred = preds_all[i][0]
        out.append(f"  Vehicle {i+1}: {pred:.1f} ± {sigma:.1f} lbs")
    out.append("Regression model: Weight = {:.3f}".format(beta[0]))
    for j, key in enumerate(feature_keys, start=1):
        out.append(f"          + ({beta[j]:+.3f}) * {key}")
    return out

# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        rows = read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        sys.exit(1)

    print("== CAR-03 ==")
    for line in car_03(rows):
        print(line)

    print("\n== CAR-04 ==")
    for line in car_04(rows):
        print(line)

if __name__ == "__main__":
    main()
