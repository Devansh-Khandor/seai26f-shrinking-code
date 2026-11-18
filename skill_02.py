#!/usr/bin/env python3
"""
Skill 02: Predict missing Mpg+ and Lbs- values

Usage:
    python skill_02.py <path to auto93.csv>
"""

import csv
import sys
import json
import math
import statistics
from skill_01 import read_csv  # reuse CSV parsing logic

# ----------------------------------------------------------------------
# Helper matrix functions

def matrix_transpose(m):
    return [list(col) for col in zip(*m)]

def matrix_multiply(a, b):
    """Multiply matrix a (m x n) with matrix b (n x p)."""
    m, n = len(a), len(a[0])
    n2, p = len(b), len(b[0])
    assert n == n2
    res = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for k in range(n):
            aik = a[i][k]
            for j in range(p):
                res[i][j] += aik * b[k][j]
    return res

def vector_matrix_multiply(v, m):
    """Multiply vector v (length n) with matrix m (n x p)."""
    n = len(v)
    p = len(m[0])
    res = [0.0] * p
    for i in range(n):
        vi = v[i]
        for j in range(p):
            res[j] += vi * m[i][j]
    return res

def invert_matrix(m):
    """Invert a square matrix using Gauss‑Jordan elimination."""
    n = len(m)
    # create augmented matrix [m | I]
    aug = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(m)]
    # forward elimination
    for i in range(n):
        # pivot
        pivot_row = None
        for r in range(i, n):
            if abs(aug[r][i]) > 1e-12:
                pivot_row = r
                break
        if pivot_row is None:
            raise ValueError("Matrix is singular")
        if pivot_row != i:
            aug[i], aug[pivot_row] = aug[pivot_row], aug[i]
        pivot = aug[i][i]
        # normalize pivot row
        for col in range(2*n):
            aug[i][col] /= pivot
        # eliminate other rows
        for r in range(n):
            if r != i:
                factor = aug[r][i]
                if abs(factor) > 1e-12:
                    for col in range(2*n):
                        aug[r][col] -= factor * aug[i][col]
    # extract inverse
    inv = [row[n:] for row in aug]
    return inv

# ----------------------------------------------------------------------
# Linear regression helper

def linear_regression(X, y):
    """
    Fit ordinary least squares.
    X: list of lists, each inner list is a row with intercept first
    y: list of target values
    Returns weight vector.
    """
    XT = matrix_transpose(X)
    XTX = matrix_multiply(XT, X)
    XTy = vector_matrix_multiply(y, XT)
    XTX_inv = invert_matrix(XTX)
    weights = vector_matrix_multiply(XTy, XTX_inv)
    return weights

def predict_row(row_features, weights):
    """row_features: list of feature values (intercept first)"""
    return sum(f * w for f, w in zip(row_features, weights))

# ----------------------------------------------------------------------
# Prediction logic for CAR-03 and CAR-04

def get_numeric_features(row, exclude):
    """Return dict of feature name -> float for all numeric columns excluding 'exclude'."""
    return {k: float(v) for k, v in row.items() if k != exclude and v is not None}

def train_and_predict(target_col, all_rows, held_out_rows):
    """
    Train linear regression on all_rows (excluding held_out_rows) to predict target_col.
    Return predictions dict and mae over held_out_rows where target is present.
    """
    # Determine feature columns: all numeric columns except target
    feature_names = [k for k in all_rows[0].keys() if k != target_col]
    # Build training data
    X_train = []
    y_train = []
    for row in all_rows:
        if row in held_out_rows:
            continue
        if row.get(target_col) is None:
            continue
        # Ensure all features are present
        features = {}
        skip = False
        for fn in feature_names:
            val = row.get(fn)
            if val is None:
                skip = True
                break
            features[fn] = float(val)
        if skip:
            continue
        # Build row with intercept
        X_train.append([1.0] + [features[fn] for fn in feature_names])
        y_train.append(row[target_col])

    if not X_train:
        raise RuntimeError(f"No training data for {target_col}")

    weights = linear_regression(X_train, y_train)

    # Predict on held-out rows
    predictions = {}
    mae_n = 0
    mae_sum = 0.0
    for row in held_out_rows:
        # Build feature row; if any missing, skip prediction
        features = {}
        skip = False
        for fn in feature_names:
            val = row.get(fn)
            if val is None:
                skip = True
                break
            features[fn] = float(val)
        if skip:
            continue
        pred = predict_row([1.0] + [features[fn] for fn in feature_names], weights)
        row_no = all_rows.index(row) + 1  # 1-based
        predictions[str(row_no)] = pred
        if row.get(target_col) is not None:
            mae_sum += abs(pred - row[target_col])
            mae_n += 1
    mae = mae_sum / mae_n if mae_n > 0 else 0.0
    return predictions, mae

def train_and_predict_with_uncertainty(target_col, all_rows, held_out_rows):
    """
    Train linear regression for target_col and return predictions with uncertainty.
    Uncertainty is +/- residual standard deviation (sigma).
    """
    # Determine feature columns
    feature_names = [k for k in all_rows[0].keys() if k != target_col]
    # Build training data
    X_train = []
    y_train = []
    for row in all_rows:
        if row in held_out_rows:
            continue
        if row.get(target_col) is None:
            continue
        features = {}
        skip = False
        for fn in feature_names:
            val = row.get(fn)
            if val is None:
                skip = True
                break
            features[fn] = float(val)
        if skip:
            continue
        X_train.append([1.0] + [features[fn] for fn in feature_names])
        y_train.append(row[target_col])

    if not X_train:
        raise RuntimeError(f"No training data for {target_col}")

    weights = linear_regression(X_train, y_train)

    # Compute residual standard deviation
    residuals = []
    for x_row, y_val in zip(X_train, y_train):
        pred = predict_row(x_row, weights)
        residuals.append(y_val - pred)
    sigma = math.sqrt(sum(r*r for r in residuals) / (len(y_train) - len(weights)))

    # Predict on held-out rows with uncertainty
    predictions = {}
    for row in held_out_rows:
        features = {}
        skip = False
        for fn in feature_names:
            val = row.get(fn)
            if val is None:
                skip = True
                break
            features[fn] = float(val)
        if skip:
            continue
        pred = predict_row([1.0] + [features[fn] for fn in feature_names], weights)
        low = pred - sigma
        high = pred + sigma
        row_no = all_rows.index(row) + 1
        predictions[str(row_no)] = {"estimate": pred, "low": low, "high": high}
    return predictions

# ----------------------------------------------------------------------
# Main execution

def main(csv_path):
    rows = read_csv(csv_path)

    # Identify held‑out rows (row number divisible by 5)
    held_out = [row for i, row in enumerate(rows, start=1) if i % 5 == 0]

    # CAR‑03: predict Mpg+
    mpg_predictions, mpg_mae = train_and_predict("Mpg+", rows, held_out)
    print("== CAR-03 ==")
    # Output at least three lines
    for i, (row_no, pred) in enumerate(mpg_predictions.items()):
        if i >= 3:
            break
        print(f"Row {row_no}: predicted Mpg+ = {pred:.3f}")
    print(f"ANSWER CAR-03 {json.dumps({'predictions': mpg_predictions, 'mae': mpg_mae})}")

    # CAR‑04: predict Lbs- with uncertainty
    lbs_predictions = train_and_predict_with_uncertainty("Lbs-", rows, held_out)
    print("\n== CAR-04 ==")
    for i, (row_no, data) in enumerate(lbs_predictions.items()):
        if i >= 3:
            break
        print(f"Row {row_no}: estimate Lbs- = {data['estimate']:.3f} "
              f"(low={data['low']:.3f} high={data['high']:.3f})")
    print(f"ANSWER CAR-04 {json.dumps({'predictions': lbs_predictions})}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
