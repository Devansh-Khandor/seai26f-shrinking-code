#!/usr/bin/env python3
"""
Skill 09: Outlier detection and regression‑based anomaly flagging for used‑car dealership.

CAR-17: Flag vehicle records whose combinations of characteristics are unusually
different from the rest of our inventory so that staff can review them.

CAR-18: Identify vehicles whose reported fuel efficiency or weight differs
substantially from what similar vehicles would lead us to expect, so we can
check for possible recording mistakes.
"""

import csv
import sys
import math

# ----------------------------------------------------------------------
# Data parsing
# ----------------------------------------------------------------------
NUMERIC_KEYS = [
    'Clndrs', 'Volume', 'HpX', 'Model', 'origin',
    'Lbs-', 'Acc+', 'Mpg+'
]

def read_data(csv_path):
    """Read CSV and convert numeric columns to float or None."""
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in NUMERIC_KEYS:
                try:
                    val = row.get(k, '')
                    row[k] = float(val) if val != '' else None
                except (ValueError, KeyError):
                    row[k] = None
            rows.append(row)
    return rows

# ----------------------------------------------------------------------
# Basic linear algebra (no external libs)
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

def mat_inv(A):
    """Inverse of a square matrix using Gaussian elimination."""
    n = len(A)
    AM = [A[i][:] + [float(i == j) for j in range(n)] for i in range(n)]
    for i in range(n):
        pivot = AM[i][i]
        if abs(pivot) < 1e-12:
            # find non‑zero pivot
            for r in range(i+1, n):
                if abs(AM[r][i]) > 1e-12:
                    AM[i], AM[r] = AM[r], AM[i]
                    pivot = AM[i][i]
                    break
            else:
                raise ValueError("Matrix not invertible")
        for j in range(2*n):
            AM[i][j] /= pivot
        for r in range(n):
            if r == i:
                continue
            factor = AM[r][i]
            for j in range(2*n):
                AM[r][j] -= factor * AM[i][j]
    return [row[n:] for row in AM]

def mat_vec_mult(A, v):
    """Multiply matrix A (m x n) with vector v (n)."""
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def solve_linear_system(A, b):
    """Solve A x = b."""
    return mat_vec_mult(mat_inv(A), b)

# ----------------------------------------------------------------------
# Simple linear regression using normal equations
# ----------------------------------------------------------------------
def linear_regression(X, y):
    """
    X: list of rows, each row is a list of predictors (excluding bias)
    y: list of target values
    Returns beta (list of coefficients, last element is bias) and XTX.
    """
    X_bias = [row + [1.0] for row in X]
    XTX = mat_mult(transpose(X_bias), X_bias)
    XTy = mat_vec_mult(transpose(X_bias), y)
    beta = solve_linear_system(XTX, XTy)
    return beta, XTX

def predict(beta, features):
    """Predict target from beta and features (no bias)."""
    return sum(beta[i] * features[i] for i in range(len(features))) + beta[-1]

# ----------------------------------------------------------------------
# CAR-17: Multivariate outlier detection
# ----------------------------------------------------------------------
def car_17(rows):
    print("== CAR-17 ==")

    # Gather numeric values
    weights = [r['Lbs-'] for r in rows if r['Lbs-'] is not None]
    mpgs   = [r['Mpg+'] for r in rows if r['Mpg+'] is not None]
    accs   = [r['Acc+'] for r in rows if r['Acc+'] is not None]

    if not (weights and mpgs and accs):
        print("Insufficient data for outlier analysis.")
        return

    wmin, wmax = min(weights), max(weights)
    mmin, mmax = min(mpgs), max(mpgs)
    amin, amax = min(accs), max(accs)

    # Normalise to [0,1]
    def norm(v, mn, mx):
        return (v - mn) / (mx - mn) if mx != mn else 0.5

    distances = []
    for idx, r in enumerate(rows, start=1):
        if r['Lbs-'] is None or r['Mpg+'] is None or r['Acc+'] is None:
            continue
        nw = norm(r['Lbs-'], wmin, wmax)
        nm = norm(r['Mpg+'], mmin, mmax)
        na = norm(r['Acc+'], amin, amax)
        # Euclidean distance from centre (0.5,0.5,0.5)
        d = math.sqrt((nw - 0.5)**2 + (nm - 0.5)**2 + (na - 0.5)**2)
        distances.append((d, idx, r))

    # Take top 3 largest distances
    distances.sort(key=lambda x: x[0], reverse=True)
    top3 = distances[:3]

    if not top3:
        print("No records with complete data to evaluate.")
        return

    print("Top outlier candidates (distance from median in normalised space):")
    for d, idx, r in top3:
        print(f"  Row #{idx} – Dist={d:.3f} | Weight={r['Lbs-']:.1f} | MPG={r['Mpg+']:.1f} | Acc={r['Acc+']:.1f}")

# ----------------------------------------------------------------------
# CAR-18: Regression‑based anomaly detection
# ----------------------------------------------------------------------
def car_18(rows):
    print("== CAR-18 ==")

    # Prepare data for mpg regression (predict mpg from weight & accel)
    X_mpg, y_mpg = [], []
    for r in rows:
        if r['Mpg+'] is not None and r['Lbs-'] is not None and r['Acc+'] is not None:
            X_mpg.append([r['Lbs-'], r['Acc+']])
            y_mpg.append(r['Mpg+'])
    if len(X_mpg) < 2:
        print("Insufficient data to build mpg regression.")
        return

    beta_mpg, _ = linear_regression(X_mpg, y_mpg)

    # Prepare data for weight regression (predict weight from mpg & accel)
    X_wgt, y_wgt = [], []
    for r in rows:
        if r['Lbs-'] is not None and r['Mpg+'] is not None and r['Acc+'] is not None:
            X_wgt.append([r['Mpg+'], r['Acc+']])
            y_wgt.append(r['Lbs-'])
    if len(X_wgt) < 2:
        print("Insufficient data to build weight regression.")
        return

    beta_wgt, _ = linear_regression(X_wgt, y_wgt)

    # Compute residuals for each record that has both actual values
    residuals = []
    for idx, r in enumerate(rows, start=1):
        if r['Lbs-'] is None or r['Mpg+'] is None or r['Acc+'] is None:
            continue

        # Predicted values
        pred_mpg = predict(beta_mpg, [r['Lbs-'], r['Acc+']])
        pred_wgt = predict(beta_wgt, [r['Mpg+'], r['Acc+']])

        res_mpg = abs(r['Mpg+'] - pred_mpg)
        res_wgt = abs(r['Lbs-'] - pred_wgt)

        # Use the larger of the two residuals as anomaly score
        score = max(res_mpg, res_wgt)
        residuals.append((score, idx, r, res_mpg, res_wgt))

    if not residuals:
        print("No records with complete data for regression comparison.")
        return

    # Sort by descending anomaly score
    residuals.sort(key=lambda x: x[0], reverse=True)
    top3 = residuals[:3]

    print("Top anomaly candidates (large deviation from regression predictions):")
    for score, idx, r, res_mpg, res_wgt in top3:
        print(f"  Row #{idx} – Anomaly={score:.2f} | Weight={r['Lbs-']:.1f} (res={res_wgt:.1f}) | MPG={r['Mpg+']:.1f} (res={res_mpg:.2f}) | Acc={r['Acc+']:.1f}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main(csv_path):
    try:
        rows = read_data(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        sys.exit(1)

    if not rows:
        print("No data found in CSV.")
        return

    car_17(rows)
    car_18(rows)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
