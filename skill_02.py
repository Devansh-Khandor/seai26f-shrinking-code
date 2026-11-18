#!/usr/bin/env python3
"""
Skill 02: Fuel‑efficiency and weight estimation for used‑car inventory.

The script implements CAR‑03 (fuel‑efficiency prediction) and CAR‑04
(weight estimation with uncertainty).  It follows the exact
specifications in the assignment description.

Usage:
    python skill_02.py <csv_path>
"""

import sys
import csv
import json
import math

# ------------------------------------------------------------------
# Utility functions for linear algebra (no numpy)
# ------------------------------------------------------------------
def mat_transpose(A):
    return [[row[i] for row in A] for i in range(len(A[0]))]

def mat_mul(A, B):
    # multiply matrices A (m×n) and B (n×p)
    return [[sum(a*b for a,b in zip(row, col)) for col in zip(*B)] for row in A]

def mat_vec_mul(A, v):
    # multiply matrix A (m×n) by vector v (n)
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def mat_identity(n):
    return [[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]

def mat_inv(A):
    n = len(A)
    AM = [row[:] for row in A]
    IM = mat_identity(n)
    for fd in range(n):
        # pivot
        pivot = AM[fd][fd]
        if abs(pivot) < 1e-12:
            # swap with non‑zero row
            swap = None
            for r in range(fd+1, n):
                if abs(AM[r][fd]) > 1e-12:
                    swap = r
                    break
            if swap is None:
                raise ValueError("Singular matrix, cannot invert")
            AM[fd], AM[swap] = AM[swap], AM[fd]
            IM[fd], IM[swap] = IM[swap], IM[fd]
            pivot = AM[fd][fd]
        # normalize pivot row
        for j in range(n):
            AM[fd][j] /= pivot
            IM[fd][j] /= pivot
        # eliminate other rows
        for i in range(n):
            if i == fd:
                continue
            factor = AM[i][fd]
            for j in range(n):
                AM[i][j] -= factor * AM[fd][j]
                IM[i][j] -= factor * IM[fd][j]
    return IM

# ------------------------------------------------------------------
# Data handling
# ------------------------------------------------------------------
def read_csv(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for line_num, line in enumerate(reader, start=1):
            parsed = {}
            for k, v in line.items():
                if v == '?' or v is None:
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = None
            rows.append((line_num, parsed))  # line_num is 1‑based row index
    return fieldnames, rows

# ------------------------------------------------------------------
# Linear regression utilities
# ------------------------------------------------------------------
def fit_linear_regression(X, y):
    """
    X: list of feature vectors (including intercept column)
    y: list of target values
    Returns:
        w: weights (including intercept)
        inv_XtX: inverse of XᵀX
        sigma2: residual variance (n-d)
    """
    n = len(X)
    d = len(X[0])
    XT = mat_transpose(X)
    XtX = mat_mul(XT, X)
    inv_XtX = mat_inv(XtX)
    XtY = [sum(XT[i][j] * y[j] for j in range(n)) for i in range(d)]
    w = mat_vec_mul(inv_XtX, XtY)
    # predictions for training set
    y_pred = [sum(X[i][j] * w[j] for j in range(d)) for i in range(n)]
    residuals = [y[i] - y_pred[i] for i in range(n)]
    sigma2 = sum(r*r for r in residuals) / (n - d)
    return w, inv_XtX, sigma2

def predict_linear_regression(X_row, w, inv_XtX, sigma2):
    """
    Predict target value and 95% CI for a single feature vector (incl intercept)
    """
    d = len(w)
    y_hat = sum(X_row[i] * w[i] for i in range(d))
    # SE of prediction
    temp = mat_vec_mul(inv_XtX, X_row)
    scalar = sum(X_row[i] * temp[i] for i in range(d))
    se = math.sqrt(sigma2 * (1 + scalar))
    ci_low = y_hat - 1.96 * se
    ci_high = y_hat + 1.96 * se
    return y_hat, ci_low, ci_high

# ------------------------------------------------------------------
# Main processing
# ------------------------------------------------------------------
def main(csv_path):
    fieldnames, rows = read_csv(csv_path)
    num_rows = len(rows)
    # Determine held‑out rows (divisible by 5)
    heldout_nums = {num for num, _ in rows if num % 5 == 0}
    # Budget
    budget = min(30, num_rows // 5)

    # ------------------------------------------------------------------
    # CAR‑03: fuel‑efficiency prediction (Mpg+)
    # ------------------------------------------------------------------
    # Features for CAR‑03: all columns except Mpg+
    feat_names_c03 = [name for name in fieldnames if name != 'Mpg+']
    # Build training set: first `budget` rows that are not held‑out and have all needed values
    train_rows_c03 = []
    for num, rec in rows:
        if num % 5 == 0:
            continue
        if len(train_rows_c03) >= budget:
            break
        # ensure all features present
        if any(rec.get(name) is None for name in feat_names_c03 + ['Mpg+']):
            continue
        train_rows_c03.append((num, rec))
    # Prepare X and y for training
    X_train_c03 = []
    y_train_c03 = []
    for _, rec in train_rows_c03:
        X_row = [1.0]  # intercept
        X_row += [rec[name] for name in feat_names_c03]
        X_train_c03.append(X_row)
        y_train_c03.append(rec['Mpg+'])
    # Fit regression
    if X_train_c03:
        w_c03, inv_XtX_c03, sigma2_c03 = fit_linear_regression(X_train_c03, y_train_c03)
    else:
        w_c03 = inv_XtX_c03 = sigma2_c03 = None

    # Predictions for held‑out rows
    predictions_c03 = {}
    mae_sum = 0.0
    mae_cnt = 0
    # Output some sample predictions (first 3)
    sample_lines_c03 = []
    for idx, (num, rec) in enumerate(rows):
        if num not in heldout_nums:
            continue
        # Ensure we have all features needed (except target)
        if any(rec.get(name) is None for name in feat_names_c03):
            continue
        X_row = [1.0] + [rec[name] for name in feat_names_c03]
        pred_val = predict_linear_regression(X_row, w_c03, inv_XtX_c03, sigma2_c03)[0]
        predictions_c03[str(num)] = pred_val
        # true value
        true_val = rec['Mpg+']
        mae_sum += abs(pred_val - true_val)
        mae_cnt += 1
        if idx < 3:
            sample_lines_c03.append(f"Row {num}: {pred_val:.4f}")
    mae = mae_sum / mae_cnt if mae_cnt else None

    # ------------------------------------------------------------------
    # CAR‑04: weight prediction (Lbs-) with uncertainty
    # ------------------------------------------------------------------
    feat_names_c04 = [name for name in fieldnames if name != 'Lbs-']
    train_rows_c04 = []
    for num, rec in rows:
        if num % 5 == 0:
            continue
        if len(train_rows_c04) >= budget:
            break
        if any(rec.get(name) is None for name in feat_names_c04 + ['Lbs-']):
            continue
        train_rows_c04.append((num, rec))
    X_train_c04 = []
    y_train_c04 = []
    for _, rec in train_rows_c04:
        X_row = [1.0] + [rec[name] for name in feat_names_c04]
        X_train_c04.append(X_row)
        y_train_c04.append(rec['Lbs-'])
    if X_train_c04:
        w_c04, inv_XtX_c04, sigma2_c04 = fit_linear_regression(X_train_c04, y_train_c04)
    else:
        w_c04 = inv_XtX_c04 = sigma2_c04 = None

    predictions_c04 = {}
    sample_lines_c04 = []
    for idx, (num, rec) in enumerate(rows):
        if num not in heldout_nums:
            continue
        if any(rec.get(name) is None for name in feat_names_c04):
            continue
        X_row = [1.0] + [rec[name] for name in feat_names_c04]
        est, low, high = predict_linear_regression(X_row, w_c04, inv_XtX_c04, sigma2_c04)
        predictions_c04[str(num)] = {"estimate": est, "low": low, "high": high}
        if idx < 3:
            sample_lines_c04.append(f"Row {num}: estimate={est:.2f}, low={low:.2f}, high={high:.2f}")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    # CAR‑03
    print("== CAR-03 ==")
    for line in sample_lines_c03:
        print(line)
    print("ANSWER CAR-03 " + json.dumps({"predictions": predictions_c03, "mae": mae}, separators=(',', ':')))

    # CAR‑04
    print("== CAR-04 ==")
    for line in sample_lines_c04:
        print(line)
    print("ANSWER CAR-04 " + json.dumps({"predictions": predictions_c04}, separators=(',', ':')))

# ------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_02.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])
