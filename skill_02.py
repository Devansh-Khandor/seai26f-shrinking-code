#!/usr/bin/env python3
"""
skill_02.py

Usage:
    python skill_02.py <csv_path>

Performs the following tasks for the Apache performance dataset:

* APACHE-03 – Predict the Performance- of held‑out rows (row numbers divisible by 5)
  using a simple linear regression model built from the remaining rows.
  Outputs the predictions and the mean absolute error (MAE).

* APACHE-04 – For every pair of historical rows that differ in exactly one setting
  column, estimate the performance difference (predicted Performance- of B minus A)
  and provide simple uncertainty bounds based on the MAE.

Author: OpenAI ChatGPT
"""

import csv
import sys
import json
import math
from collections import defaultdict

# --------------------------------------------------------------------------- #
# Helper functions for matrix operations (no external libraries)

def transpose(mat):
    return list(map(list, zip(*mat)))

def mat_mult(A, B):
    """Multiply matrices A (n x m) and B (m x p) -> (n x p)."""
    result = [[0.0] * len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for k in range(len(B)):
            aik = A[i][k]
            if aik == 0:
                continue
            for j in range(len(B[0])):
                result[i][j] += aik * B[k][j]
    return result

def mat_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def solve_linear_system(A, b):
    """
    Solve Ax = b for x using Gaussian elimination with partial pivoting.
    A is a square matrix, b is a column vector.
    Returns list of solutions.
    """
    n = len(A)
    # Create augmented matrix
    M = [A[i][:] + [b[i][0]] for i in range(n)]

    for k in range(n):
        # Pivot
        piv = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[piv][k]) < 1e-12:
            # Singular matrix – add a tiny epsilon to diagonal
            M[piv][k] += 1e-12
        if piv != k:
            M[k], M[piv] = M[piv], M[k]

        # Normalize pivot row
        div = M[k][k]
        for j in range(k, n + 1):
            M[k][j] /= div

        # Eliminate below
        for i in range(n):
            if i == k:
                continue
            factor = M[i][k]
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]

    # Extract solutions
    x = [row[-1] for row in M]
    return x

# --------------------------------------------------------------------------- #
# CSV reading utilities

def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    return header, rows

def is_goal(col):
    return col.endswith('+') or col.endswith('-')

def is_input(col):
    return not col.endswith('X') and not is_goal(col)

def parse_int(val):
    if val == '?' or val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None

# --------------------------------------------------------------------------- #
# Linear regression model

def build_regression(header, rows, feature_cols, target_col, impute_means):
    # Build design matrix X (n_samples x (p+1))
    X = []
    y = []
    for r in rows:
        target_val = parse_int(r.get(target_col))
        if target_val is None:
            continue
        features = [1]  # intercept
        for c in feature_cols:
            v = parse_int(r.get(c))
            if v is None:
                v = impute_means[c]
            features.append(v)
        X.append(features)
        y.append([target_val])  # column vector

    # Compute beta = (X^T X)^-1 X^T y
    X_T = transpose(X)
    XtX = mat_mult(X_T, X)
    # Solve XtX * beta = XtX * y? Actually we need (XtX)^-1 XtX y -> simpler: beta = solve(XtX, XtX*y)
    # Compute XtY
    XtY = mat_mult(X_T, y)
    beta = solve_linear_system(XtX, XtY)
    return beta  # list of coefficients

def predict(beta, impute_means, row, feature_cols):
    features = [1]
    for c in feature_cols:
        v = parse_int(row.get(c))
        if v is None:
            v = impute_means[c]
        features.append(v)
    # dot product
    val = sum(f * b for f, b in zip(features, beta))
    return val

# --------------------------------------------------------------------------- #
# Main logic for APACHE-03 and APACHE-04

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    # Identify columns
    target_col = next((c for c in header if c == 'Performance-'), None)
    if not target_col:
        print("No Performance- column found.", file=sys.stderr)
        sys.exit(1)

    feature_cols = [c for c in header if is_input(c)]

    # Compute impute means for each feature across all rows (excluding missing)
    impute_means = {}
    for c in feature_cols:
        total = 0
        count = 0
        for r in rows:
            v = parse_int(r.get(c))
            if v is not None:
                total += v
                count += 1
        impute_means[c] = (total / count) if count > 0 else 0

    # Split held-out rows (row number divisible by 5)
    held_out_indices = [i for i in range(len(rows)) if (i + 1) % 5 == 0]  # 0-based
    training_rows = [rows[i] for i in range(len(rows)) if i not in held_out_indices]
    held_out_rows = [rows[i] for i in held_out_indices]

    # Build regression model on training data
    beta = build_regression(header, training_rows, feature_cols, target_col, impute_means)

    # APACHE-03: predict held-out rows
    predictions = {}
    abs_errors = []
    for idx, r in zip(held_out_indices, held_out_rows):
        pred_val = predict(beta, impute_means, r, feature_cols)
        # Round to nearest integer for consistency
        pred_int = round(pred_val)
        predictions[str(idx + 1)] = pred_int  # row numbers are 1-based
        actual = parse_int(r.get(target_col))
        if actual is not None:
            abs_errors.append(abs(pred_int - actual))

    mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0

    print("== APACHE-03 ==")
    for row_num, pred in sorted(predictions.items(), key=lambda x: int(x[0])):
        print(f"Row {row_num} Prediction: {pred}")
    print(f"MAE: {mae:.2f}")
    print("ANSWER APACHE-03", json.dumps({"predictions": predictions, "mae": round(mae, 3)}))

    # --------------------------------------------------------------------- #
    # APACHE-04: pairs of rows differing in exactly one setting
    # Build predictions for all rows with valid target
    all_preds = {}
    for idx, r in enumerate(rows):
        target = parse_int(r.get(target_col))
        if target is not None:
            all_preds[idx + 1] = predict(beta, impute_means, r, feature_cols)

    # Find pairs
    pairs = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            r1 = rows[i]
            r2 = rows[j]
            # Both must have target
            if (i + 1) not in all_preds or (j + 1) not in all_preds:
                continue
            diff_count = 0
            diff_col = None
            for c in feature_cols:
                v1 = parse_int(r1.get(c))
                v2 = parse_int(r2.get(c))
                v1 = v1 if v1 is not None else impute_means[c]
                v2 = v2 if v2 is not None else impute_means[c]
                if v1 != v2:
                    diff_count += 1
                    diff_col = c
                if diff_count > 1:
                    break
            if diff_count == 1:
                a = i + 1
                b = j + 1
                diff_pred = round(all_preds[b] - all_preds[a], 3)
                low = diff_pred - mae
                high = diff_pred + mae
                pairs.append({
                    "a": a,
                    "b": b,
                    "setting": diff_col,
                    "difference": diff_pred,
                    "low": round(low, 3),
                    "high": round(high, 3)
                })

    # Output at least five pairs; if fewer, output all
    output_pairs = pairs[:5] if len(pairs) >= 5 else pairs

    print("\n== APACHE-04 ==")
    for p in output_pairs:
        print(f"Rows {p['a']} vs {p['b']}: setting={p['setting']}, diff={p['difference']:.3f} (low={p['low']:.3f}, high={p['high']:.3f})")
    print("ANSWER APACHE-04", json.dumps({"pairs": output_pairs}))

if __name__ == "__main__":
    main()
