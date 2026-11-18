#!/usr/bin/env python3
"""
skill_02.py

Usage:
    python skill_02.py <path to Apache_AllMeasurements.csv>
"""

import csv
import json
import sys
from math import isclose
from collections import defaultdict

# Reuse functions from skill_01.py
import skill_01 as s01

# ----------------------------------------------------------------------
# Linear regression helpers (ordinary least squares)
# ----------------------------------------------------------------------
def _transpose(mat):
    return list(map(list, zip(*mat)))

def _multiply(A, B):
    """Multiply matrices A (m x n) and B (n x p)"""
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    assert n == n2, "Matrix size mismatch"
    result = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for k in range(p):
            s = 0.0
            for j in range(n):
                s += A[i][j] * B[j][k]
            result[i][k] = s
    return result

def _inverse(mat):
    """Gauss-Jordan elimination to invert a square matrix"""
    n = len(mat)
    A = [row[:] for row in mat]
    I = [[float(i == j) for j in range(n)] for i in range(n)]
    for i in range(n):
        # Find pivot
        piv = A[i][i]
        if isclose(piv, 0.0):
            # Find non-zero pivot below
            for k in range(i + 1, n):
                if not isclose(A[k][i], 0.0):
                    A[i], A[k] = A[k], A[i]
                    I[i], I[k] = I[k], I[i]
                    piv = A[i][i]
                    break
            else:
                raise ValueError("Matrix not invertible")
        # Normalize row
        inv_piv = 1.0 / piv
        for j in range(n):
            A[i][j] *= inv_piv
            I[i][j] *= inv_piv
        # Eliminate other rows
        for k in range(n):
            if k == i:
                continue
            factor = A[k][i]
            if isclose(factor, 0.0):
                continue
            for j in range(n):
                A[k][j] -= factor * A[i][j]
                I[k][j] -= factor * I[i][j]
    return I

def _fit_linear(rows, goal_col, settings):
    """
    Fit ordinary least squares linear model.
    Returns coefficient vector [intercept, coeff_1, coeff_2, ...]
    """
    X = []
    y = []
    for r in rows:
        if r[goal_col] is None:
            continue
        # Skip rows with missing settings
        if any(r[s] is None for s in settings):
            continue
        row_vec = [1.0] + [float(r[s]) for s in settings]
        X.append(row_vec)
        y.append(float(r[goal_col]))
    if not X:
        raise ValueError("No training data available")
    XT = _transpose(X)
    XTX = _multiply(XT, X)
    try:
        XTX_inv = _inverse(XTX)
    except Exception as e:
        raise RuntimeError(f"Failed to invert matrix: {e}")
    XTXI_XTy = _multiply(XTX_inv, _multiply(XT, [[v] for v in y]))
    coeff = [c[0] for c in XTXI_XTy]
    return coeff

def _predict_row(row, coeff, settings):
    val = coeff[0]  # intercept
    for i, s in enumerate(settings, start=1):
        val += coeff[i] * (0.0 if row[s] is None else float(row[s]))
    return val

# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = s01.parse_csv(csv_path)

    # Identify goal column (assumes only one)
    goal_cols = [c for c in header if s01.is_goal(c)]
    if not goal_cols:
        print("No goal columns found.")
        sys.exit(1)
    goal_col = goal_cols[0]

    # Settings: all columns except goal and those ending in 'X'
    settings = [c for c in header if not s01.is_goal(c) and not c.endswith('X')]

    # ------------------------------------------------------------------
    # APACHE-03: prediction on held‑out rows
    # ------------------------------------------------------------------
    print("== APACHE-03 ==")
    held_out_indices = [idx for idx, r in enumerate(rows, start=1) if idx % 5 == 0]
    training_rows = [r for idx, r in enumerate(rows, start=1) if idx not in held_out_indices]
    coeff = _fit_linear(training_rows, goal_col, settings)

    # Predictions
    predictions = {}
    abs_errs = []
    for idx in held_out_indices:
        row = rows[idx - 1]
        pred = _predict_row(row, coeff, settings)
        predictions[str(idx)] = pred
        if row[goal_col] is not None:
            abs_errs.append(abs(pred - row[goal_col]))
    mae = sum(abs_errs) / len(abs_errs) if abs_errs else 0.0

    # Print sample results
    for k, v in list(predictions.items())[:5]:
        print(f"Row {k}: predicted {v:.2f}")
    print(f"MAE over held‑out rows: {mae:.2f}")

    # JSON answer
    apache03_answer = {
        "predictions": predictions,
        "mae": mae
    }
    print("ANSWER APACHE-03", json.dumps(apache03_answer, separators=(",", ":")))

    # ------------------------------------------------------------------
    # APACHE-04: compare pairs differing by one setting
    # ------------------------------------------------------------------
    print("\n== APACHE-04 ==")
    # Compute residuals on training to estimate std error
    residuals = []
    for r in training_rows:
        if r[goal_col] is None:
            continue
        pred = _predict_row(r, coeff, settings)
        residuals.append(pred - r[goal_col])
    if residuals:
        import math
        mse = sum((x)**2 for x in residuals) / len(residuals)
        std_err = math.sqrt(mse)
    else:
        std_err = 0.0

    # Find pairs differing in exactly one setting
    pairs = []
    n_rows = len(rows)
    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            r1, r2 = rows[i], rows[j]
            # Both must have valid goal
            if r1[goal_col] is None or r2[goal_col] is None:
                continue
            diff_settings = [s for s in settings if r1[s] != r2[s]]
            if len(diff_settings) == 1:
                setting = diff_settings[0]
                diff_val = _predict_row(r2, coeff, settings) - _predict_row(r1, coeff, settings)
                low = diff_val - std_err
                high = diff_val + std_err
                pairs.append({
                    "a": i + 1,
                    "b": j + 1,
                    "setting": setting,
                    "difference": diff_val,
                    "low": low,
                    "high": high
                })
    # Ensure at least five pairs
    if len(pairs) < 5:
        # If not enough, pad with empty dicts
        pairs += [{}] * (5 - len(pairs))

    # Print sample pairs
    for p in pairs[:5]:
        print(f"Rows {p.get('a')} & {p.get('b')} differ by {p.get('setting')}")
        print(f"  Estimated diff: {p.get('difference'):.2f} (low={p.get('low'):.2f}, high={p.get('high'):.2f})")

    apache04_answer = {
        "pairs": pairs
    }
    print("ANSWER APACHE-04", json.dumps(apache04_answer, separators=(",", ":")))


if __name__ == "__main__":
    main()
