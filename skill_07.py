#!/usr/bin/env python3
"""
skill_07.py

Implements APACHE-13 and APACHE-14.

- APACHE-13: sequentially evaluate at most 30 configurations using only previously
  evaluated goal values.  The algorithm starts with a small fixed sample, fits a
  linear regression on the available data, then repeatedly selects the un‑evaluated
  configuration with the lowest predicted goal value, evaluates it, and updates the
  model.  This continues until 30 rows have been evaluated (or all rows with a
  goal value have been examined).  The best (lowest) goal among the evaluated
  rows is reported.

- APACHE-14: the recommended order to evaluate configurations is simply the order
  discovered by the algorithm above.  It is a greedy strategy that uses only the
  goal values of configurations that have already been evaluated.

The output format follows the specification: a section header, at least three
result lines, and a single line with the JSON answer.
"""

import sys
import csv
import json
from math import isclose

def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, reader.fieldnames

def to_float(val):
    if val is None or val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def get_settings(columns, goal_col):
    return [c for c in columns
            if c != goal_col and not c.endswith('X')
            and not (c.endswith('+') or c.endswith('-'))]

def build_training_data(rows, evaluated, settings, goal_col):
    """
    Return X (list of lists, first column 1.0 for intercept) and y (list of floats)
    for rows that are in evaluated list and have a goal value and all settings.
    """
    X = []
    y = []
    for idx in evaluated:
        row = rows[idx-1]
        g = to_float(row[goal_col])
        if g is None:
            continue
        vals = [1.0]  # intercept
        ok = True
        for s in settings:
            v = to_float(row[s])
            if v is None:
                ok = False
                break
            vals.append(v)
        if not ok:
            continue
        X.append(vals)
        y.append(g)
    return X, y

def solve_linear(A, b):
    """
    Solve Ax = b using Gaussian elimination with partial pivoting.
    A: square matrix n x n, b: vector length n
    Returns x or None if matrix is singular.
    """
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    # Forward elimination
    for i in range(n):
        # pivot
        max_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[max_row][i]) < 1e-12:
            return None  # singular
        M[i], M[max_row] = M[max_row], M[i]
        pivot = M[i][i]
        for j in range(i, n+1):
            M[i][j] /= pivot
        for r in range(i+1, n):
            factor = M[r][i]
            for j in range(i, n+1):
                M[r][j] -= factor * M[i][j]
    # Back substitution
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        x[i] = M[i][n]
        for j in range(i+1, n):
            x[i] -= M[i][j] * x[j]
    return x

def fit_linear_regression(X, y):
    """
    Fit linear regression with intercept using normal equation.
    Returns coefficients list [intercept, beta1, beta2, ...].
    """
    n = len(X)
    if n == 0:
        return None
    # Compute X^T X and X^T y
    p = len(X[0])  # number of features incl. intercept
    XtX = [[0.0]*p for _ in range(p)]
    Xty = [0.0]*p
    for i in range(n):
        xi = X[i]
        yi = y[i]
        for a in range(p):
            Xty[a] += xi[a] * yi
            for b in range(a, p):
                XtX[a][b] += xi[a] * xi[b]
    # symmetrize
    for a in range(p):
        for b in range(a+1, p):
            XtX[b][a] = XtX[a][b]
    coeffs = solve_linear(XtX, Xty)
    return coeffs  # may be None

def predict(row, settings, coeffs):
    """
    Predict goal for a row given regression coefficients.
    coeffs: [intercept, beta1, ...]
    """
    if coeffs is None:
        return None
    val = coeffs[0]  # intercept
    for i, s in enumerate(settings):
        v = to_float(row[s])
        if v is None:
            return None
        val += coeffs[i+1] * v
    return val

def evaluate_rows(rows, goal_col, settings, max_eval=30):
    """
    Implements APACHE-13: sequential evaluation with linear regression.
    Returns evaluated indices list and best (index, value).
    """
    total_rows = len(rows)
    max_eval = min(max_eval, total_rows)
    evaluated = []

    # Start with a small fixed sample (first 5 rows or all rows if less)
    init_sample = min(5, total_rows)
    for idx in range(1, init_sample+1):
        evaluated.append(idx)

    best_idx = None
    best_val = None

    # Main loop
    while len(evaluated) < max_eval:
        # Fit regression on current evaluated data
        X, y = build_training_data(rows, evaluated, settings, goal_col)
        coeffs = fit_linear_regression(X, y)

        # Find next row to evaluate
        next_idx = None
        best_pred = None

        # Look for candidate rows not yet evaluated that have all settings and a goal value
        for idx in range(1, total_rows+1):
            if idx in evaluated:
                continue
            row = rows[idx-1]
            if to_float(row[goal_col]) is None:
                continue
            # ensure all settings present
            ok = True
            for s in settings:
                if to_float(row[s]) is None:
                    ok = False
                    break
            if not ok:
                continue

            pred = predict(row, settings, coeffs) if coeffs is not None else None
            if pred is None:
                continue
            if best_pred is None or pred < best_pred:
                best_pred = pred
                next_idx = idx

        # If no suitable candidate (e.g., all missing or coeffs None), pick next in order
        if next_idx is None:
            for idx in range(1, total_rows+1):
                if idx in evaluated:
                    continue
                if to_float(rows[idx-1][goal_col]) is not None:
                    next_idx = idx
                    break
            if next_idx is None:
                break  # nothing left

        evaluated.append(next_idx)

        # Update best
        g = to_float(rows[next_idx-1][goal_col])
        if best_idx is None or g < best_val:
            best_idx = next_idx
            best_val = g

    # After loop, find best among evaluated
    if best_idx is None:
        # compute over evaluated rows
        best_idx = min(evaluated,
                       key=lambda i: to_float(rows[i-1][goal_col]) or float('inf'))
        best_val = to_float(rows[best_idx-1][goal_col])
    return evaluated, best_idx, best_val

def main(csv_path):
    rows, columns = read_csv(csv_path)

    goal_col = None
    for c in columns:
        if c.endswith('+') or c.endswith('-'):
            goal_col = c
            break
    if goal_col is None:
        print("No goal column found.")
        sys.exit(1)

    settings = get_settings(columns, goal_col)

    # --- APACHE-13 ---
    evaluated, best_idx, best_val = evaluate_rows(rows, goal_col, settings, max_eval=30)

    print("== APACHE-13 ==")
    print(f"Evaluated rows ({len(evaluated)}): {evaluated}")
    print(f"Best row: {best_idx} (goal={best_val})")
    answer_13 = {"evaluated": evaluated, "best": best_idx}
    print("ANSWER APACHE-13 " + json.dumps(answer_13, separators=(',', ':')))

    # --- APACHE-14 ---
    # Recommended order is the same as evaluated order
    order = evaluated

    print("\n== APACHE-14 ==")
    print(f"Evaluation order ({len(order)} rows): {order}")
    answer_14 = {"order": order}
    print("ANSWER APACHE-14 " + json.dumps(answer_14, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
