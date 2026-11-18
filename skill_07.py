#!/usr/bin/env python3
"""
Skill 07: Apache configuration exploration with limited evaluation budget

This program implements two new analyses for the Apache configuration data:

* **APACHE-13** – Find a configuration with a low Performance- value while evaluating
  as few candidate configurations as reasonably possible.  The algorithm
  chooses rows to evaluate adaptively: it trains a linear regression model on
  the rows already evaluated and selects the next row that the model predicts
  to have the lowest performance value.

* **APACHE-14** – Recommend an evaluation order that discovers promising
  configurations early.  The order is built with the same adaptive strategy
  as in APACHE‑13, but the full order of up to the budget is output.

The program prints a header line for each analysis, a few human‑readable
lines, and ends each section with a machine‑checked answer line that
contains a single‑line JSON object.

The JSON objects are derived from the data in the file that is
supplied on the command line; they are never hard‑coded.
"""

import csv
import json
import math
import sys
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# CSV utilities
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[str], List[Dict[str, object]]]:
    """Read a CSV file into a header list and a list of row dictionaries."""
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows: List[Dict[str, object]] = []
        for row in rdr:
            d: Dict[str, object] = {}
            for h, val in zip(header, row):
                if val == "?":
                    d[h] = None
                else:
                    try:
                        if "." in val or "e" in val.lower():
                            d[h] = float(val)
                        else:
                            d[h] = int(val)
                    except ValueError:
                        d[h] = val
            rows.append(d)
    return header, rows


def is_goal(col: str) -> bool:
    return col.endswith("+") or col.endswith("-")


def is_input(col: str) -> bool:
    return not col.endswith("X") and not is_goal(col)


# ----------------------------------------------------------------------
# Linear regression helpers (normal equations + Gauss‑Jordan)
# ----------------------------------------------------------------------
def _mat_transpose(A: List[List[float]]) -> List[List[float]]:
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def _gauss_jordan(A: List[List[float]]) -> List[List[float]]:
    """Return inverse of square matrix A, or empty list if singular."""
    n = len(A)
    M = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(A)]
    for col in range(n):
        # Pivot selection
        sel = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[sel][col]) < 1e-12:
            return []  # singular
        # Swap
        M[col], M[sel] = M[sel], M[col]
        # Normalize pivot row
        div = M[col][col]
        M[col] = [x / div for x in M[col]]
        # Eliminate other rows
        for r in range(n):
            if r != col:
                factor = M[r][col]
                M[r] = [M[r][i] - factor * M[col][i] for i in range(2 * n)]
    return [row[n:] for row in M]  # inverse


def _solve_normal_equation(X: List[List[float]], y: List[float]) -> List[float]:
    """Solve for beta in linear regression using normal equations."""
    m = len(X[0])
    XTX = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i, m):
            val = sum(X[k][i] * X[k][j] for k in range(len(X)))
            XTX[i][j] = XTX[j][i] = val
    XTy = [sum(X[k][i] * y[k] for k in range(len(X))) for i in range(m)]
    inv = _gauss_jordan(XTX)
    if not inv:
        return [0.0] * m  # fallback
    return [sum(inv[i][j] * XTy[j] for j in range(m)) for i in range(m)]


# ----------------------------------------------------------------------
# Modeling utilities
# ----------------------------------------------------------------------
def train_regression(
    evaluated_rows: List[Tuple[int, Dict[str, object]]],
    input_cols: List[str],
) -> Tuple[List[float], Dict[str, float]]:
    """Train a linear regression on evaluated rows.
    Returns beta vector and feature means for imputation.
    """
    if not evaluated_rows:
        return [], {}
    # Compute feature means (for imputation)
    feature_means: Dict[str, float] = {}
    for col in input_cols:
        s = 0.0
        c = 0
        for _, row in evaluated_rows:
            v = row.get(col)
            if v is not None:
                s += v
                c += 1
        feature_means[col] = s / c if c else 0.0

    # Build X and y
    X: List[List[float]] = []
    y: List[float] = []
    for _, row in evaluated_rows:
        vec = [1.0]
        for col in input_cols:
            v = row.get(col)
            vec.append(v if v is not None else feature_means[col])
        X.append(vec)
        y.append(row[goal_col])  # goal_col set later
    beta = _solve_normal_equation(X, y)
    return beta, feature_means


def predict(beta: List[float], feature_means: Dict[str, float], row: Dict[str, object], input_cols: List[str]) -> float:
    vec = [1.0]
    for col in input_cols:
        v = row.get(col)
        vec.append(v if v is not None else feature_means.get(col, 0.0))
    return sum(b * x for b, x in zip(beta, vec))


# ----------------------------------------------------------------------
# Active selection algorithm
# ----------------------------------------------------------------------
def build_order(
    rows: List[Dict[str, object]],
    input_cols: List[str],
    budget: int,
) -> List[int]:
    """Return an order of row numbers (1‑based) to evaluate."""
    # Rows with a goal value
    rows_with_goal = [(idx + 1, row) for idx, row in enumerate(rows) if row.get(goal_col) is not None]
    remaining = rows_with_goal.copy()

    order: List[int] = []
    evaluated: List[Tuple[int, Dict[str, object]]] = []

    while len(order) < budget and remaining:
        if not evaluated:
            # pick first row by row number
            idx, row = remaining.pop(0)
            order.append(idx)
            evaluated.append((idx, row))
            continue

        beta, feature_means = train_regression(evaluated, input_cols)
        # Predict remaining rows
        best_pred = math.inf
        best_idx = None
        best_row = None
        for idx, row in remaining:
            pred = predict(beta, feature_means, row, input_cols)
            if pred < best_pred or (pred == best_pred and idx < best_idx):
                best_pred = pred
                best_idx = idx
                best_row = row
        # Remove selected from remaining
        remaining = [(i, r) for i, r in remaining if i != best_idx]
        order.append(best_idx)
        evaluated.append((best_idx, best_row))

    return order


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    goal_cols = [c for c in header if is_goal(c)]
    if not goal_cols:
        print("No goal columns found.")
        sys.exit(1)
    goal_col = goal_cols[0]  # use first goal column

    input_cols = [c for c in header if is_input(c)]

    budget = min(30, len(rows) // 5)

    # ------------------------------------------------------------------
    # APACHE-13
    # ------------------------------------------------------------------
    print("== APACHE-13 ==")

    order13 = build_order(rows, input_cols, budget)

    # Evaluate selected rows and find best
    evaluated_rows = [(idx, rows[idx - 1]) for idx in order13]
    best_row = min(evaluated_rows, key=lambda p: p[1][goal_col])[0] if evaluated_rows else None

    print(f"Evaluated rows (1‑based indices): {order13}")
    if best_row:
        print(f"Best row found: {best_row}")
    else:
        print("No evaluable rows found.")

    print(f"ANSWER APACHE-13 {json.dumps({'evaluated': order13, 'best': best_row})}")

    # ------------------------------------------------------------------
    # APACHE-14
    # ------------------------------------------------------------------
    print("\n== APACHE-14 ==")

    order14 = build_order(rows, input_cols, budget)

    print(f"Recommended evaluation order (1‑based indices): {order14}")

    print(f"ANSWER APACHE-14 {json.dumps({'order': order14})}")
