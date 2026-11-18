#!/usr/bin/env python3
"""
Skill 02: Apache configuration prediction and comparison

This program implements two new analyses for the Apache configuration data:

* **APACHE-03** – Estimate the Performance- goal for held‑out rows using a
  linear model trained on the rest of the data.  It also reports the mean
  absolute error of the estimates.

* **APACHE-04** – For every pair of historically observed rows that differ
  in exactly one input setting, estimate the difference in Performance- and
  provide a simple uncertainty interval based on the residual error of the
  model.

The program prints a header line for each analysis, a few human‑readable
lines, and ends each section with a machine‑checked answer line that
contains a single‑line JSON object.  The JSON objects are derived
directly from the CSV file supplied on the command line and are never
hard‑coded.

The implementation uses only the Python standard library.  A tiny
linear‑regression solver based on normal equations and Gauss‑Jordan
elimination is written in‑line so that the code remains self‑contained.
"""

import csv
import json
import math
import sys
from typing import Dict, List, Tuple, Any

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Read the CSV and return header and list of rows as dicts."""
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        for row in rdr:
            row_dict = {}
            for h, val in zip(header, row):
                if val == "?":
                    row_dict[h] = None
                else:
                    # try int/float conversion
                    try:
                        if "." in val or "e" in val.lower():
                            row_dict[h] = float(val)
                        else:
                            row_dict[h] = int(val)
                    except ValueError:
                        row_dict[h] = val
            rows.append(row_dict)
    return header, rows


def is_goal(col: str) -> bool:
    return col.endswith("+") or col.endswith("-")


def is_input(col: str) -> bool:
    return not col.endswith("X") and not is_goal(col)


# ----------------------------------------------------------------------
# Linear regression helper (normal equations + Gauss‑Jordan)
# ----------------------------------------------------------------------
def _mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    n, m = len(A), len(B[0])
    p = len(B)
    return [[sum(A[i][k] * B[k][j] for k in range(p)) for j in range(m)] for i in range(n)]


def _mat_transpose(A: List[List[float]]) -> List[List[float]]:
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def _gauss_jordan(A: List[List[float]]) -> List[List[float]]:
    n = len(A)
    # Augment with identity
    M = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(A)]
    for col in range(n):
        # Pivot selection
        sel = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[sel][col]) < 1e-12:
            continue  # singular, skip
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
    return [row[n:] for row in M]  # extract inverse


def _solve_normal_equation(X: List[List[float]], y: List[float]) -> List[float]:
    # XTX = X^T * X
    m = len(X[0])
    XTX = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i, m):
            val = sum(X[k][i] * X[k][j] for k in range(len(X)))
            XTX[i][j] = XTX[j][i] = val
    # XTy
    XTy = [sum(X[k][i] * y[k] for k in range(len(X))) for i in range(m)]
    # Invert XTX
    inv = _gauss_jordan(XTX)
    # beta = inv * XTy
    return [sum(inv[i][j] * XTy[j] for j in range(m)) for i in range(m)]


# ----------------------------------------------------------------------
# Prediction model building
# ----------------------------------------------------------------------
def build_model(
    rows: List[Dict[str, Any]],
    input_cols: List[str],
    target_col: str,
    exclude_idx: set[int],
) -> Tuple[List[float], Dict[str, float], List[int]]:
    """
    Build linear regression model.

    :param rows: all rows (dicts)
    :param input_cols: list of input column names
    :param target_col: target column name
    :param exclude_idx: set of row indices (0‑based) to exclude from training
    :return: beta coefficients, feature means, indices of training rows
    """
    # Compute means for imputation
    means: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for col in input_cols:
        total = 0.0
        c = 0
        for idx, row in enumerate(rows):
            if idx in exclude_idx:
                continue
            v = row[col]
            if v is not None:
                total += v
                c += 1
        means[col] = total / c if c else 0.0
        counts[col] = c

    # Prepare training data
    X: List[List[float]] = []
    y: List[float] = []
    train_indices: List[int] = []

    for idx, row in enumerate(rows):
        if idx in exclude_idx:
            continue
        tgt = row[target_col]
        if tgt is None:
            continue  # cannot train without target
        vec = [1.0]  # intercept
        for col in input_cols:
            v = row[col]
            vec.append(v if v is not None else means[col])
        X.append(vec)
        y.append(tgt)
        train_indices.append(idx)

    beta = _solve_normal_equation(X, y)
    return beta, means, train_indices


def predict_row(beta: List[float], means: Dict[str, float], row: Dict[str, Any], input_cols: List[str]) -> float:
    vec = [1.0]
    for col in input_cols:
        v = row[col]
        vec.append(v if v is not None else means[col])
    return sum(b * x for b, x in zip(beta, vec))


# ----------------------------------------------------------------------
# Main analyses
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    header, rows = read_csv(csv_path)

    # Identify columns
    input_cols = [c for c in header if is_input(c)]
    goal_cols = [c for c in header if is_goal(c)]
    target_col = "Performance-"

    if target_col not in goal_cols:
        print(f"Target column {target_col} not found.")
        sys.exit(1)

    # Determine row numbers (1‑based)
    N = len(rows)
    heldout_indices = {i for i in range(N) if (i + 1) % 5 == 0}

    # ------------------------------------------------------------------
    # APACHE-03
    # ------------------------------------------------------------------
    print("== APACHE-03 ==")

    # Build model on non‑heldout rows
    beta, means, train_idx = build_model(rows, input_cols, target_col, heldout_indices)

    # Predict held‑out rows
    predictions: Dict[int, float] = {}
    mae_total = 0.0
    mae_count = 0
    for idx in sorted(heldout_indices):
        row = rows[idx]
        pred = predict_row(beta, means, row, input_cols)
        predictions[idx + 1] = pred  # 1‑based
        true_val = row[target_col]
        if true_val is not None:
            mae_total += abs(pred - true_val)
            mae_count += 1

    mae = mae_total / mae_count if mae_count else 0.0

    # Human‑readable output (at least 3 lines)
    for row_no, pred in sorted(predictions.items()):
        print(f"Row {row_no}: predicted Performance- = {pred:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")

    # Machine‑checked answer
    print(
        "ANSWER APACHE-03 "
        + json.dumps(
            {"predictions": {str(k): v for k, v in predictions.items()}, "mae": mae}
        )
    )

    # ------------------------------------------------------------------
    # APACHE-04
    # ------------------------------------------------------------------
    print("\n== APACHE-04 ==")

    # Compute residual standard deviation from training data
    residuals: List[float] = []
    for idx in train_idx:
        row = rows[idx]
        true_val = row[target_col]
        if true_val is None:
            continue
        pred = predict_row(beta, means, row, input_cols)
        residuals.append(true_val - pred)
    sigma = math.sqrt(sum(r * r for r in residuals) / len(residuals)) if residuals else 0.0
    interval = 2 * sigma  # simple 95% approx

    # Find pairs that differ in exactly one input setting
    pairs: List[Dict[str, Any]] = []

    for i in range(N):
        for j in range(i + 1, N):
            row_a, row_b = rows[i], rows[j]
            diff_cols = [c for c in input_cols if row_a[c] != row_b[c]]
            if len(diff_cols) == 1:
                setting = diff_cols[0]
                pred_a = predict_row(beta, means, row_a, input_cols)
                pred_b = predict_row(beta, means, row_b, input_cols)
                diff = pred_b - pred_a
                pair = {
                    "a": i + 1,
                    "b": j + 1,
                    "setting": setting,
                    "difference": diff,
                    "low": diff - interval,
                    "high": diff + interval,
                }
                pairs.append(pair)
                if len(pairs) >= 5:
                    break
        if len(pairs) >= 5:
            break

    # Human‑readable output (at least 5 lines)
    for p in pairs:
        print(
            f"Rows {p['a']} & {p['b']} differ in '{p['setting']}': "
            f"predicted diff = {p['difference']:.2f} (±{interval:.2f})"
        )

    # Machine‑checked answer
    print(
        "ANSWER APACHE-04 "
        + json.dumps(
            {"pairs": [
                {
                    "a": p["a"],
                    "b": p["b"],
                    "setting": p["setting"],
                    "difference": p["difference"],
                    "low": p["low"],
                    "high": p["high"],
                }
                for p in pairs
            ]}
        )
    )


# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
