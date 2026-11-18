#!/usr/bin/env python3
"""
skill_07.py: Implements CAR-13 and CAR-14 for used‑car dealership data.

Usage:
    python skill_07.py <csv_path>
"""

import csv
import json
import math
import sys
from typing import List, Tuple, Dict, Any

# ------------------------------------------------------------------
# CSV utilities
# ------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[str], List[List[Any]]]:
    """Read CSV, convert numeric cells, keep '?' as None."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows: List[List[Any]] = []
        for row in reader:
            conv_row: List[Any] = []
            for v in row:
                if v == "?":
                    conv_row.append(None)
                else:
                    try:
                        if "." in v or "e" in v.lower():
                            conv_row.append(float(v))
                        else:
                            conv_row.append(int(v))
                    except ValueError:
                        conv_row.append(v)
            rows.append(conv_row)
    return header, rows

# ------------------------------------------------------------------
# Goal/feature handling
# ------------------------------------------------------------------
def goal_indices(header: List[str]) -> List[int]:
    """Indices of goal columns (ending with '+' or '-')."""
    return [i for i, h in enumerate(header) if h.endswith(("+", "-"))]

def non_goal_numeric_indices(header: List[str], rows: List[List[Any]]) -> List[int]:
    """Indices of columns that are not goals and not 'X', and contain numeric data."""
    indices = []
    for i, h in enumerate(header):
        if h.endswith(("X", "+", "-")):
            continue
        # check if any row has numeric value at this column
        for r in rows:
            v = r[i]
            if v is not None and isinstance(v, (int, float)):
                indices.append(i)
                break
    return indices

# ------------------------------------------------------------------
# Normalized goal score
# ------------------------------------------------------------------
def normalized_score(row: List[Any], gidx: List[int], mins: Dict[int, float],
                     maxs: Dict[int, float], ismax: Dict[int, bool]) -> float:
    """Sum of normalized goal values for a row."""
    score = 0.0
    for idx in gidx:
        v = row[idx]
        if v is None:
            continue
        mn, mx = mins[idx], maxs[idx]
        if mn is None or mx is None or mx == mn:
            norm = 0.5
        else:
            if ismax[idx]:
                norm = (v - mn) / (mx - mn)
            else:
                norm = (mx - v) / (mx - mn)
        score += norm
    return score

def compute_goal_stats(rows: List[List[Any]], gidx: List[int]) -> Tuple[Dict[int, float], Dict[int, float]]:
    mins, maxs = {}, {}
    for idx in gidx:
        vals = [r[idx] for r in rows if r[idx] is not None]
        if vals:
            mins[idx] = min(vals)
            maxs[idx] = max(vals)
        else:
            mins[idx] = maxs[idx] = None
    return mins, maxs

# ------------------------------------------------------------------
# Linear regression helpers (closed‑form)
# ------------------------------------------------------------------
def matrix_inverse(A: List[List[float]]) -> List[List[float]]:
    """Inverse of a square matrix using Gauss–Jordan elimination."""
    n = len(A)
    M = [row[:] + [float(i == j) for i in range(n)] for j, row in enumerate(A)]
    for i in range(n):
        # Pivot
        pivot = i
        while pivot < n and abs(M[pivot][i]) < 1e-12:
            pivot += 1
        if pivot == n:
            raise ValueError("Singular matrix")
        if pivot != i:
            M[i], M[pivot] = M[pivot], M[i]
        div = M[i][i]
        M[i] = [v / div for v in M[i]]
        for r in range(n):
            if r != i:
                factor = M[r][i]
                M[r] = [rv - factor * iv for rv, iv in zip(M[r], M[i])]
    return [row[n:] for row in M]

def mat_vec_mul(A: List[List[float]], v: List[float]) -> List[float]:
    """Matrix times vector."""
    m = len(A)
    n = len(v)
    res = [0.0 for _ in range(m)]
    for i in range(m):
        s = 0.0
        for j in range(n):
            s += A[i][j] * v[j]
        res[i] = s
    return res

# ------------------------------------------------------------------
# Regression training and prediction
# ------------------------------------------------------------------
def train_linear_regression(X: List[List[float]], y: List[float]) -> List[float]:
    """Return coefficients beta for X * beta = y (intercept + features)."""
    # Compute X^T X
    n_features = len(X[0])
    XTX = [[0.0] * n_features for _ in range(n_features)]
    for xi in X:
        for i in range(n_features):
            for j in range(n_features):
                XTX[i][j] += xi[i] * xi[j]
    inv_XTX = matrix_inverse(XTX)
    # Compute X^T y
    XTy = [0.0] * n_features
    for xi, yi in zip(X, y):
        for i in range(n_features):
            XTy[i] += xi[i] * yi
    beta = mat_vec_mul(inv_XTX, XTy)
    return beta

def predict(beta: List[float], features: List[float]) -> float:
    return sum(b * f for b, f in zip(beta, features))

# ------------------------------------------------------------------
# CAR‑13 implementation
# ------------------------------------------------------------------
def car13(header: List[str], rows: List[List[Any]]) -> Tuple[List[int], int]:
    """Return (evaluated row numbers, best evaluated row)."""
    n_rows = len(rows)
    budget = min(30, n_rows // 5)
    gidx = goal_indices(header)
    ismax = {i: header[i].endswith("+") for i in gidx}
    mins_g, maxs_g = compute_goal_stats(rows, gidx)

    # numeric features indices (non‑goal, non‑X)
    fidx = non_goal_numeric_indices(header, rows)

    evaluated = []
    eval_set = set()

    # start with the first row
    evaluated.append(1)
    eval_set.add(1)

    # function to build X, y from evaluated rows
    def build_XY():
        X: List[List[float]] = []
        y: List[float] = []
        for rnum in evaluated:
            r = rows[rnum - 1]
            feat = [1.0] + [float(r[i]) if r[i] is not None else 0.0 for i in fidx]
            X.append(feat)
            y.append(normalized_score(r, gidx, mins_g, maxs_g, ismax))
        return X, y

    # If we have only one evaluated row, we cannot train; just pick random next
    while len(evaluated) < budget:
        if len(evaluated) <= len(fidx):
            # not enough data to fit, pick the next unexplored row with lowest index
            next_row = next(i for i in range(1, n_rows + 1) if i not in eval_set)
            evaluated.append(next_row)
            eval_set.add(next_row)
            continue

        X, y = build_XY()
        beta = train_linear_regression(X, y)

        # predict for all remaining rows
        best_pred = None
        best_row = None
        for rnum in range(1, n_rows + 1):
            if rnum in eval_set:
                continue
            r = rows[rnum - 1]
            feat = [1.0] + [float(r[i]) if r[i] is not None else 0.0 for i in fidx]
            pred = predict(beta, feat)
            if best_pred is None or pred > best_pred:
                best_pred = pred
                best_row = rnum
        if best_row is None:
            break
        evaluated.append(best_row)
        eval_set.add(best_row)

    # find best evaluated row by actual score
    best_score = -1e9
    best_row = None
    for rnum in evaluated:
        r = rows[rnum - 1]
        sc = normalized_score(r, gidx, mins_g, maxs_g, ismax)
        if sc > best_score:
            best_score = sc
            best_row = rnum

    return evaluated, best_row

# ------------------------------------------------------------------
# CAR‑14 implementation
# ------------------------------------------------------------------
def car14(header: List[str], rows: List[List[Any]]) -> List[int]:
    """Return evaluation order (first 30 rows)."""
    n_rows = len(rows)
    order = []
    budget = min(30, n_rows // 5)

    gidx = goal_indices(header)
    ismax = {i: header[i].endswith("+") for i in gidx}
    mins_g, maxs_g = compute_goal_stats(rows, gidx)

    fidx = non_goal_numeric_indices(header, rows)

    placed = set()

    # start with row 1
    order.append(1)
    placed.add(1)

    while len(order) < budget:
        if len(order) <= len(fidx):
            # not enough data to fit, pick the next unused row
            next_row = next(i for i in range(1, n_rows + 1) if i not in placed)
            order.append(next_row)
            placed.add(next_row)
            continue

        # train on already placed rows (assuming we will know their goal values)
        X: List[List[float]] = []
        y: List[float] = []
        for rnum in order:
            r = rows[rnum - 1]
            feat = [1.0] + [float(r[i]) if r[i] is not None else 0.0 for i in fidx]
            X.append(feat)
            y.append(normalized_score(r, gidx, mins_g, maxs_g, ismax))
        beta = train_linear_regression(X, y)

        best_pred = None
        best_row = None
        for rnum in range(1, n_rows + 1):
            if rnum in placed:
                continue
            r = rows[rnum - 1]
            feat = [1.0] + [float(r[i]) if r[i] is not None else 0.0 for i in fidx]
            pred = predict(beta, feat)
            if best_pred is None or pred > best_pred:
                best_pred = pred
                best_row = rnum
        if best_row is None:
            break
        order.append(best_row)
        placed.add(best_row)

    return order

# ------------------------------------------------------------------
# Main driver
# ------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    # CAR-13
    print("== CAR-13 ==")
    evaluated, best_row = car13(header, rows)
    # print at least three lines of results
    for rnum in evaluated:
        r = rows[rnum - 1]
        gidx = goal_indices(header)
        mins_g, maxs_g = compute_goal_stats(rows, gidx)
        ismax = {i: header[i].endswith("+") for i in gidx}
        score = normalized_score(r, gidx, mins_g, maxs_g, ismax)
        print(f"Evaluated row {rnum}: score {round(score,4)}")
    print(f"Best evaluated row: {best_row}")
    print(f"ANSWER CAR-13 {json.dumps({'evaluated': evaluated, 'best': best_row}, separators=(',', ':'))}")

    # CAR-14
    print("\n== CAR-14 ==")
    order = car14(header, rows)
    for i, rnum in enumerate(order, start=1):
        print(f"Step {i}: row {rnum}")
    print(f"ANSWER CAR-14 {json.dumps({'order': order}, separators=(',', ':'))}")

if __name__ == "__main__":
    main()
