#!/usr/bin/env python3
"""
CAR-03: Predict missing Mpg+ values from vehicle characteristics.
CAR-04: Predict missing Lbs- values and give a confidence interval.
"""

import csv
import json
import math
import sys
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------
# Utility functions for linear algebra
# ----------------------------------------------------------------------
def transpose(m: List[List[float]]) -> List[List[float]]:
    return [list(r) for r in zip(*m)]


def mat_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    rows, cols, inner = len(a), len(b[0]), len(a[0])
    result = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            aik = a[i][k]
            if aik == 0:
                continue
            for j in range(cols):
                result[i][j] += aik * b[k][j]
    return result


def solve_linear(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """
    Solve Ax = b via Gaussian elimination with partial pivoting.
    Returns None if matrix is singular.
    """
    n = len(a)
    # Augment
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for i in range(n):
        # Pivot
        max_row = max(range(i, n), key=lambda r: abs(aug[r][i]))
        if abs(aug[max_row][i]) < 1e-12:
            return None
        aug[i], aug[max_row] = aug[max_row], aug[i]
        # Normalize
        pivot = aug[i][i]
        for j in range(i, n + 1):
            aug[i][j] /= pivot
        # Eliminate
        for r in range(n):
            if r == i:
                continue
            factor = aug[r][i]
            for j in range(i, n + 1):
                aug[r][j] -= factor * aug[i][j]
    return [aug[i][-1] for i in range(n)]


def invert_matrix(a: List[List[float]]) -> Optional[List[List[float]]]:
    """
    Compute inverse of a square matrix a via Gauss-Jordan elimination.
    Returns None if singular.
    """
    n = len(a)
    aug = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(a)]
    for i in range(n):
        # Pivot
        max_row = max(range(i, n), key=lambda r: abs(aug[r][i]))
        if abs(aug[max_row][i]) < 1e-12:
            return None
        aug[i], aug[max_row] = aug[max_row], aug[i]
        # Normalize
        pivot = aug[i][i]
        for j in range(2 * n):
            aug[i][j] /= pivot
        # Eliminate
        for r in range(n):
            if r == i:
                continue
            factor = aug[r][i]
            for j in range(2 * n):
                aug[r][j] -= factor * aug[i][j]
    inv = [row[n:] for row in aug]
    return inv


# ----------------------------------------------------------------------
# Data handling
# ----------------------------------------------------------------------
def parse_csv(path: str) -> Tuple[List[Dict[str, Optional[float]]], List[str]]:
    """
    Return list of rows (dict column->value or None) and header names.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed = {}
            for k, v in row.items():
                v = v.strip()
                parsed[k] = None if v == "?" else float(v)
            rows.append(parsed)
        header = reader.fieldnames
    return rows, header


def is_numeric(rows: List[Dict[str, Optional[float]]], col: str) -> bool:
    return all(isinstance(r[col], (int, float)) for r in rows if r[col] is not None)


def get_feature_columns(
    header: List[str], target: str, exclude_x: bool = True
) -> List[str]:
    feats = []
    for c in header:
        if c == target:
            continue
        if exclude_x and c.endswith("X"):
            continue
        if is_numeric(rows, c):
            feats.append(c)
    return feats


# ----------------------------------------------------------------------
# Regression utilities
# ----------------------------------------------------------------------
def train_weights(
    rows: List[Dict[str, Optional[float]]], feats: List[str], target: str
) -> Tuple[List[float], List[float], float, List[float], List[List[float]]]:
    """
    Train linear regression with bias term.
    Returns (weights, means, mse, X_mean, inv_XTX)
    """
    X = []
    y = []
    # Compute means of features for imputation
    mean_f = {}
    for f in feats:
        vals = [r[f] for r in rows if r[f] is not None]
        mean_f[f] = sum(vals) / len(vals) if vals else 0.0

    # Build design matrix
    for r in rows:
        # Skip rows with missing target or any missing feature
        if r[target] is None:
            continue
        row_feats = []
        skip = False
        for f in feats:
            val = r[f]
            if val is None:
                skip = True
                break
            row_feats.append(val)
        if skip:
            continue
        X.append([1.0] + row_feats)  # bias
        y.append(r[target])

    if not X:
        return [0.0] * (len(feats) + 1), [0.0] * (len(feats) + 1), 0.0, [], []

    # Compute A = X^T X, b = X^T y
    Xt = transpose(X)
    A = mat_mul(Xt, X)
    b = [sum(row[i] * y[i] for i in range(len(y))) for i in range(len(A))]

    weights = solve_linear(A, b)
    if weights is None:
        weights = [0.0] * (len(feats) + 1)

    # Compute MSE
    n = len(y)
    preds = [sum(weights[j] * X[i][j] for j in range(len(weights))) for i in range(n)]
    mse = sum((preds[i] - y[i]) ** 2 for i in range(n)) / n

    # Compute mean of X without bias for interval calculation
    X_mean = [0.0] * len(feats)
    for j in range(len(feats)):
        X_mean[j] = sum(row[j + 1] for row in X) / n

    inv_XTX = invert_matrix(A)

    return weights, mean_f, mse, X_mean, inv_XTX


def predict_row(
    row: Dict[str, Optional[float]], feats: List[str], weights: List[float], means: Dict[str, float]
) -> float:
    val = 0.0
    for j, f in enumerate(feats):
        v = row[f]
        if v is None:
            v = means[f]
        val += weights[j + 1] * v
    val += weights[0]  # bias
    return val


def prediction_interval(
    row: Dict[str, Optional[float]],
    feats: List[str],
    weights: List[float],
    means: Dict[str, float],
    X_mean: List[float],
    inv_XTX: List[List[float]],
    mse: float,
) -> Tuple[float, float, float]:
    """
    Return (estimate, low, high) using 95% normal approximation.
    """
    # Estimate
    est = predict_row(row, feats, weights, means)

    # Compute (x - mean) vector
    diff = []
    for f in feats:
        v = row[f]
        if v is None:
            v = means[f]
        diff.append(v - X_mean[feats.index(f)])

    # Compute (x - mean)^T * inv_XTX * (x - mean)
    # First multiply inv_XTX * diff
    if inv_XTX is None:
        se = 0.0
    else:
        temp = [sum(inv_XTX[i][j] * diff[j] for j in range(len(diff))) for i in range(len(diff))]
        quad = sum(diff[i] * temp[i] for i in range(len(diff)))
        se = math.sqrt(mse * (1 + quad))

    # 95% interval
    low = est - 1.96 * se
    high = est + 1.96 * se
    return est, low, high


# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    global rows
    rows, header = parse_csv(csv_path)

    # Identify held‑out rows: number divisible by 5
    held_out = [i + 1 for i, _ in enumerate(rows) if (i + 1) % 5 == 0]

    # ------------------------------------------------------------------
    # CAR-03: predict Mpg+
    print("== CAR-03 ==")
    target_mpg = "Mpg+"
    feats_mpg = get_feature_columns(header, target_mpg)
    # Train on all rows not held‑out and with complete data
    train_rows_mpg = [r for i, r in enumerate(rows) if (i + 1) not in held_out]
    weights_mpg, means_mpg, mse_mpg, _, _ = train_weights(train_rows_mpg, feats_mpg, target_mpg)

    predictions_mpg: Dict[str, float] = {}
    mae_sum = 0.0
    count = 0
    for idx in held_out:
        r = rows[idx - 1]
        pred = predict_row(r, feats_mpg, weights_mpg, means_mpg)
        predictions_mpg[str(idx)] = round(pred, 6)
        if r[target_mpg] is not None:
            mae_sum += abs(pred - r[target_mpg])
            count += 1
        print(f"Row {idx}: predicted mpg {pred:.6f}")

    mae = mae_sum / count if count else float("nan")
    print(f"\nMAE: {mae:.6f}")

    ans_mpg = {
        "predictions": predictions_mpg,
        "mae": round(mae, 6),
    }
    print(f"ANSWER CAR-03 {json.dumps(ans_mpg, separators=(',', ':'))}")

    # ------------------------------------------------------------------
    # CAR-04: predict Lbs- with confidence interval
    print("\n== CAR-04 ==")
    target_wgt = "Lbs-"
    feats_wgt = get_feature_columns(header, target_wgt)
    train_rows_wgt = [r for i, r in enumerate(rows) if (i + 1) not in held_out]
    (
        weights_wgt,
        means_wgt,
        mse_wgt,
        X_mean_wgt,
        inv_XTX_wgt,
    ) = train_weights(train_rows_wgt, feats_wgt, target_wgt)

    predictions_wgt: Dict[str, Dict[str, float]] = {}
    for idx in held_out:
        r = rows[idx - 1]
        est, low, high = prediction_interval(
            r, feats_wgt, weights_wgt, means_wgt, X_mean_wgt, inv_XTX_wgt, mse_wgt
        )
        predictions_wgt[str(idx)] = {
            "estimate": round(est, 6),
            "low": round(low, 6),
            "high": round(high, 6),
        }
        print(
            f"Row {idx}: estimate {est:.6f} "
            f"[{low:.6f}, {high:.6f}]"
        )

    ans_wgt = {"predictions": predictions_wgt}
    print(f"ANSWER CAR-04 {json.dumps(ans_wgt, separators=(',', ':'))}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
