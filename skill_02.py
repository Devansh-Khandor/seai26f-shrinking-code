#!/usr/bin/env python3
import csv
import json
import sys
import math
from typing import List, Tuple, Dict, Any, Optional

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[str], List[List[Any]]]:
    """Read CSV, convert numeric cells, keep '?' as None."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for row in reader:
            converted = []
            for v in row:
                if v == "?":
                    converted.append(None)
                else:
                    try:
                        if "." in v or "e" in v.lower():
                            converted.append(float(v))
                        else:
                            converted.append(int(v))
                    except ValueError:
                        converted.append(v)
            rows.append(converted)
    return header, rows

def is_goal(col: str) -> bool:
    return col.endswith('+') or col.endswith('-')

def get_feature_indices(header: List[str], exclude_target: str) -> List[int]:
    goal_cols = [h for h in header if is_goal(h)]
    return [
        i
        for i, h in enumerate(header)
        if not h.endswith('X')
        and h not in goal_cols
        and h != exclude_target
    ]

# ------------------------------------------------------------------
# Matrix helpers
# ------------------------------------------------------------------
def matrix_inverse(A: List[List[float]]) -> List[List[float]]:
    """Inverse of a square matrix using Gauss–Jordan elimination."""
    n = len(A)
    # Augment with identity
    M = [row[:] + [float(i == j) for i in range(n)] for j, row in enumerate(A)]
    # Forward elimination
    for i in range(n):
        # Find pivot
        pivot = i
        while pivot < n and abs(M[pivot][i]) < 1e-12:
            pivot += 1
        if pivot == n:
            raise ValueError("Singular matrix")
        if pivot != i:
            M[i], M[pivot] = M[pivot], M[i]
        # Normalize pivot row
        div = M[i][i]
        M[i] = [v / div for v in M[i]]
        # Eliminate other rows
        for r in range(n):
            if r != i:
                factor = M[r][i]
                M[r] = [rv - factor * iv for rv, iv in zip(M[r], M[i])]
    # Extract inverse
    return [row[n:] for row in M]

def mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Matrix multiplication."""
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    assert n == n2, "Incompatible dimensions"
    result = [[0.0 for _ in range(p)] for _ in range(m)]
    for i in range(m):
        for k in range(n):
            aik = A[i][k]
            if aik == 0:
                continue
            for j in range(p):
                result[i][j] += aik * B[k][j]
    return result

def mat_vec_mul(A: List[List[float]], v: List[float]) -> List[float]:
    """Matrix times vector."""
    m = len(A)
    n = len(v)
    assert len(A[0]) == n
    res = [0.0 for _ in range(m)]
    for i in range(m):
        s = 0.0
        for j in range(n):
            s += A[i][j] * v[j]
        res[i] = s
    return res

# ------------------------------------------------------------------
# Regression helpers
# ------------------------------------------------------------------
def train_linear_regression(
    rows: List[List[Any]], feature_indices: List[int], target_index: int
) -> Tuple[List[float], List[List[float]], List[float], float]:
    """Return coefficients, inverse XTX, XTy, residual variance sigma^2."""
    X = []
    y = []
    for row in rows:
        if row[target_index] is None:
            continue
        if any(row[i] is None for i in feature_indices):
            continue
        x = [1.0] + [float(row[i]) for i in feature_indices]
        X.append(x)
        y.append(float(row[target_index]))
    if not X:
        raise ValueError("No training data after filtering")
    n_features = len(X[0])
    # Compute X^T X
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
    # Compute coefficients beta = inv_XTX * XTy
    beta = mat_vec_mul(inv_XTX, XTy)
    # Residual variance
    n_train = len(X)
    p = n_features
    residuals = [
        yi - sum(beta[j] * xi[j] for j in range(p))
        for xi, yi in zip(X, y)
    ]
    sigma2 = sum(r * r for r in residuals) / max(1, n_train - p)
    return beta, inv_XTX, XTy, sigma2

def predict_linear(
    beta: List[float], inv_XTX: List[List[float]], x0: List[float], sigma2: float
) -> Tuple[float, float]:
    """Return prediction and 95% CI low/high."""
    pred = sum(beta[j] * x0[j] for j in range(len(beta)))
    # Compute standard error
    p = len(beta)
    # x0^T * inv_XTX * x0
    x0Tx0 = 0.0
    for i in range(p):
        for j in range(p):
            x0Tx0 += x0[i] * inv_XTX[i][j] * x0[j]
    se = math.sqrt(sigma2 * (1 + x0Tx0))
    ci_low = pred - 1.96 * se
    ci_high = pred + 1.96 * se
    return pred, ci_low, ci_high

# ------------------------------------------------------------------
# Main logic for CAR-03 and CAR-04
# ------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)
    path = sys.argv[1]
    header, rows = read_csv(path)

    # Determine row numbers (1-based)
    n_rows = len(rows)
    held_out_indices = {i + 1 for i in range(n_rows) if (i + 1) % 5 == 0}
    history = [rows[i] for i in range(n_rows) if (i + 1) not in held_out_indices]

    # Indices
    mpg_idx = header.index("Mpg+")
    lbs_idx = header.index("Lbs-")

    # ----------------- CAR-03 -----------------
    feature_indices_mpg = get_feature_indices(header, exclude_target="Mpg+")
    beta_mpg, inv_XTX_mpg, XTy_mpg, sigma2_mpg = train_linear_regression(
        history, feature_indices_mpg, mpg_idx
    )

    predictions_mpg: Dict[str, float] = {}
    mae = 0.0
    n_mae = 0

    for idx in held_out_indices:
        row = rows[idx - 1]
        # Build feature vector
        x = [1.0] + [float(row[i]) for i in feature_indices_mpg]
        pred, _, _ = predict_linear(beta_mpg, inv_XTX_mpg, x, sigma2_mpg)
        predictions_mpg[str(idx)] = pred
        # Compute error
        actual = float(row[mpg_idx]) if row[mpg_idx] is not None else None
        if actual is not None:
            mae += abs(pred - actual)
            n_mae += 1

    mae = mae / n_mae if n_mae else 0.0

    print("== CAR-03 ==")
    for rno, val in sorted(predictions_mpg.items(), key=lambda kv: int(kv[0])):
        print(f"Row {rno}: {val:.4f}")
    print(f"Mean absolute error: {mae:.4f}")
    json_mpg = {"predictions": predictions_mpg, "mae": mae}
    print(f"ANSWER CAR-03 {json.dumps(json_mpg, separators=(',', ':'))}")

    # ----------------- CAR-04 -----------------
    feature_indices_lbs = get_feature_indices(header, exclude_target="Lbs-")
    beta_lbs, inv_XTX_lbs, XTy_lbs, sigma2_lbs = train_linear_regression(
        history, feature_indices_lbs, lbs_idx
    )

    predictions_lbs: Dict[str, Dict[str, float]] = {}
    for idx in held_out_indices:
        row = rows[idx - 1]
        x = [1.0] + [float(row[i]) for i in feature_indices_lbs]
        pred, low, high = predict_linear(beta_lbs, inv_XTX_lbs, x, sigma2_lbs)
        predictions_lbs[str(idx)] = {"estimate": pred, "low": low, "high": high}

    print("\n== CAR-04 ==")
    for rno, vals in sorted(predictions_lbs.items(), key=lambda kv: int(kv[0])):
        print(f"Row {rno}: estimate={vals['estimate']:.4f} "
              f"low={vals['low']:.4f} high={vals['high']:.4f}")
    json_lbs = {"predictions": predictions_lbs}
    print(f"ANSWER CAR-04 {json.dumps(json_lbs, separators=(',', ':'))}")

if __name__ == "__main__":
    main()
