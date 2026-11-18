#!/usr/bin/env python3
"""
Entry point for CAR-03 and CAR-04 skills.
"""

import json
import math
from typing import List, Tuple, Dict, Any

# The ezr library is available in the same directory.
from ezr import csv  # we only need csv to parse the file

# ----------------------------------------------------------------------
#  Utility: small matrix operations for linear regression
# ----------------------------------------------------------------------
def mat_transpose(A: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*A)]


def mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    n, m = len(A), len(A[0])
    p = len(B[0])
    res = [[0.0] * p for _ in range(n)]
    for i in range(n):
        for k in range(m):
            aik = A[i][k]
            if aik == 0:
                continue
            for j in range(p):
                res[i][j] += aik * B[k][j]
    return res


def gauss_jordan(A: List[List[float]]) -> List[List[float]]:
    """Return the inverse of a square matrix A using Gauss‑Jordan."""
    n = len(A)
    # Create augmented matrix
    M = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(A)]

    # Forward elimination
    for i in range(n):
        # Pivot
        piv = i
        for r in range(i, n):
            if abs(M[r][i]) > abs(M[piv][i]):
                piv = r
        if abs(M[piv][i]) < 1e-12:
            raise Exception("Singular matrix encountered during inversion.")
        if piv != i:
            M[i], M[piv] = M[piv], M[i]

        # Normalize pivot row
        div = M[i][i]
        for j in range(2 * n):
            M[i][j] /= div

        # Eliminate other rows
        for r in range(n):
            if r == i:
                continue
            factor = M[r][i]
            if factor != 0:
                for j in range(2 * n):
                    M[r][j] -= factor * M[i][j]

    # Extract inverse part
    inv = [row[n:] for row in M]
    return inv


def linear_regression(X: List[List[float]], y: List[float]) -> List[float]:
    """Return coefficients (intercept first) for ordinary least squares."""
    X_T = mat_transpose(X)
    XtX = mat_mul(X_T, X)
    XtX_inv = gauss_jordan(XtX)
    X_T_y = mat_mul(X_T, [[v] for v in y])  # column vector
    beta_col = mat_mul(XtX_inv, X_T_y)
    return [beta_col[i][0] for i in range(len(beta_col))]


def predict_row(beta: List[float], features: List[float]) -> float:
    """Compute dot product of beta and feature vector (including intercept)."""
    return sum(b * f for b, f in zip(beta, features))


# ----------------------------------------------------------------------
#  Core skill logic
# ----------------------------------------------------------------------
def load_csv(path: str) -> Tuple[List[str], List[List[Any]]]:
    rows = [row for row in csv(path)]
    if not rows:
        raise ValueError("CSV file is empty or not found.")
    header = rows[0]
    data = rows[1:]
    return header, data


def build_features(
    row: List[Any], feature_idxs: List[int]
) -> List[float] | None:
    try:
        return [float(row[i]) for i in feature_idxs]
    except (ValueError, TypeError):
        return None


def train_and_predict(
    data: List[List[Any]],
    target_idx: int,
    feature_idxs: List[int],
    held_out_mask: List[bool],
) -> Tuple[Dict[str, float], float]:
    # Training data
    X_train: List[List[float]] = []
    y_train: List[float] = []

    for row_num, (row, is_held_out) in enumerate(zip(data, held_out_mask), start=1):
        if is_held_out:
            continue
        target = row[target_idx]
        if target == "?":
            continue
        feats = build_features(row, feature_idxs)
        if feats is None:
            continue
        X_train.append([1.0] + feats)  # intercept
        y_train.append(float(target))

    # Fit model
    beta = linear_regression(X_train, y_train)

    # Residual std from training predictions
    preds_train = [
        predict_row(beta, [1.0] + build_features(row, feature_idxs))
        for row in data
        if not held_out_mask[data.index(row)]
        and row[target_idx] != "?"
        and build_features(row, feature_idxs) is not None
    ]
    resids = [
        y - p for y, p in zip(y_train, preds_train)
    ]
    resid_std = math.sqrt(sum(r * r for r in resids) / len(resids)) if resids else 0.0

    # Predictions for held‑out rows
    predictions: Dict[str, float] = {}
    mae_total = 0.0
    count_mae = 0

    for row_num, (row, is_held_out) in enumerate(zip(data, held_out_mask), start=1):
        if not is_held_out:
            continue
        target = row[target_idx]
        if target == "?":
            continue
        feats = build_features(row, feature_idxs)
        if feats is None:
            continue
        pred = predict_row(beta, [1.0] + feats)
        predictions[str(row_num)] = round(pred, 6)
        mae_total += abs(float(target) - pred)
        count_mae += 1

    mae = round(mae_total / count_mae, 6) if count_mae else 0.0
    return predictions, mae


def train_and_predict_uncert(
    data: List[List[Any]],
    target_idx: int,
    feature_idxs: List[int],
    held_out_mask: List[bool],
) -> Dict[str, Any]:
    # Training data
    X_train: List[List[float]] = []
    y_train: List[float] = []

    for row_num, (row, is_held_out) in enumerate(zip(data, held_out_mask), start=1):
        if is_held_out:
            continue
        target = row[target_idx]
        if target == "?":
            continue
        feats = build_features(row, feature_idxs)
        if feats is None:
            continue
        X_train.append([1.0] + feats)
        y_train.append(float(target))

    beta = linear_regression(X_train, y_train)

    preds_train = [
        predict_row(beta, [1.0] + build_features(row, feature_idxs))
        for row in data
        if not held_out_mask[data.index(row)]
        and row[target_idx] != "?"
        and build_features(row, feature_idxs) is not None
    ]
    resids = [y - p for y, p in zip(y_train, preds_train)]
    resid_std = math.sqrt(sum(r * r for r in resids) / len(resids)) if resids else 0.0

    predictions: Dict[str, Any] = {}
    for row_num, (row, is_held_out) in enumerate(zip(data, held_out_mask), start=1):
        if not is_held_out:
            continue
        target = row[target_idx]
        if target == "?":
            continue
        feats = build_features(row, feature_idxs)
        if feats is None:
            continue
        est = predict_row(beta, [1.0] + feats)
        low = est - 1.96 * resid_std
        high = est + 1.96 * resid_std
        predictions[str(row_num)] = {
            "estimate": round(est, 6),
            "low": round(low, 6),
            "high": round(high, 6),
        }

    return predictions


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    header, data = load_csv(path)

    # Find column indices
    mpg_idx = header.index("Mpg+")
    lbs_idx = header.index("Lbs-")
    # Feature indices for each task
    mpg_features = [i for i in range(len(header)) if i != mpg_idx]
    lbs_features = [i for i in range(len(header)) if i != lbs_idx]

    # Determine held‑out mask: rows divisible by 5
    held_out_mask = [(row_num % 5 == 0) for row_num in range(1, len(data) + 1)]

    # CAR‑03: Predict Mpg+
    mpg_preds, mpg_mae = train_and_predict(
        data, mpg_idx, mpg_features, held_out_mask
    )
    print("== CAR-03 ==")
    for r, p in sorted(mpg_preds.items(), key=lambda x: int(x[0])):
        print(f"{r}: {p}")
    print(f"MAE: {mpg_mae}")
    print(f"ANSWER CAR-03 {json.dumps({'predictions': mpg_preds, 'mae': mpg_mae}, separators=(',', ':'))}")

    # CAR‑04: Predict Lbs-
    lbs_preds = train_and_predict_uncert(
        data, lbs_idx, lbs_features, held_out_mask
    )
    print("\n== CAR-04 ==")
    for r, obj in sorted(lbs_preds.items(), key=lambda x: int(x[0])):
        est = obj["estimate"]
        print(f"{r}: estimate={est}, low={obj['low']}, high={obj['high']}")
    print(f"ANSWER CAR-04 {json.dumps({'predictions': lbs_preds}, separators=(',', ':'))}")


if __name__ == "__main__":
    import sys

    main()
