#!/usr/bin/env python3
"""
skill_02.py <csv_path>
Implements CAR-03 (fuel efficiency prediction) and CAR-04 (weight estimation with uncertainty).
"""

import csv
import json
import sys
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

# --------------------------------------------------------------------------- #
# Utility: matrix operations (no external libs)
# --------------------------------------------------------------------------- #

def transpose(a: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*a)]

def mat_mult(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Multiply two matrices a (m x n) and b (n x p)."""
    result = [[0.0 for _ in range(len(b[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for k in range(len(b)):
            aik = a[i][k]
            if aik == 0.0:
                continue
            for j in range(len(b[0])):
                result[i][j] += aik * b[k][j]
    return result

def mat_inverse(a: List[List[float]]) -> List[List[float]]:
    """Inverse of a square matrix using Gauss-Jordan elimination."""
    n = len(a)
    # create augmented matrix [a | I]
    aug = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(a)]
    # forward elimination
    for i in range(n):
        # pivot
        pivot = aug[i][i]
        if abs(pivot) < 1e-12:
            # find non-zero pivot
            for r in range(i+1, n):
                if abs(aug[r][i]) > 1e-12:
                    aug[i], aug[r] = aug[r], aug[i]
                    pivot = aug[i][i]
                    break
        if abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular.")
        # normalize row
        for j in range(2*n):
            aug[i][j] /= pivot
        # eliminate other rows
        for r in range(n):
            if r == i:
                continue
            factor = aug[r][i]
            if factor != 0.0:
                for j in range(2*n):
                    aug[r][j] -= factor * aug[i][j]
    # extract inverse
    inv = [row[n:] for row in aug]
    return inv

# --------------------------------------------------------------------------- #
# Regression helpers
# --------------------------------------------------------------------------- #

def prepare_matrix(rows: List[List[float]]) -> Tuple[List[List[float]], List[float]]:
    """Return X (with intercept) and y from rows of [x1, x2, ..., y]."""
    X = [[1.0] + list(r[:-1]) for r in rows]
    y = [r[-1] for r in rows]
    return X, y

def normal_equation(X: List[List[float]], y: List[float]) -> List[float]:
    """Solve (X^T X) beta = X^T y."""
    Xt = transpose(X)
    XtX = mat_mult(Xt, X)
    Xty = mat_mult(Xt, [[val] for val in y])
    XtX_inv = mat_inverse(XtX)
    beta_mat = mat_mult(XtX_inv, Xty)
    beta = [row[0] for row in beta_mat]
    return beta, XtX_inv

def predict_row(beta: List[float], x: List[float]) -> float:
    """Compute beta^T * [1, x1, x2, ...]."""
    return sum(b * val for b, val in zip(beta, [1.0] + x))

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_data(csv_path: str) -> Tuple[List[int], List[Dict[str, Optional[float]]], List[str]]:
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for i, row in enumerate(reader, start=1):
            parsed: Dict[str, Optional[float]] = {}
            for k, v in row.items():
                if v == '?' or v == '':
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = None
            rows.append((i, parsed))
    return rows, headers

# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    rows, headers = load_data(csv_path)

    # Identify held‑out rows
    held_out = {i for i, _ in rows if i % 5 == 0}

    # Feature sets
    mpg_features = ["Clndrs","Volume","HpX","Model","origin","Lbs-","Acc+"]
    weight_features = ["Clndrs","Volume","HpX","Model","origin","Acc+"]  # exclude Lbs- (target)
    # Note: we could add Mpg+ but it may be missing for held‑out rows

    # ----------------------------------------------------------------------- #
    # CAR-03: Mpg+ prediction
    # ----------------------------------------------------------------------- #

    # Build training data (exclude held‑out rows)
    mpg_train_rows = []
    for idx, data in rows:
        if idx in held_out:
            continue
        # target must exist
        if data.get("Mpg+") is None:
            continue
        # all features must exist
        if any(data.get(f) is None for f in mpg_features):
            continue
        row = [data[f] for f in mpg_features] + [data["Mpg+"]]
        mpg_train_rows.append(row)

    # If insufficient data, skip prediction
    mpg_preds = {}
    mae = 0.0
    if mpg_train_rows:
        X_mpg, y_mpg = prepare_matrix(mpg_train_rows)
        beta_mpg, _ = normal_equation(X_mpg, y_mpg)
        # Precompute means for imputation
        feature_means = {}
        for f in mpg_features:
            vals = [data[f] for _, data in rows if data.get(f) is not None]
            feature_means[f] = sum(vals) / len(vals) if vals else 0.0

        # Predict on held‑out rows
        errors = []
        for idx, data in rows:
            if idx not in held_out:
                continue
            if data.get("Mpg+") is None:
                continue  # cannot compute MAE
            # build feature vector with imputation
            x = [data.get(f) if data.get(f) is not None else feature_means[f] for f in mpg_features]
            pred = predict_row(beta_mpg, x)
            mpg_preds[str(idx)] = pred
            errors.append(abs(pred - data["Mpg+"]))

        mae = sum(errors) / len(errors) if errors else 0.0

    # Print CAR-03 section
    print("== CAR-03 ==")
    print(f"Predicted Mpg+ for held‑out rows: {len(mpg_preds)}")
    print(f"MAE: {mae:.4f}")
    answer_car03 = {"predictions": mpg_preds, "mae": mae}
    print(f"ANSWER CAR-03 {json.dumps(answer_car03)}")

    # ----------------------------------------------------------------------- #
    # CAR-04: Lbs- prediction with uncertainty
    # ----------------------------------------------------------------------- #

    weight_train_rows = []
    for idx, data in rows:
        if idx in held_out:
            continue
        if data.get("Lbs-") is None:
            continue
        if any(data.get(f) is None for f in weight_features):
            continue
        row = [data[f] for f in weight_features] + [data["Lbs-"]]
        weight_train_rows.append(row)

    weight_preds = {}
    if weight_train_rows:
        X_w, y_w = prepare_matrix(weight_train_rows)
        beta_w, XtX_inv_w = normal_equation(X_w, y_w)
        # Residual standard error
        preds_y = [predict_row(beta_w, r[:-1]) for r in weight_train_rows]
        residuals = [y - py for y, py in zip(y_w, preds_y)]
        n = len(weight_train_rows)
        p = len(beta_w)
        sigma_hat = (sum(r**2 for r in residuals) / (n - p))**0.5

        # Feature means for imputation
        feat_means = {}
        for f in weight_features:
            vals = [data[f] for _, data in rows if data.get(f) is not None]
            feat_means[f] = sum(vals) / len(vals) if vals else 0.0

        # Precompute (X^T X)^-1 for se prediction
        XtX_inv_w_mat = XtX_inv_w  # already matrix

        for idx, data in rows:
            if idx not in held_out:
                continue
            # Build feature vector with imputation
            x = [data.get(f) if data.get(f) is not None else feat_means[f] for f in weight_features]
            pred_est = predict_row(beta_w, x)

            # Compute prediction standard error
            # se = sigma * sqrt(1/n + (x-mean_x)^T * (X^T X)^-1 * (x-mean_x))
            # compute mean vector of training features (including intercept)
            mean_x = [0.0]  # intercept
            for f in weight_features:
                mean_x.append(feat_means[f])

            diff = [x[i] - mean_x[i] for i in range(len(x)+1)]
            # matrix-vector product: (X^T X)^-1 * diff
            temp = [sum(XtX_inv_w_mat[i][j] * diff[j] for j in range(len(diff))) for i in range(len(diff))]
            val = sum(diff[i] * temp[i] for i in range(len(diff)))
            se = sigma_hat * (1/n + val)**0.5
            # 95% interval
            low = pred_est - 1.96 * se
            high = pred_est + 1.96 * se
            weight_preds[str(idx)] = {"estimate": pred_est, "low": low, "high": high}

    # Print CAR-04 section
    print("== CAR-04 ==")
    print(f"Predicted Lbs- for held‑out rows: {len(weight_preds)}")
    # Show a sample
    sample = next(iter(weight_preds.items())) if weight_preds else ("", {})
    print(f"Sample prediction for row {sample[0]}: {sample[1]}")
    answer_car04 = {"predictions": weight_preds}
    print(f"ANSWER CAR-04 {json.dumps(answer_car04)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_02.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])
