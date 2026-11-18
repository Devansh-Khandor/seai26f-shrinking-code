#!/usr/bin/env python3
"""
skill_02.py: Predict missing Mpg+ (CAR-03) and Lbs- (CAR-04) with uncertainty.
"""

import csv
import json
import math
import sys
from collections import OrderedDict

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def parse_cell(s: str):
    """Convert a CSV cell to int, float, or keep string; '?' -> None."""
    if s == "?":
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def read_csv(path: str):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [list(map(parse_cell, row)) for row in reader]
    return header, rows


def is_goal(col: str):
    return col.endswith(("+", "-"))


def is_input(col: str):
    """Return True if column can be used as a predictor."""
    return not col.endswith("X") and not is_goal(col)


def build_matrix(rows, feature_idx):
    """
    Build X matrix (with intercept) and Y vector for the target column.
    Returns (X, Y) as lists of lists and list of floats.
    """
    X = []
    Y = []
    for row in rows:
        # Skip rows with missing feature or target
        if any(row[i] is None for i in feature_idx + [row[-1]]):
            continue
        # Intercept
        X.append([1.0] + [row[i] for i in feature_idx])
        Y.append(float(row[row[-1]]))  # last element is target
    return X, Y


# --------------------------------------------------------------------------- #
# Linear algebra helpers
# --------------------------------------------------------------------------- #

def mat_mul(A, B):
    """Multiply matrices A (m x n) and B (n x p)."""
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    assert n == n2, "Matrix size mismatch"
    result = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for k in range(n):
            aik = A[i][k]
            if aik:
                for j in range(p):
                    result[i][j] += aik * B[k][j]
    return result


def mat_transpose(A):
    """Transpose matrix A."""
    return [list(row) for row in zip(*A)]


def solve_linear_system(A, B):
    """
    Solve Ax = B for x using Gaussian elimination with partial pivoting.
    A is square matrix m x m, B is vector length m.
    Returns vector x.
    """
    m = len(A)
    # Build augmented matrix
    aug = [row[:] + [B[i]] for i, row in enumerate(A)]
    # Forward elimination
    for k in range(m):
        # Pivot
        max_row = max(range(k, m), key=lambda i: abs(aug[i][k]))
        if abs(aug[max_row][k]) < 1e-12:
            # Singular matrix; return zeros
            return [0.0] * m
        if max_row != k:
            aug[k], aug[max_row] = aug[max_row], aug[k]
        # Eliminate below
        for i in range(k + 1, m):
            factor = aug[i][k] / aug[k][k]
            for j in range(k, m + 1):
                aug[i][j] -= factor * aug[k][j]
    # Back substitution
    x = [0.0] * m
    for i in reversed(range(m)):
        s = aug[i][m]
        for j in range(i + 1, m):
            s -= aug[i][j] * x[j]
        x[i] = s / aug[i][i]
    return x


def linear_regression(X, Y):
    """
    Return beta coefficients and residual standard deviation sigma.
    """
    if not X:
        return [], 0.0
    # XTX
    XTX = mat_mul(mat_transpose(X), X)
    # XTY
    XTY = [sum(X[i][j] * Y[i] for i in range(len(X))) for j in range(len(XTX[0]))]
    beta = solve_linear_system(XTX, XTY)
    # Residuals
    residuals = [
        Y[i] - sum(beta[j] * X[i][j] for j in range(len(beta))) for i in range(len(X))
    ]
    n = len(X)
    p = len(beta)
    if n - p > 0:
        sigma = math.sqrt(sum(r * r for r in residuals) / (n - p))
    else:
        sigma = 0.0
    return beta, sigma


# --------------------------------------------------------------------------- #
# Prediction functions
# --------------------------------------------------------------------------- #

def predict_with_model(beta, row, feature_idx):
    """Predict y for a single row using beta coefficients."""
    val = 1.0  # intercept
    for j, fi in enumerate(feature_idx, start=1):
        val += beta[j] * row[fi]
    return val


# --------------------------------------------------------------------------- #
# Main logic for CAR-03 and CAR-04
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    header, rows = read_csv(csv_path)

    # Identify columns
    goal_cols = [c for c in header if is_goal(c)]
    input_cols = [c for c in header if is_input(c)]

    # Map column names to indices
    idx = {name: i for i, name in enumerate(header)}

    # Identify held-out rows (divisible by 5)
    held_out = [(i + 1, row) for i, row in enumerate(rows) if (i + 1) % 5 == 0]
    training_rows = [row for i, row in enumerate(rows) if (i + 1) % 5 != 0]

    # -----------------------------------------------------------------------
    # CAR-03: predict Mpg+
    # -----------------------------------------------------------------------
    mpg_col = "Mpg+"
    mpg_idx = idx[mpg_col]

    # Features for Mpg+: all input columns except goals and X
    mpg_features = [idx[c] for c in input_cols if c not in goal_cols]

    # Build training matrix
    X_train_mpg, Y_train_mpg = [], []
    for row in training_rows:
        if any(row[i] is None for i in mpg_features + [mpg_idx]):
            continue
        X_train_mpg.append([1.0] + [row[i] for i in mpg_features])
        Y_train_mpg.append(float(row[mpg_idx]))

    # Fit linear regression
    beta_mpg, sigma_mpg = linear_regression(X_train_mpg, Y_train_mpg)

    # Predict held-out
    mpg_predictions = {}
    actuals = {}
    for row_num, row in held_out:
        if any(row[i] is None for i in mpg_features):
            continue
        pred = predict_with_model(beta_mpg, row, mpg_features)
        mpg_predictions[str(row_num)] = round(pred, 4)
        actual_val = row[mpg_idx]
        if actual_val is not None:
            actuals[str(row_num)] = float(actual_val)

    # Compute MAE
    mae_mpg = sum(abs(mpg_predictions[k] - actuals[k]) for k in mpg_predictions) / len(mpg_predictions)

    # Output CAR-03
    print("== CAR-03 ==")
    for k in sorted(mpg_predictions):
        print(f"Row {k}: predicted Mpg+ = {mpg_predictions[k]}")
    print(f"MAE = {round(mae_mpg,4)}")
    print("ANSWER CAR-03", json.dumps({
        "predictions": mpg_predictions,
        "mae": round(mae_mpg, 6)
    }, separators=(",", ":")))

    # -----------------------------------------------------------------------
    # CAR-04: predict Lbs- with uncertainty
    # -----------------------------------------------------------------------
    lbs_col = "Lbs-"
    lbs_idx = idx[lbs_col]

    # Features for Lbs-: all input columns except Lbs- itself
    lbs_features = [idx[c] for c in input_cols if c != lbs_col]

    # Build training matrix
    X_train_lbs, Y_train_lbs = [], []
    for row in training_rows:
        if any(row[i] is None for i in lbs_features + [lbs_idx]):
            continue
        X_train_lbs.append([1.0] + [row[i] for i in lbs_features])
        Y_train_lbs.append(float(row[lbs_idx]))

    beta_lbs, sigma_lbs = linear_regression(X_train_lbs, Y_train_lbs)

    # Predict held-out
    lbs_predictions = {}
    for row_num, row in held_out:
        if any(row[i] is None for i in lbs_features):
            continue
        est = predict_with_model(beta_lbs, row, lbs_features)
        low = est - sigma_lbs
        high = est + sigma_lbs
        lbs_predictions[str(row_num)] = {
            "estimate": round(est, 4),
            "low": round(low, 4),
            "high": round(high, 4)
        }

    # Output CAR-04
    print("\n== CAR-04 ==")
    for k in sorted(lbs_predictions):
        d = lbs_predictions[k]
        print(f"Row {k}: Lbs- estimate={d['estimate']}, low={d['low']}, high={d['high']}")
    print("ANSWER CAR-04", json.dumps({
        "predictions": lbs_predictions
    }, separators=(",", ":")))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
