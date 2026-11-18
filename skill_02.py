#!/usr/bin/env python3
"""
Skill 02: Predict missing values and estimate weight uncertainty.
"""
import sys
import json
import math
import csv
from typing import List, Dict, Tuple

# --------------------------------------------------------------------------- #
# Reuse helpers from skill_01
# --------------------------------------------------------------------------- #

# Import the parsing and utility functions from skill_01
try:
    from skill_01 import parse_csv, is_missing, to_float
except Exception:
    # In case skill_01 is not yet available, reimplement minimal versions
    def parse_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            header = reader.fieldnames if reader.fieldnames else []
        return rows, header

    def is_missing(val: str) -> bool:
        return val == '?' or val is None

    def to_float(val: str) -> float:
        try:
            return float(val)
        except ValueError:
            return math.nan

# --------------------------------------------------------------------------- #
# Linear algebra helpers
# --------------------------------------------------------------------------- #

def matrix_transpose(A: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*A)]

def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Standard matrix multiplication."""
    result = [[0.0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for k in range(len(B)):
            aik = A[i][k]
            if aik == 0:
                continue
            for j in range(len(B[0])):
                result[i][j] += aik * B[k][j]
    return result

def matrix_vector_multiply(A: List[List[float]], v: List[float]) -> List[float]:
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def matrix_inverse(A: List[List[float]]) -> List[List[float]]:
    """Gauss-Jordan elimination for square matrix."""
    n = len(A)
    # Augment with identity
    AI = [row[:] + [float(i == j) for i in range(n)] for j, row in enumerate(A)]
    # Forward elimination
    for i in range(n):
        # Pivot
        pivot = i
        for r in range(i, n):
            if abs(AI[r][i]) > abs(AI[pivot][i]):
                pivot = r
        if abs(AI[pivot][i]) < 1e-12:
            raise ValueError("Matrix singular")
        AI[i], AI[pivot] = AI[pivot], AI[i]
        # Normalize pivot row
        div = AI[i][i]
        AI[i] = [x / div for x in AI[i]]
        # Eliminate other rows
        for r in range(n):
            if r != i:
                factor = AI[r][i]
                AI[r] = [AI[r][c] - factor * AI[i][c] for c in range(2 * n)]
    # Extract inverse
    inv = [row[n:] for row in AI]
    return inv

# --------------------------------------------------------------------------- #
# Regression helpers
# --------------------------------------------------------------------------- #

def build_design_matrix(rows: List[Dict[str, str]],
                        feature_cols: List[str],
                        target_col: str,
                        missing_ok: bool = False) -> Tuple[List[List[float]], List[float]]:
    """
    Build X matrix (with intercept) and y vector.
    If missing_ok is False, rows with any missing in features or target are skipped.
    """
    X = []
    y = []
    for row in rows:
        # Check missing
        vals = []
        missing = False
        for col in feature_cols:
            v = row[col]
            if is_missing(v):
                missing = True
                break
            vals.append(to_float(v))
        target_val = row[target_col]
        if is_missing(target_val):
            missing = True
        if missing:
            if missing_ok:
                # Include row with None placeholders
                X.append([1.0] + [float('nan')] * len(feature_cols))
                y.append(float('nan'))
            else:
                continue
        else:
            X.append([1.0] + vals)
            y.append(to_float(target_val))
    return X, y

def compute_regression(X: List[List[float]], y: List[float]) -> Tuple[List[float], float]:
    """
    Compute least squares coefficients and residual standard deviation.
    Returns (beta, sigma).
    """
    # Filter rows where any element is NaN (should not happen)
    X_clean = []
    y_clean = []
    for xi, yi in zip(X, y):
        if any(math.isnan(val) for val in xi) or math.isnan(yi):
            continue
        X_clean.append(xi)
        y_clean.append(yi)
    n = len(X_clean)
    if n == 0:
        raise ValueError("No training data")
    m = len(X_clean[0])  # number of predictors including intercept
    # Compute X^T X
    Xt = matrix_transpose(X_clean)
    XtX = matrix_multiply(Xt, X_clean)
    # Compute X^T y
    Xty = matrix_vector_multiply(Xt, y_clean)
    # Solve (X^T X) beta = X^T y
    XtX_inv = matrix_inverse(XtX)
    beta = matrix_vector_multiply(XtX_inv, Xty)
    # Compute predictions for residuals
    preds = matrix_vector_multiply(X_clean, beta)
    residuals = [yi - pi for yi, pi in zip(y_clean, preds)]
    rss = sum(r ** 2 for r in residuals)
    sigma = math.sqrt(rss / (n - m)) if (n - m) > 0 else 0.0
    return beta, sigma

def predict(beta: List[float], feature_row: List[float]) -> float:
    return sum(b * f for b, f in zip(beta, feature_row))

# --------------------------------------------------------------------------- #
# Main skill logic
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    rows, header = parse_csv(csv_path)

    # Identify columns
    goal_columns = {col: col[-1] for col in header if col.endswith('-') or col.endswith('+')}
    # For CAR-03 target Mpg+
    target_car03 = 'Mpg+'
    # For CAR-04 target Lbs-
    target_car04 = 'Lbs-'

    # Features: all columns except target and those ending with 'X'
    def feature_cols_excluding(target: str) -> List[str]:
        return [c for c in header if c != target and not c.endswith('X')]

    feature_car03 = feature_cols_excluding(target_car03)
    feature_car04 = feature_cols_excluding(target_car04)

    # Determine held-out rows (divisible by 5)
    held_out_indices = {i for i in range(1, len(rows) + 1) if i % 5 == 0}
    training_rows = [row for idx, row in enumerate(rows, start=1) if idx not in held_out_indices]

    # ---------------- CAR-03 ----------------
    X3, y3 = build_design_matrix(training_rows, feature_car03, target_car03)
    beta3, sigma3 = compute_regression(X3, y3)

    predictions_car03: Dict[str, float] = {}
    mae_sum = 0.0
    count_mae = 0

    for idx, row in enumerate(rows, start=1):
        if idx not in held_out_indices:
            continue
        # Build feature vector
        feat_vals = []
        skip = False
        for col in feature_car03:
            val = row[col]
            if is_missing(val):
                skip = True
                break
            feat_vals.append(to_float(val))
        if skip:
            continue
        feature_vector = [1.0] + feat_vals
        pred = predict(beta3, feature_vector)
        predictions_car03[str(idx)] = pred
        # MAE calculation using true value
        actual = to_float(row[target_car03]) if not is_missing(row[target_car03]) else float('nan')
        if not math.isnan(actual):
            mae_sum += abs(pred - actual)
            count_mae += 1

    mae_car03 = mae_sum / count_mae if count_mae > 0 else float('nan')

    # ---------------- CAR-04 ----------------
    X4, y4 = build_design_matrix(training_rows, feature_car04, target_car04)
    beta4, sigma4 = compute_regression(X4, y4)

    predictions_car04: Dict[str, Dict[str, float]] = {}

    for idx, row in enumerate(rows, start=1):
        if idx not in held_out_indices:
            continue
        feat_vals = []
        skip = False
        for col in feature_car04:
            val = row[col]
            if is_missing(val):
                skip = True
                break
            feat_vals.append(to_float(val))
        if skip:
            continue
        feature_vector = [1.0] + feat_vals
        estimate = predict(beta4, feature_vector)
        low = estimate - 1.96 * sigma4
        high = estimate + 1.96 * sigma4
        predictions_car04[str(idx)] = {
            "estimate": estimate,
            "low": low,
            "high": high
        }

    # ---------------- Output ----------------

    # CAR-03 output
    print("== CAR-03 ==")
    print(f"Predicted Mpg+ for {len(predictions_car03)} held-out rows")
    print(f"MAE: {mae_car03:.4f}")
    # Print first 3 predictions for sanity
    for i, (row_num, val) in enumerate(list(predictions_car03.items())[:3], start=1):
        print(f"  Row {row_num}: {val:.3f}")
    print()

    # CAR-04 output
    print("== CAR-04 ==")
    print(f"Estimated weight for {len(predictions_car04)} held-out rows")
    # Print first 3 predictions for sanity
    for i, (row_num, obj) in enumerate(list(predictions_car04.items())[:3], start=1):
        est = obj["estimate"]
        low = obj["low"]
        high = obj["high"]
        print(f"  Row {row_num}: estimate={est:.2f}, low={low:.2f}, high={high:.2f}")
    print()

    # Machine-checked answers
    print(f"ANSWER CAR-03 {json.dumps({'predictions': predictions_car03, 'mae': mae_car03}, separators=(',', ':'))}")
    print(f"ANSWER CAR-04 {json.dumps({'predictions': predictions_car04}, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
