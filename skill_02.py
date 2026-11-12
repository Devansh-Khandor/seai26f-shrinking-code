#!/usr/bin/env python3
"""
Skill 02: Impute missing fuel‑efficiency and predict vehicle weight.

Usage:
    python skill_02.py <path-to-auto93.csv>
"""

import csv
import sys
import math
import statistics
from pathlib import Path

# Import functions from skill_01 that we can reuse
from skill_01 import load_data, compute_stats, compute_balanced_scores, format_value


def _feature_matrix(rows, feature_names):
    """Return matrix X (with bias term) and vector y for rows that have all features."""
    X = []
    y = []
    for r in rows:
        try:
            row_vals = [1.0]  # bias term
            for fn in feature_names:
                row_vals.append(float(r[fn]))
            X.append(row_vals)
            y.append(float(r[feature_names[0]]))  # placeholder, will be overridden
        except (KeyError, ValueError, TypeError):
            # Skip rows with missing values
            continue
    return X, y


def _prepare_regression(rows, target_field, predictor_fields):
    """Prepare X matrix and y vector for regression."""
    X = []
    y = []
    for r in rows:
        try:
            y_val = float(r[target_field])
            if math.isnan(y_val):
                continue
            row_vals = [1.0]  # bias
            for fn in predictor_fields:
                row_vals.append(float(r[fn]))
            X.append(row_vals)
            y.append(y_val)
        except (KeyError, ValueError, TypeError):
            continue
    return X, y


def _gaussian_elimination(A, b):
    """Solve linear system A x = b using Gaussian elimination with partial pivoting."""
    n = len(A)
    # Augmented matrix
    M = [row[:] + [b_val] for row, b_val in zip(A, b)]

    # Forward elimination
    for i in range(n):
        # Pivot selection
        pivot = i
        max_val = abs(M[i][i])
        for r in range(i + 1, n):
            if abs(M[r][i]) > max_val:
                max_val = abs(M[r][i])
                pivot = r
        if max_val < 1e-12:
            raise ValueError("Singular matrix during regression.")
        # Swap pivot row
        if pivot != i:
            M[i], M[pivot] = M[pivot], M[i]
        # Normalize pivot row
        piv_val = M[i][i]
        for j in range(i, n + 1):
            M[i][j] /= piv_val
        # Eliminate below
        for k in range(i + 1, n):
            factor = M[k][i]
            for j in range(i, n + 1):
                M[k][j] -= factor * M[i][j]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
    return x


def _fit_regression(X, y):
    """Fit linear regression, return coefficients and residual standard deviation."""
    beta = _gaussian_elimination(X, y)
    # Compute residuals
    n = len(y)
    p = len(beta)
    residuals = [y[i] - sum(X[i][j] * beta[j] for j in range(p)) for i in range(n)]
    residual_std = math.sqrt(sum(r**2 for r in residuals) / (n - p))
    return beta, residual_std


def _predict(row, beta, predictor_fields):
    """Predict target value for a row using beta coefficients."""
    val = beta[0]  # bias
    for j, fn in enumerate(predictor_fields, start=1):
        try:
            val += float(row[fn]) * beta[j]
        except (KeyError, ValueError, TypeError):
            val += 0.0
    return val


def _impute_mpg(rows):
    """Impute missing mpg values using linear regression."""
    # Prepare regression data
    target = "Mpg+"
    predictors = ["Lbs-", "Acc+", "Volume", "HpX", "Clndrs"]
    X, y = _prepare_regression(rows, target, predictors)

    beta, residual_std = _fit_regression(X, y)

    # Find rows with missing mpg
    missing = []
    for r in rows:
        try:
            mpg = float(r["Mpg+"])
            if math.isnan(mpg):
                missing.append(r)
        except (KeyError, ValueError, TypeError):
            missing.append(r)

    predictions = []
    for r in missing:
        pred = _predict(r, beta, predictors)
        predictions.append((r, pred, residual_std))
    return predictions


def _predict_weight(rows):
    """Predict weight for each vehicle and estimate uncertainty."""
    # Prepare regression data
    target = "Lbs-"
    predictors = ["Mpg+", "Acc+", "Volume", "HpX", "Clndrs"]
    X, y = _prepare_regression(rows, target, predictors)

    beta, residual_std = _fit_regression(X, y)

    predictions = []
    for r in rows:
        # Only predict if weight is present to compute residual
        try:
            weight = float(r["Lbs-"])
            if math.isnan(weight):
                continue
            pred = _predict(r, beta, predictors)
            predictions.append((r, pred, residual_std))
        except (KeyError, ValueError, TypeError):
            continue
    return predictions


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <path-to-auto93.csv>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    if not Path(csv_path).is_file():
        print(f"Error: file '{csv_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    rows = load_data(csv_path)
    if not rows:
        print("Error: no data loaded.", file=sys.stderr)
        sys.exit(1)

    # ---- CAR-03: Impute missing MPG ----
    print("== CAR-03 ==")
    mpg_predictions = _impute_mpg(rows)
    if not mpg_predictions:
        print("No missing MPG values to impute.")
    else:
        # Show at least three predictions
        for idx, (r, pred, std) in enumerate(mpg_predictions[:3], start=1):
            print(
                f"{idx}. Model: {r['Model']}, "
                f"Imputed MPG: {format_value(pred)}, "
                f"Uncertainty: ±{format_value(std)}"
            )

    # ---- CAR-04: Predict weight ----
    print("\n== CAR-04 ==")
    weight_predictions = _predict_weight(rows)
    if not weight_predictions:
        print("Weight predictions could not be generated.")
    else:
        # Show at least three predictions
        for idx, (r, pred, std) in enumerate(weight_predictions[:3], start=1):
            print(
                f"{idx}. Model: {r['Model']}, "
                f"Predicted Weight: {format_value(pred)} lbs, "
                f"Uncertainty: ±{format_value(std)} lbs"
            )


if __name__ == "__main__":
    main()
