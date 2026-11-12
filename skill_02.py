#!/usr/bin/env python3
"""
skill_02.py
-----------

This script implements two new Apache data‑analysis requests:

  APACHE-03: Estimate the likely performance of a configuration using
  a linear regression model trained on historical data.

  APACHE-04: For every pair of historically observed configurations that
  differ by exactly one setting, estimate their potential performance
  difference and provide an uncertainty estimate.

Usage:
    python skill_02.py <path_to_Apache_AllMeasurements.csv>
"""

import sys
import csv
import math
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def read_csv(csv_path):
    """
    Read the CSV file.

    Returns:
        feature_names: List of feature column names.
        X: numpy array of shape (n_samples, n_features).
        y: numpy array of shape (n_samples,).
    """
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header:
            raise ValueError("Empty CSV header")

        perf_col = header[-1]
        feature_names = header[:-1]

        X_rows = []
        y_rows = []

        for row in reader:
            if len(row) != len(header):
                continue
            try:
                X_rows.append([int(float(v)) for v in row[:-1]])
                y_rows.append(float(row[-1]))
            except ValueError:
                continue

        X = np.array(X_rows, dtype=np.float64)
        y = np.array(y_rows, dtype=np.float64)

    return feature_names, X, y


def fit_linear_model(X, y):
    """
    Fit a linear regression model and compute coefficient
    standard errors.

    Returns:
        model: fitted LinearRegression instance.
        coeff_std: numpy array of std. errors for each coefficient.
    """
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    # Residuals and variance
    residuals = y - model.predict(X)
    n, p = X.shape
    sigma2 = (residuals**2).sum() / (n - p - 1) if n - p - 1 > 0 else 0.0

    # Compute (X'X)^{-1}
    XtX_inv = np.linalg.inv(np.dot(X.T, X))
    coeff_var = sigma2 * XtX_inv
    coeff_std = np.sqrt(np.diag(coeff_var))

    return model, coeff_std


def find_pairs_with_one_diff(X):
    """
    Find all index pairs (i, j) where the two configurations differ
    in exactly one feature.

    Returns:
        pairs: list of tuples (i, j, diff_index)
    """
    n = X.shape[0]
    pairs = []

    # For efficiency, compute a hash of each configuration to avoid
    # comparing all pairs when impossible.  We still need to check
    # each pair anyway for the one‑diff condition.
    for i in range(n):
        for j in range(i + 1, n):
            diff = X[i] ^ X[j]  # bitwise XOR because values are 0/1
            diff_count = int(diff.sum())
            if diff_count == 1:
                diff_index = int(diff.argmax())
                pairs.append((i, j, diff_index))
    return pairs


# --------------------------------------------------------------------------- #
# APACHE‑03
# --------------------------------------------------------------------------- #
def apache_03_predict(model, X, y):
    """
    Generate predictions and compute simple diagnostics.
    """
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    r2 = model.score(X, y)

    return {
        'intercept': model.intercept_,
        'coefficients': model.coef_,
        'mae': mae,
        'r2': r2,
        'preds': preds,
    }


# --------------------------------------------------------------------------- #
# APACHE‑04
# --------------------------------------------------------------------------- #
def apache_04_compare_pairs(
    feature_names, X, y, model, coeff_std, pairs
):
    """
    For each pair differing by one feature, compute:
      - actual difference
      - predicted difference (coefficient * diff)
      - uncertainty (std error of coefficient)
    Return aggregated stats and top 3 pairs by absolute actual diff.
    """
    results = []

    for i, j, diff_idx in pairs:
        actual_diff = y[j] - y[i]
        predicted_diff = model.coef_[diff_idx] * (X[j, diff_idx] - X[i, diff_idx])  # ±coeff
        uncertainty = coeff_std[diff_idx]

        results.append({
            'pair': (i, j),
            'feature': feature_names[diff_idx],
            'actual_diff': actual_diff,
            'predicted_diff': predicted_diff,
            'uncertainty': uncertainty,
        })

    if not results:
        return {
            'count': 0,
            'mean_actual_diff': 0.0,
            'mean_predicted_diff': 0.0,
            'mean_uncertainty': 0.0,
            'top3': [],
        }

    # Aggregate statistics
    actuals = np.array([r['actual_diff'] for r in results])
    preds = np.array([r['predicted_diff'] for r in results])
    uncerts = np.array([r['uncertainty'] for r in results])

    mean_actual = actuals.mean()
    mean_pred = preds.mean()
    mean_unc = uncerts.mean()

    # Top 3 pairs by absolute actual difference
    sorted_results = sorted(
        results,
        key=lambda r: abs(r['actual_diff']),
        reverse=True
    )
    top3 = sorted_results[:3]

    return {
        'count': len(results),
        'mean_actual_diff': mean_actual,
        'mean_predicted_diff': mean_pred,
        'mean_uncertainty': mean_unc,
        'top3': top3,
    }


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        feature_names, X, y = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Fit model once, reused for both requests
    model, coeff_std = fit_linear_model(X, y)

    # -------------------- APACHE-03 -------------------- #
    pred_stats = apache_03_predict(model, X, y)
    print("== APACHE-03 ==")
    print(f"Intercept: {pred_stats['intercept']:.3f}")
    coeff_str = ", ".join(f"{c:.3f}" for c in pred_stats['coefficients'])
    print(f"Coefficients: [{coeff_str}]")
    print(f"Mean Absolute Error (MAE): {pred_stats['mae']:.3f}")
    print(f"R²: {pred_stats['r2']:.3f}")

    # -------------------- APACHE-04 -------------------- #
    pairs = find_pairs_with_one_diff(X)
    compare_stats = apache_04_compare_pairs(
        feature_names, X, y, model, coeff_std, pairs
    )
    print("\n== APACHE-04 ==")
    print(f"Number of configuration pairs differing by one setting: {compare_stats['count']}")
    print(f"Average actual performance difference: {compare_stats['mean_actual_diff']:.3f}")
    print(f"Average predicted performance difference: {compare_stats['mean_predicted_diff']:.3f}")
    print(f"Average coefficient uncertainty: {compare_stats['mean_uncertainty']:.3f}")

    if compare_stats['top3']:
        print("\nTop 3 pairs by absolute actual performance difference:")
        for r in compare_stats['top3']:
            i, j = r['pair']
            feat = r['feature']
            act = r['actual_diff']
            pred = r['predicted_diff']
            uncert = r['uncertainty']
            print(f"  Config {i} vs {j} (feature {feat}): "
                  f"actual diff {act:.3f}, "
                  f"pred diff {pred:.3f}, "
                  f"uncertainty {uncert:.3f}")

if __name__ == "__main__":
    main()
