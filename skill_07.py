#!/usr/bin/env python3
"""
skill_07.py
-----------

This script implements two new Apache data‑analysis requests:

  APACHE-13: Find a configuration with a low performance‑metric value
             while evaluating as few candidate configurations as
             reasonably possible.

  APACHE-14: When I have a limited benchmarking budget, recommend
             which configurations should be evaluated first so that we
             can discover promising server settings early.

Usage:
    python skill_07.py <path_to_Apache_AllMeasurements.csv>
"""

import sys
import csv
import math
import random

import numpy as np
from sklearn.linear_model import LinearRegression


# --------------------------------------------------------------------------- #
# CSV reading helper
# --------------------------------------------------------------------------- #
def read_csv(csv_path):
    """
    Read the CSV file into feature_names, X (int array), y (float array).
    """
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header:
            raise ValueError("CSV header is empty")
        feature_names = header[:-1]
        X_rows, y_rows = [], []
        for row in reader:
            if len(row) != len(header):
                continue
            try:
                X_rows.append([int(float(v)) for v in row[:-1]])
                y_rows.append(float(row[-1]))
            except ValueError:
                continue
    X = np.array(X_rows, dtype=np.int8)
    y = np.array(y_rows, dtype=np.float64)
    return feature_names, X, y


# --------------------------------------------------------------------------- #
# Model fitting utilities
# --------------------------------------------------------------------------- #
def fit_model(X_train, y_train):
    """
    Fit a linear regression model and return the fitted model,
    the variance of residuals, and the inverse of X'X.
    """
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)

    residuals = y_train - model.predict(X_train)
    n, p = X_train.shape
    sigma2 = (residuals**2).sum() / (n - p - 1) if n - p - 1 > 0 else 0.0

    # Inverse of X'X (used for prediction uncertainty)
    XtX_inv = np.linalg.inv(np.dot(X_train.T, X_train))

    return model, sigma2, XtX_inv


def predict_with_uncertainty(model, sigma2, XtX_inv, X_test):
    """
    Predict performance on X_test and estimate standard error of prediction.
    Returns yhat, se where se is the standard error of prediction.
    """
    yhat = model.predict(X_test)
    # For each row x_i, compute variance = sigma^2 * (1 + x_i^T * (X'X)^{-1} * x_i)
    # We add intercept term to x_i: [1, x_i]
    ones = np.ones((X_test.shape[0], 1))
    X_ext = np.hstack((ones, X_test))
    # Build augmented inverse matrix including intercept:
    # | 1     0^T |
    # | 0  (X'X)^{-1} |
    # The variance term for intercept is sigma^2 (since intercept variance is sigma^2 * (1/n))
    # For simplicity, we approximate by ignoring intercept variance, which is negligible for large n.
    # Compute x_i^T * XtX_inv * x_i for each i
    # Note: XtX_inv corresponds to covariance of coefficients (excluding intercept).
    # We need to add 0 for intercept column because we ignore it.
    # So se^2 = sigma^2 * (1 + x_i^T * XtX_inv * x_i)
    # Compute dot product efficiently
    se_squared = sigma2 * (1.0 + np.sum(X_test @ XtX_inv * X_test, axis=1))
    se = np.sqrt(se_squared)
    return yhat, se


# --------------------------------------------------------------------------- #
# APACHE-13: Find low‑performance configuration with few evaluations
# --------------------------------------------------------------------------- #
def apache_13(feature_names, X, y, train_frac=0.2):
    """
    Train on a small fraction of data, then choose the candidate with
    the lowest predicted performance and evaluate its true value.
    Returns a dict with details.
    """
    n_samples = X.shape[0]
    indices = list(range(n_samples))
    random.seed(42)
    random.shuffle(indices)

    n_train = max(1, int(train_frac * n_samples))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test = X[test_idx]
    y_true_test = y[test_idx]

    model, sigma2, XtX_inv = fit_model(X_train, y_train)
    yhat_test, _ = predict_with_uncertainty(model, sigma2, XtX_inv, X_test)

    # Choose candidate with lowest predicted performance
    best_test_pos = np.argmin(yhat_test)
    best_global_idx = test_idx[best_test_pos]

    result = {
        "train_size": n_train,
        "candidate_index": best_global_idx,
        "predicted_performance": yhat_test[best_test_pos],
        "true_performance": y[best_global_idx],
        "candidate_features": X[best_global_idx],
    }
    return result


# --------------------------------------------------------------------------- #
# APACHE-14: Recommend configurations to evaluate first (budgeted)
# --------------------------------------------------------------------------- #
def apache_14(feature_names, X, y, train_frac=0.2, top_n=5):
    """
    Using the same small training set, compute lower confidence bounds
    for all unseen configurations and recommend those with the
    lowest bounds (most promising).
    Returns a list of tuples (index, lower_bound, predicted, se).
    """
    n_samples = X.shape[0]
    indices = list(range(n_samples))
    random.seed(42)
    random.shuffle(indices)

    n_train = max(1, int(train_frac * n_samples))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test = X[test_idx]
    y_true_test = y[test_idx]

    model, sigma2, XtX_inv = fit_model(X_train, y_train)
    yhat_test, se_test = predict_with_uncertainty(model, sigma2, XtX_inv, X_test)

    # Compute lower confidence bound: yhat - 1.96 * se
    lower_bound = yhat_test - 1.96 * se_test

    # Get top_n candidates with lowest lower_bound
    sorted_pos = np.argsort(lower_bound)
    recommendations = []
    for pos in sorted_pos[:top_n]:
        idx = test_idx[pos]
        recommendations.append(
            (idx, lower_bound[pos], yhat_test[pos], se_test[pos])
        )
    return {
        "train_size": n_train,
        "recommendations": recommendations,
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

    if X.size == 0:
        print("No data available.")
        sys.exit(0)

    # ------------------------------------------------------------------- #
    # APACHE-13
    # ------------------------------------------------------------------- #
    result13 = apache_13(feature_names, X, y)
    print("== APACHE-13 ==")
    print(f"Training set size: {result13['train_size']}")
    print(f"Selected configuration index: {result13['candidate_index']}")
    print(f"Predicted performance: {result13['predicted_performance']:.2f}")
    print(f"True performance: {result13['true_performance']:.2f}")
    print(f"Feature vector: {result13['candidate_features']}")

    # ------------------------------------------------------------------- #
    # APACHE-14
    # ------------------------------------------------------------------- #
    result14 = apache_14(feature_names, X, y)
    print("\n== APACHE-14 ==")
    print(f"Training set size: {result14['train_size']}")
    print("Recommended configurations to evaluate first (top 5 by lowest lower bound):")
    for idx, lb, pred, se in result14["recommendations"]:
        print(f"  Index {idx:4d} | lower bound {lb:7.2f} | predicted {pred:7.2f} | SE {se:5.2f}")

if __name__ == "__main__":
    main()
