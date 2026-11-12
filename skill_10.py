#!/usr/bin/env python3
"""
skill_10.py
-----------

This script implements two Apache data‑analysis requests:

  APACHE-19: Compare our configuration‑selection system against random
             configuration selection under the same benchmarking budget,
             and show which approach discovers better‑performing
             configurations.

  APACHE-20: Compare our performance predictions against a simple
             historical‑average estimate and show whether our system
             provides a meaningful improvement on previously unseen
             configurations.

Usage:
    python skill_10.py <path_to_Apache_AllMeasurements.csv>
"""

import sys
import csv
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

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
# Model fitting utilities (copied from skill_07)
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
    XtX_inv = np.linalg.inv(np.dot(X_train.T, X_train))
    return model, sigma2, XtX_inv

def predict_with_uncertainty(model, sigma2, XtX_inv, X_test):
    """
    Predict performance on X_test and estimate standard error of prediction.
    """
    yhat = model.predict(X_test)
    # variance of prediction = sigma^2 * (1 + x_i^T * (X'X)^{-1} * x_i)
    se_squared = sigma2 * (1.0 + np.sum(X_test @ XtX_inv * X_test, axis=1))
    se = np.sqrt(se_squared)
    return yhat, se

# --------------------------------------------------------------------------- #
# APACHE-19: Selection vs random
# --------------------------------------------------------------------------- #
def compare_selection_vs_random(X, y, train_frac=0.2, seed=42):
    """
    Split data into training and test sets, fit model, compute lower bounds,
    and compare best/mean/median performance between model‑guided
    selection and random selection under the same budget.
    Returns a dict with statistics.
    """
    n_samples = X.shape[0]
    indices = list(range(n_samples))
    random.seed(seed)
    random.shuffle(indices)

    n_train = max(1, int(train_frac * n_samples))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Fit model
    model, sigma2, XtX_inv = fit_model(X_train, y_train)
    yhat, se = predict_with_uncertainty(model, sigma2, XtX_inv, X_test)
    lower_bound = yhat - 1.96 * se

    # Budget: number of configs to evaluate
    budget = max(1, int(0.1 * len(test_idx)))  # 10% of test set or at least one

    # Model‑guided selection
    best_model_pos = np.argmin(lower_bound)
    model_selected_idx = test_idx[np.argsort(lower_bound)[:budget]]
    model_selected_perf = y_test[np.argsort(lower_bound)[:budget]]

    # Random selection
    random_selected_idx = random.sample(test_idx, budget)
    random_selected_perf = y[random_selected_idx]

    # Statistics
    def stats(arr):
        return {
            "best": arr.min(),
            "mean": arr.mean(),
            "median": np.median(arr)
        }

    return {
        "budget": budget,
        "model": stats(model_selected_perf),
        "random": stats(random_selected_perf),
        "model_indices": model_selected_idx,
        "random_indices": random_selected_idx
    }

# --------------------------------------------------------------------------- #
# APACHE-20: Prediction vs baseline average
# --------------------------------------------------------------------------- #
def compare_prediction_vs_baseline(X, y, train_frac=0.2, seed=42):
    """
    Compare mean absolute error of the linear model predictions
    against a baseline that always uses the training set mean.
    Returns a dict with metrics.
    """
    n_samples = X.shape[0]
    indices = list(range(n_samples))
    random.seed(seed)
    random.shuffle(indices)

    n_train = max(1, int(train_frac * n_samples))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model, sigma2, XtX_inv = fit_model(X_train, y_train)
    yhat, _ = predict_with_uncertainty(model, sigma2, XtX_inv, X_test)

    # Baseline estimate: mean of training performance
    baseline_mean = y_train.mean()
    baseline_pred = np.full_like(y_test, baseline_mean)

    mae_model = mean_absolute_error(y_test, yhat)
    mae_baseline = mean_absolute_error(y_test, baseline_pred)

    improvement = mae_baseline - mae_model  # positive means model better

    return {
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "improvement": improvement,
        "n_test": len(y_test)
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
    # APACHE-19
    # ------------------------------------------------------------------- #
    stats_19 = compare_selection_vs_random(X, y)
    print("== APACHE-19 ==")
    print(f"Benchmarking budget (configurations evaluated): {stats_19['budget']}")
    print(f"Model‑guided selection best performance: {stats_19['model']['best']:.2f}")
    print(f"Random selection best performance: {stats_19['random']['best']:.2f}")
    print(f"Model‑guided selection mean performance: {stats_19['model']['mean']:.2f}")
    print(f"Random selection mean performance: {stats_19['random']['mean']:.2f}")
    print(f"Model‑guided selection median performance: {stats_19['model']['median']:.2f}")
    print(f"Random selection median performance: {stats_19['random']['median']:.2f}")
    better = "model" if stats_19['model']['best'] < stats_19['random']['best'] else "random"
    print(f"Conclusion: {better} selection discovers better configurations under this budget.")

    # ------------------------------------------------------------------- #
    # APACHE-20
    # ------------------------------------------------------------------- #
    stats_20 = compare_prediction_vs_baseline(X, y)
    print("\n== APACHE-20 ==")
    print(f"Test set size: {stats_20['n_test']}")
    print(f"MAE of linear model predictions: {stats_20['mae_model']:.2f}")
    print(f"MAE of historical‑average baseline: {stats_20['mae_baseline']:.2f}")
    print(f"Absolute improvement (lower MAE): {stats_20['improvement']:.2f}")
    if stats_20['improvement'] > 0:
        print("Conclusion: The linear model provides a meaningful improvement over the baseline.")
    else:
        print("Conclusion: The linear model does not improve predictions compared to the baseline.")

if __name__ == "__main__":
    main()
