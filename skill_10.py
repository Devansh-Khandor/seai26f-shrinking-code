#!/usr/bin/env python3
"""
Apache configuration analysis: Benchmarking comparison and prediction accuracy.

This script implements:
  * APACHE-19 – Compare our configuration‑selection system (predictive model)
                against random selection under a fixed benchmarking budget, and
                report which approach discovers better‑performing configurations.
  * APACHE-20 – Compare our linear‑regression predictions against a simple
                historical‑average estimate and report whether the model
                provides a meaningful improvement on unseen configurations.

Usage:
    python skill_10.py <path_to_Apache_AllMeasurements.csv>
"""

import csv
import sys
import random
import math

import numpy as np
from sklearn.linear_model import LinearRegression
from statistics import mean, stdev


# --------------------------------------------------------------------------- #
# CSV reading and numeric conversion
# --------------------------------------------------------------------------- #
def read_csv(csv_path: str):
    """Read CSV, convert values to numeric types, and return rows and performance column name."""
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        sys.exit(1)

    if not rows:
        print("No data found in CSV.")
        sys.exit(0)

    # Find the performance column (case‑insensitive startswith "performance")
    perf_col = None
    for col in rows[0].keys():
        if col.lower().startswith("performance"):
            perf_col = col
            break
    if perf_col is None:
        print("Performance column not found.")
        sys.exit(1)

    # Convert values to numeric
    for r in rows:
        for k, v in r.items():
            if k == perf_col:
                r[k] = float(v)
            else:
                try:
                    r[k] = int(v)
                except ValueError:
                    r[k] = 0
    return rows, perf_col


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def get_setting_cols(rows, perf_col):
    return [c for c in rows[0].keys() if c != perf_col]


def best_performance(rows_subset, perf_col):
    """Return the lowest (best) performance value in a subset."""
    return min(r[perf_col] for r in rows_subset)


# --------------------------------------------------------------------------- #
# APACHE-19: Predictive vs Random Selection
# --------------------------------------------------------------------------- #
def apache_19(rows, perf_col, setting_cols):
    """Compare predictive selection with random selection under a fixed budget."""
    n_total = len(rows)
    # Benchmark budget: 5% of total, at least 1, at most 20
    budget = max(1, min(20, int(0.05 * n_total)))
    if budget > n_total:
        budget = n_total

    # ----------------- Random selection -----------------
    # Run multiple random samples to get an average best result
    n_trials = 5
    best_random_scores = []
    for _ in range(n_trials):
        sample = random.sample(rows, budget)
        best_random_scores.append(best_performance(sample, perf_col))
    avg_best_random = mean(best_random_scores)

    # ----------------- Predictive selection -----------------
    # Fit linear regression on the full dataset
    X = np.array([[r[col] for col in setting_cols] for r in rows])
    y = np.array([r[perf_col] for r in rows])
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)

    # Select top‑budget configurations by predicted performance
    sorted_idx = np.argsort(preds)
    top_pred_idx = sorted_idx[:budget]
    top_pred_rows = [rows[i] for i in top_pred_idx]
    best_pred_performance_val = best_performance(top_pred_rows, perf_col)

    # ----------------- Reporting -----------------
    print("== APACHE-19 ==")
    print(f"Benchmark budget: {budget} configurations")
    print(f"Average best performance over {n_trials} random samples: {avg_best_random:.2f}")
    print(f"Best performance found by predictive selection: {best_pred_performance_val:.2f}")

    if best_pred_performance_val < avg_best_random:
        print("Conclusion: Predictive selection discovered better configurations on average.")
    else:
        print("Conclusion: Random selection performed as well or better than predictive selection.")


# --------------------------------------------------------------------------- #
# APACHE-20: Prediction vs Historical Average
# --------------------------------------------------------------------------- #
def apache_20(rows, perf_col, setting_cols):
    """Compare linear‑regression predictions against a simple historical average."""
    # Train/test split: 70% train, 30% test
    n = len(rows)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(0.7 * n)
    train_idx = indices[:split]
    test_idx = indices[split:]

    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]

    X_train = np.array([[r[col] for col in setting_cols] for r in train_rows])
    y_train = np.array([r[perf_col] for r in train_rows])

    X_test = np.array([[r[col] for col in setting_cols] for r in test_rows])
    y_test = np.array([r[perf_col] for r in test_rows])

    # Predictive model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae_pred = float(np.mean(np.abs(y_test - y_pred)))

    # Historical average baseline
    mean_train_perf = float(np.mean(y_train))
    y_avg_pred = np.full_like(y_test, mean_train_perf)
    mae_avg = float(np.mean(np.abs(y_test - y_avg_pred)))

    # ----------------- Reporting -----------------
    print("\n== APACHE-20 ==")
    print(f"MAE of linear‑regression predictions on unseen configs: {mae_pred:.2f}")
    print(f"MAE of historical‑average baseline on unseen configs:   {mae_avg:.2f}")

    if mae_pred < mae_avg:
        improvement = (mae_avg - mae_pred) / mae_avg * 100
        print(f"Conclusion: Predictive model improves over simple average by {improvement:.1f}%.")
    else:
        print("Conclusion: Predictive model does not improve over simple average.")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, perf_col = read_csv(csv_path)
    setting_cols = get_setting_cols(rows, perf_col)

    apache_19(rows, perf_col, setting_cols)
    apache_20(rows, perf_col, setting_cols)


if __name__ == "__main__":
    main()
