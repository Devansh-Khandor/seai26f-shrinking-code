#!/usr/bin/env python3
"""
APACHE skill 02

This script extends the previous analysis with two new capabilities:

APACHE-03: Estimate the performance value for each configuration using a
linear regression model trained on the historical data.

APACHE-04: For any two historical configurations that differ by exactly one
setting, estimate the difference in their performance and quantify the
uncertainty of that estimate.

Usage:
    python skill_02.py <path_to_Apache_AllMeasurements.csv>
"""

import sys
import csv
import math
from statistics import mean
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------

def read_csv(path):
    """Read CSV and return list of dicts with numeric values."""
    configs = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cfg = {}
            for k, v in row.items():
                try:
                    cfg[k] = int(v)
                except ValueError:
                    cfg[k] = v
            configs.append(cfg)
    return configs

# ---------------------------------------------------------------------------

def build_features(configs, perf_col='Performance-'):
    """Return feature matrix X and target vector y."""
    # Column order
    all_cols = list(configs[0].keys())
    all_cols.remove(perf_col)
    X = np.array([[cfg[col] for col in all_cols] for cfg in configs])
    y = np.array([cfg[perf_col] for cfg in configs])
    return X, y, all_cols

# ---------------------------------------------------------------------------

def train_model(X, y):
    """Train linear regression and return model + prediction error std."""
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    residuals = y - preds
    sigma = math.sqrt(mean_squared_error(y, preds))
    return model, sigma

# ---------------------------------------------------------------------------

def apace_03(configs, model, all_cols):
    """Print estimated performance for first 5 configurations."""
    print("== APACHE-03 ==")
    X = np.array([[cfg[col] for col in all_cols] for cfg in configs])
    preds = model.predict(X)
    print("First 5 configurations: estimated vs actual performance")
    for i, (pred, cfg) in enumerate(zip(preds, configs), start=1):
        if i > 5:
            break
        actual = cfg['Performance-']
        print(f"{i}. Estimated: {pred:.2f}  |  Actual: {actual}")

# ---------------------------------------------------------------------------

def pairwise_diff_one_setting(configs, all_cols):
    """Yield indices and configs of pairs differing by exactly one feature."""
    n = len(configs)
    for i in range(n):
        for j in range(i+1, n):
            diff = 0
            for col in all_cols:
                if configs[i][col] != configs[j][col]:
                    diff += 1
                if diff > 1:
                    break
            if diff == 1:
                yield i, j, configs[i], configs[j]

# ---------------------------------------------------------------------------

def apache_04(configs, model, sigma, all_cols):
    """Estimate performance differences for pairs differing by one setting."""
    print("\n== APACHE-04 ==")
    pairs = list(pairwise_diff_one_setting(configs, all_cols))
    if not pairs:
        print("No pairs of configurations differ by exactly one setting.")
        return
    # Compute predictions
    X = np.array([[cfg[col] for col in all_cols] for cfg in configs])
    preds = model.predict(X)
    # Build list of (abs_diff, i, j, diff, uncertainty)
    results = []
    for i, j, ci, cj in pairs:
        diff = preds[i] - preds[j]
        uncertainty = math.sqrt(2) * sigma
        results.append((abs(diff), i, j, diff, uncertainty))
    # Sort descending by abs difference
    results.sort(reverse=True, key=lambda x: x[0])
    # Print top 5
    print("Top 5 pairs with largest predicted performance difference:")
    for rank, (abs_diff, i, j, diff, uncertainty) in enumerate(results[:5], start=1):
        print(f"{rank}. Config {i+1} vs Config {j+1}: "
              f"Predicted diff = {diff:.2f}, Uncertainty = {uncertainty:.2f}")

# ---------------------------------------------------------------------------

def main(csv_path):
    configs = read_csv(csv_path)
    if not configs:
        print("No data found.")
        return
    perf_col = 'Performance-'
    X, y, all_cols = build_features(configs, perf_col)
    model, sigma = train_model(X, y)
    apace_03(configs, model, all_cols)
    apache_04(configs, model, sigma, all_cols)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <path_to_Apache_AllMeasurements.csv>")
        sys.exit(1)
    main(sys.argv[1])
