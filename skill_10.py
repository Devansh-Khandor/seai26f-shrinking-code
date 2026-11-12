#!/usr/bin/env python3
"""
APACHE skill 10

This script implements two new analyses for the Apache benchmark data:

APACHE-19: Compare our configuration-selection system against random
configuration selection under the same benchmarking budget, and show
which approach discovers better-performing configurations.

APACHE-20: Compare our performance predictions against a simple
historical‑average estimate and show whether our system provides a
meaningful improvement on previously unseen configurations.

Usage:
    python skill_10.py <path_to_Apache_AllMeasurements.csv>
"""

import sys
import csv
import random
import math
from statistics import mean, stdev
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------

def read_csv(path):
    """Read CSV and return a list of configuration dicts with numeric values."""
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

def build_feature_matrix(configs, perf_col='Performance-'):
    """Return feature matrix X and target vector y."""
    feature_names = [k for k in configs[0].keys() if k != perf_col]
    X = np.array([[cfg[name] for name in feature_names] for cfg in configs], dtype=int)
    y = np.array([cfg[perf_col] for cfg in configs], dtype=int)
    return X, y, feature_names

# ---------------------------------------------------------------------------

def apache_19(configs, perf_col='Performance-'):
    """Analysis APACHE-19 – compare system selection vs random."""
    print("== APACHE-19 ==")

    n = len(configs)
    if n < 4:
        print("Dataset too small for a meaningful comparison.")
        return

    # deterministic shuffling for reproducibility
    rng = random.Random(42)
    indices = list(range(n))
    rng.shuffle(indices)

    train_size = int(0.8 * n)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    X, y, _ = build_feature_matrix(configs, perf_col)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds_test = model.predict(X_test)

    # Budget: evaluate 10% of test set, at least one
    k = max(1, int(0.1 * len(test_idx)))
    # System selection: best predicted performance
    sorted_by_pred = sorted(zip(test_idx, preds_test), key=lambda x: x[1])  # ascending
    sys_selected = [idx for idx, _ in sorted_by_pred[:k]]
    # Random selection
    rand_selected = rng.sample(test_idx, k)

    # Compute actual mean performance of selected sets
    sys_perf = mean([configs[i][perf_col] for i in sys_selected])
    rand_perf = mean([configs[i][perf_col] for i in rand_selected])

    print(f"Benchmark budget: evaluate {k} configurations.")
    print(f"System-selected average actual performance: {sys_perf:.2f} ms")
    print(f"Random-selected average actual performance: {rand_perf:.2f} ms")

    improvement = rand_perf - sys_perf
    if improvement > 0:
        percent = improvement / rand_perf * 100
        print(f"System outperforms random by {improvement:.2f} ms "
              f"({percent:.1f}% improvement).")
    else:
        print(f"Random selection performs better by {-improvement:.2f} ms "
              f"({-percent:.1f}% difference).")
    print()

# ---------------------------------------------------------------------------

def apache_20(configs, perf_col='Performance-'):
    """Analysis APACHE-20 – model vs historical average baseline."""
    print("== APACHE-20 ==")

    n = len(configs)
    if n < 4:
        print("Dataset too small for a meaningful comparison.")
        return

    rng = random.Random(99)
    indices = list(range(n))
    rng.shuffle(indices)

    train_size = int(0.8 * n)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    X, y, _ = build_feature_matrix(configs, perf_col)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds_model = model.predict(X_test)

    # Baseline: mean of training performances
    baseline_pred = mean(y_train)
    preds_baseline = np.full_like(y_test, baseline_pred)

    # Compute RMSE
    def rmse(arr, pred):
        return math.sqrt(mean((arr - pred) ** 2))

    rmse_model = rmse(y_test, preds_model)
    rmse_baseline = rmse(y_test, preds_baseline)

    improvement = rmse_baseline - rmse_model
    improvement_pct = improvement / rmse_baseline * 100 if rmse_baseline != 0 else 0

    print(f"RMSE of linear model: {rmse_model:.2f} ms")
    print(f"RMSE of historical‑average baseline: {rmse_baseline:.2f} ms")
    if improvement > 0:
        print(f"Model improves predictions by {improvement:.2f} ms "
              f"({improvement_pct:.1f}% relative improvement).")
    else:
        print(f"Baseline performs better by {-improvement:.2f} ms "
              f"({-improvement_pct:.1f}% relative difference).")
    print()

# ---------------------------------------------------------------------------

def main(csv_path):
    configs = read_csv(csv_path)
    if not configs:
        print("No data found.")
        return

    apache_19(configs)
    apache_20(configs)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <path_to_Apache_AllMeasurements.csv>")
        sys.exit(1)
    main(sys.argv[1])
