#!/usr/bin/env python3
"""
APACHE skill 07

This script implements two new analyses for the Apache benchmark data:

APACHE-13: Help identify a configuration with a low performance-metric value
by examining as few candidate configurations as reasonably possible.

APACHE-14: When benchmarking resources are limited, recommend which
configurations should be evaluated first so that promising server settings
are discovered early.

Usage:
    python skill_07.py <path_to_Apache_AllMeasurements.csv>
"""

import sys
import csv
import math
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------

def read_csv(path):
    """Read CSV and return a list of configuration dictionaries."""
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

def format_config(cfg):
    """Return a string representation of a configuration."""
    return ', '.join(f'{k}={v}' for k, v in cfg.items())

# ---------------------------------------------------------------------------

def apache_13(configs, perf_col='Performance-'):
    """Analysis APACHE-13 – find the configuration with the lowest performance."""
    print("== APACHE-13 ==")
    if not configs:
        print("No configurations found.")
        return

    # Find the configuration with the minimal performance value
    best_cfg = min(configs, key=lambda c: c[perf_col])
    best_idx = configs.index(best_cfg) + 1  # 1-based index

    print(f"Configuration with the lowest performance found:")
    print(f"  Configuration #{best_idx}: {format_config(best_cfg)}")
    print(f"  Performance: {best_cfg[perf_col]} ms")
    print()

# ---------------------------------------------------------------------------

def build_features(configs, perf_col='Performance-'):
    """Return X matrix, y vector and column order."""
    all_cols = list(configs[0].keys())
    all_cols.remove(perf_col)
    X = np.array([[cfg[col] for col in all_cols] for cfg in configs])
    y = np.array([cfg[perf_col] for cfg in configs])
    return X, y, all_cols

# ---------------------------------------------------------------------------

def apache_14(configs, perf_col='Performance-', top_n=5):
    """Analysis APACHE-14 – recommend configurations to evaluate first."""
    print("== APACHE-14 ==")
    if not configs:
        print("No configurations found.")
        return

    X, y, feature_cols = build_features(configs, perf_col)

    # Train a simple linear regression model to predict performance
    model = LinearRegression()
    model.fit(X, y)

    # Predict performance for all configurations
    preds = model.predict(X)

    # Sort configurations by predicted performance ascending (best first)
    sorted_indices = np.argsort(preds)

    print(f"Recommended evaluation order (top {top_n} predicted best configurations):")
    for rank, idx in enumerate(sorted_indices[:top_n], start=1):
        cfg = configs[idx]
        pred_perf = preds[idx]
        actual = cfg[perf_col]
        print(f"  {rank}. Configuration #{idx + 1}: {format_config(cfg)}")
        print(f"     Predicted performance: {pred_perf:.1f} ms")
        print(f"     Actual performance:   {actual} ms")
    print()

# ---------------------------------------------------------------------------

def main(csv_path):
    configs = read_csv(csv_path)
    if not configs:
        print("No data found.")
        return

    apache_13(configs)
    apache_14(configs)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <path_to_Apache_AllMeasurements.csv>")
        sys.exit(1)
    main(sys.argv[1])
