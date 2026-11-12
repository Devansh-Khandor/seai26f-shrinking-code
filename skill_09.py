#!/usr/bin/env python3
"""
APACHE skill 09

This script implements two new analyses for the Apache benchmark data:

APACHE-17: Identify configurations whose recorded performance values are unusually
           different from the performance predicted by a linear model trained on
           historical data.

APACHE-18: Flag configurations whose combinations of settings differ
           substantially from the most similar configurations in the historical
           benchmark records, even when their measured performance is not
           obviously poor.

Usage:
    python skill_09.py <path_to_Apache_AllMeasurements.csv>
"""

import sys
import csv
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

def build_features(configs, perf_col='Performance-'):
    """Return feature matrix X, target vector y and feature list."""
    feature_names = [k for k in configs[0].keys() if k != perf_col]
    X = np.array([[cfg[name] for name in feature_names] for cfg in configs], dtype=int)
    y = np.array([cfg[perf_col] for cfg in configs], dtype=int)
    return X, y, feature_names

# ---------------------------------------------------------------------------

def train_model(X, y):
    """Train a linear regression model and return it."""
    model = LinearRegression()
    model.fit(X, y)
    return model

# ---------------------------------------------------------------------------

def apache_17(configs, perf_col='Performance-'):
    """Analysis APACHE-17 – unusually large prediction residuals."""
    print("== APACHE-17 ==")

    if not configs:
        print("No configurations found.")
        return

    X, y, feature_names = build_features(configs, perf_col)
    model = train_model(X, y)

    preds = model.predict(X)
    residuals = y - preds
    residual_std = math.sqrt(mean((residuals ** 2)))

    # Threshold: two standard deviations above mean residual
    threshold = 2 * residual_std

    # Identify outliers
    outliers = [(idx+1, cfg, y[idx], preds[idx], residuals[idx])
                for idx, cfg in enumerate(configs)
                if abs(residuals[idx]) > threshold]

    if not outliers:
        print("No configurations have unusually large prediction residuals.")
        return

    print(f"Threshold for unusual residual: ±{threshold:.2f} ms")
    print(f"Found {len(outliers)} configurations with residual magnitude > threshold.\n")
    for idx, cfg, actual, pred, resid in outliers[:10]:
        print(f"Configuration #{idx} (Performance={actual} ms)")
        print(f"  Predicted Performance: {pred:.2f} ms")
        print(f"  Residual: {resid:.2f} ms\n")

# ---------------------------------------------------------------------------

def apache_18(configs, perf_col='Performance-'):
    """Analysis APACHE-18 – configurations far from most similar peers."""
    print("== APACHE-18 ==")

    if len(configs) < 2:
        print("Not enough configurations to compute similarity.")
        return

    feature_names = [k for k in configs[0].keys() if k != perf_col]

    # Precompute feature matrices for Hamming distance
    feature_matrix = np.array([[cfg[name] for name in feature_names] for cfg in configs], dtype=int)

    n = len(configs)
    min_distances = []
    nearest_neighbor = []

    for i in range(n):
        # Compute Hamming distances to all other configs
        diffs = np.sum(feature_matrix[i] != feature_matrix, axis=1)
        diffs[i] = np.inf  # ignore self
        min_idx = np.argmin(diffs)
        min_dist = diffs[min_idx]
        min_distances.append(min_dist)
        nearest_neighbor.append(min_idx)

    mean_dist = mean(min_distances)
    std_dist = stdev(min_distances) if len(min_distances) > 1 else 0.0
    threshold = mean_dist + std_dist

    # Flag configurations with min distance above threshold
    flagged = [(idx+1, cfg, min_distances[idx], nearest_neighbor[idx])
               for idx, cfg in enumerate(configs)
               if min_distances[idx] > threshold]

    if not flagged:
        print("No configurations are substantially different from their nearest peers.")
        return

    print(f"Average nearest‑neighbor distance: {mean_dist:.2f}")
    print(f"Distance threshold for flagging: {threshold:.2f}")
    print(f"Found {len(flagged)} configurations with nearest‑neighbor distance > threshold.\n")
    for idx, cfg, dist, nn_idx in flagged[:10]:
        print(f"Configuration #{idx} (Performance={cfg[perf_col]} ms)")
        print(f"  Nearest neighbor: Config #{nn_idx+1}")
        print(f"  Hamming distance: {dist}\n")

# ---------------------------------------------------------------------------

def main(csv_path):
    configs = read_csv(csv_path)
    if not configs:
        print("No data found.")
        return

    apache_17(configs)
    apache_18(configs)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <path_to_Apache_AllMeasurements.csv>")
        sys.exit(1)
    main(sys.argv[1])
