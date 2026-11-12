#!/usr/bin/env python3
"""
APACHE skill 08

This script implements two new analyses for the Apache benchmark data:

APACHE-15: For a poorly performing configuration, recommend a single
adjustable setting change that could lower the predicted performance value,
using historically observed valid configurations as alternatives.

APACHE-16: Find the smallest number of configuration-setting changes
that could bring a poorly performing configuration closer to a
specified performance target, while only considering valid (historical)
configurations.

Usage:
    python skill_08.py <path_to_Apache_AllMeasurements.csv>
"""

import sys
import csv
import math
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------

def read_csv(path):
    """Read the CSV file and return a list of configuration dicts with numeric values."""
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

def compute_stats(configs, perf_col='Performance-'):
    """Return mean and std of the performance column."""
    performances = [c[perf_col] for c in configs]
    mean_perf = sum(performances) / len(performances)
    if len(performances) > 1:
        std_perf = math.sqrt(sum((p - mean_perf) ** 2 for p in performances) / (len(performances) - 1))
    else:
        std_perf = 0.0
    return mean_perf, std_perf

# ---------------------------------------------------------------------------

def build_feature_matrix(configs, perf_col='Performance-'):
    """Return feature matrix X, target vector y and feature name list."""
    feature_names = [k for k in configs[0].keys() if k != perf_col]
    X = np.array([[cfg[fn] for fn in feature_names] for cfg in configs], dtype=int)
    y = np.array([cfg[perf_col] for cfg in configs], dtype=int)
    return X, y, feature_names

# ---------------------------------------------------------------------------

def build_config_map(configs, feature_names):
    """Return a dict mapping feature tuple to (index, cfg) for quick lookup."""
    cfg_map = {}
    for idx, cfg in enumerate(configs):
        key = tuple(cfg[fn] for fn in feature_names)
        cfg_map[key] = (idx, cfg)
    return cfg_map

# ---------------------------------------------------------------------------

def apache_15(configs, perf_col='Performance-'):
    """APACHE-15: Recommend a single setting change for each poor configuration."""
    print("== APACHE-15 ==")

    if not configs:
        print("No configurations available.")
        return

    mean_perf, std_perf = compute_stats(configs, perf_col)
    threshold = mean_perf + std_perf  # definition of a "poor" configuration

    # Identify poorly performing configurations
    poor_configs = [(idx, cfg) for idx, cfg in enumerate(configs, start=1)
                    if cfg[perf_col] >= threshold]

    if not poor_configs:
        print("No poorly performing configurations found.")
        return

    # Prepare predictive model
    X, y, feature_names = build_feature_matrix(configs, perf_col)
    model = LinearRegression()
    model.fit(X, y)

    # Build config map for quick existence checks
    cfg_map = build_config_map(configs, feature_names)

    for idx, cfg in poor_configs:
        current_perf = cfg[perf_col]
        best_improve = 0.0
        best_change = None
        best_target_cfg = None

        # Evaluate flipping each feature
        for fn in feature_names:
            new_val = 1 - cfg[fn]  # toggle binary setting
            key = tuple(cfg[fn] if f != fn else new_val for f in feature_names)
            if key not in cfg_map:
                continue  # no such configuration historically observed

            # Predict performance for the flipped configuration
            new_X = np.array([key], dtype=int)
            pred_perf = model.predict(new_X)[0]

            improvement = current_perf - pred_perf
            if improvement > best_improve:
                best_improve = improvement
                best_change = (fn, new_val)
                best_target_cfg = cfg_map[key][1]

        if best_change:
            fn, new_val = best_change
            target_perf = best_target_cfg[perf_col]
            print(f"Config #{idx} (Perf: {current_perf} ms)")
            print(f"  Recommend toggling '{fn}' -> {new_val}")
            print(f"  Predicted performance after change: {target_perf:.1f} ms")
            print(f"  Estimated improvement: {best_improve:.1f} ms\n")
        else:
            print(f"Config #{idx} (Perf: {current_perf} ms) has no single‑setting improvement based on historical data.\n")

# ---------------------------------------------------------------------------

def apache_16(configs, perf_col='Performance-'):
    """APACHE-16: Minimal number of changes to reach a target performance."""
    print("== APACHE-16 ==")

    if not configs:
        print("No configurations available.")
        return

    mean_perf, std_perf = compute_stats(configs, perf_col)
    target_perf = mean_perf - std_perf   # target is "good" performance (mean - std)

    print(f"Target performance threshold: {target_perf:.2f} ms (mean - std)")

    # Identify poorly performing configurations
    poor_configs = [(idx, cfg) for idx, cfg in enumerate(configs, start=1)
                    if cfg[perf_col] >= mean_perf + std_perf]

    if not poor_configs:
        print("No poorly performing configurations found.")
        return

    # Build mapping for quick lookup
    feature_names = [k for k in configs[0].keys() if k != perf_col]
    target_cfgs = [(idx, cfg) for idx, cfg in enumerate(configs, start=1)
                   if cfg[perf_col] <= target_perf]

    if not target_cfgs:
        print("No configurations meet the target performance.")
        return

    # Pre‑compute feature tuples for all configs
    cfg_tuples = [tuple(cfg[fn] for fn in feature_names) for cfg in configs]
    target_tuples = [tuple(cfg[fn] for fn in feature_names) for _, cfg in target_cfgs]

    for p_idx, p_cfg in poor_configs:
        min_changes = None
        best_target = None
        best_target_idx = None

        for t_idx, t_cfg in target_cfgs:
            # compute Hamming distance
            dist = sum(1 for a, b in zip(p_cfg[fn] for fn in feature_names)
                       for c, d in zip(t_cfg[fn] for fn in feature_names)
                       if a != d)
            if min_changes is None or dist < min_changes:
                min_changes = dist
                best_target = t_cfg
                best_target_idx = t_idx

        if min_changes is None:
            print(f"Config #{p_idx} cannot reach target with any valid configuration.")
            continue

        print(f"Config #{p_idx} (Perf: {p_cfg[perf_col]} ms) requires {min_changes} "
              f"setting change(s) to reach target performance.")
        print(f"  Recommended target configuration #{best_target_idx} "
              f"with performance {best_target[perf_col]} ms\n")

# ---------------------------------------------------------------------------

def main(csv_path):
    configs = read_csv(csv_path)
    if not configs:
        print("No data found in CSV.")
        return

    apache_15(configs)
    apache_16(configs)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <path_to_Apache_AllMeasurements.csv>")
        sys.exit(1)
    main(sys.argv[1])
