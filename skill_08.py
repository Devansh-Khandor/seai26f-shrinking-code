#!/usr/bin/env python3
"""
Apache configuration optimization.

Implements:
  * APACHE-15 – Suggest a single setting change for a poorly performing
                configuration that is predicted to yield better performance.
  * APACHE-16 – Find the minimal number of setting changes that move a
                poorly performing configuration towards a specified target
                performance, using only historically observed valid
                configurations.

Usage:
    python skill_08.py <path_to_Apache_AllMeasurements.csv>
"""

import csv
import sys
import math
from collections import defaultdict
from itertools import combinations

import numpy as np
from sklearn.linear_model import LinearRegression


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def read_csv(csv_path: str):
    """Read CSV, convert numeric values, return rows and the name of the
    performance column."""
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

    # Identify performance column (case‑insensitive startswith "performance")
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
                try:
                    r[k] = float(v)
                except ValueError:
                    r[k] = float("nan")
            else:
                try:
                    r[k] = int(v)
                except ValueError:
                    r[k] = 0

    return rows, perf_col


def get_setting_cols(rows, perf_col):
    """Return list of setting column names."""
    return [c for c in rows[0].keys() if c != perf_col]


def hamming_distance(a: dict, b: dict, setting_cols):
    """Return number of settings that differ between two configs."""
    return sum(1 for col in setting_cols if a[col] != b[col])


def diff_settings(a: dict, b: dict, setting_cols):
    """Return list of (setting, val_a, val_b) where they differ."""
    return [(col, a[col], b[col]) for col in setting_cols if a[col] != b[col]]


# --------------------------------------------------------------------------- #
# APACHE-15: Suggest a single setting change
# --------------------------------------------------------------------------- #
def apache_15(rows, perf_col, setting_cols):
    """Recommend a one‑setting change for the worst configuration that
    is predicted to give lower performance."""
    # Identify the worst configuration (highest measured performance)
    worst_cfg = max(rows, key=lambda r: r[perf_col])
    worst_pred_perf = worst_cfg[perf_col]  # measured; use as reference

    # Fit linear regression on all data to predict performance
    X = np.array([[r[col] for col in setting_cols] for r in rows])
    y = np.array([r[perf_col] for r in rows])
    model = LinearRegression()
    model.fit(X, y)

    # Find all configurations that differ by exactly one setting
    best_pred = None
    best_cfg = None
    improvement = 0.0
    for r in rows:
        if r is worst_cfg:
            continue
        if hamming_distance(r, worst_cfg, setting_cols) == 1:
            pred_perf = model.predict([ [r[col] for col in setting_cols] ])[0]
            if best_pred is None or pred_perf < best_pred:
                best_pred = pred_perf
                best_cfg = r
                improvement = worst_pred_perf - pred_perf

    print("== APACHE-15 ==")
    print(f"Worst configuration (measured): {worst_cfg[perf_col]:.2f}")
    if best_cfg is None:
        print("No single‑setting change improves the predicted performance.")
        return

    diff = diff_settings(worst_cfg, best_cfg, setting_cols)
    print(f"Recommended single setting change:")
    for col, old, new in diff:
        print(f"  {col}: {old} -> {new}")
    print(f"Predicted performance after change: {best_pred:.2f}")
    print(f"Estimated improvement (lower is better): {improvement:.2f}")


# --------------------------------------------------------------------------- #
# APACHE-16: Minimal changes to reach target performance
# --------------------------------------------------------------------------- #
def apache_16(rows, perf_col, setting_cols):
    """Find minimal number of setting changes that bring worst config
    closer to a target performance, using only observed configurations."""
    # Define target as the median performance (i.e., half of configs are better)
    perf_arr = np.array([r[perf_col] for r in rows])
    target_perf = np.median(perf_arr)

    # Identify the worst configuration
    worst_cfg = max(rows, key=lambda r: r[perf_col])

    # Find all configurations with performance <= target (i.e., better or equal)
    good_configs = [r for r in rows if r[perf_col] <= target_perf]

    if not good_configs:
        print("== APACHE-16 ==")
        print("No configuration in the dataset achieves the target performance.")
        return

    # For each good configuration, compute Hamming distance to worst
    best_configs = []
    best_dist = None
    for cfg in good_configs:
        dist = hamming_distance(cfg, worst_cfg, setting_cols)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_configs = [cfg]
        elif dist == best_dist:
            best_configs.append(cfg)

    # If multiple with same minimal distance, pick the one with lowest perf
    best_cfg = min(best_configs, key=lambda r: r[perf_col])

    diff = diff_settings(worst_cfg, best_cfg, setting_cols)

    print("== APACHE-16 ==")
    print(f"Target performance (median): {target_perf:.2f}")
    print(f"Worst configuration (measured): {worst_cfg[perf_col]:.2f}")
    print(f"Minimal number of setting changes needed: {best_dist}")
    print("Configuration changes to achieve target:")
    for col, old, new in diff:
        print(f"  {col}: {old} -> {new}")
    print(f"New configuration's performance: {best_cfg[perf_col]:.2f}")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, perf_col = read_csv(csv_path)
    setting_cols = get_setting_cols(rows, perf_col)

    apache_15(rows, perf_col, setting_cols)
    print("\n")
    apache_16(rows, perf_col, setting_cols)


if __name__ == "__main__":
    main()
