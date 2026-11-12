#!/usr/bin/env python3
"""
Apache configuration performance estimation and comparison.

Usage:
    python skill_02.py <path_to_Apache_AllMeasurements.csv>
"""

import csv
import math
import sys
from collections import defaultdict
from statistics import mean, stdev

import numpy as np
from sklearn.linear_model import LinearRegression


def read_csv(csv_path):
    """Read CSV and return list of dicts with numeric conversion."""
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

    # Find performance column (case‑insensitive startswith "performance")
    perf_col = None
    for col in rows[0].keys():
        if col.lower().startswith("performance"):
            perf_col = col
            break
    if perf_col is None:
        print("Performance column not found.")
        sys.exit(1)

    # Convert values to int (or float for performance)
    for r in rows:
        for k, v in r.items():
            if k == perf_col:
                r[k] = float(v)
            else:
                r[k] = int(v)

    return rows, perf_col


def fit_regression(rows, perf_col, setting_cols):
    """Fit a linear regression model to predict performance."""
    X = np.array([[r[col] for col in setting_cols] for r in rows])
    y = np.array([r[perf_col] for r in rows])

    model = LinearRegression()
    model.fit(X, y)
    return model, y


def evaluate_model(model, X, y):
    """Return MAE and R^2 of the model."""
    y_pred = model.predict(X)
    mae = float(np.mean(np.abs(y - y_pred)))
    r2 = float(model.score(X, y))
    return mae, r2


def estimate_diff_per_setting(rows, perf_col, setting_cols):
    """
    For each setting, compute average performance difference when flipping
    that setting while keeping all other settings identical, and its std dev.
    Returns a dict: setting -> (avg_diff, std_dev)
    """
    # Map from (other settings key) -> list of (value, perf)
    diff_stats = {}
    for col in setting_cols:
        # Build pairs
        pair_diffs = []

        # Use a dict keyed by tuple of all other settings
        config_map = defaultdict(list)
        for r in rows:
            key = tuple(r[s] for s in setting_cols if s != col)
            config_map[key].append((r[col], r[perf_col]))

        for key, items in config_map.items():
            # items may contain 0 and/or 1 values
            val_perf = {v: p for v, p in items}
            if 0 in val_perf and 1 in val_perf:
                diff = val_perf[1] - val_perf[0]  # performance with setting=1 minus with 0
                pair_diffs.append(diff)

        if pair_diffs:
            avg = mean(pair_diffs)
            std = stdev(pair_diffs) if len(pair_diffs) > 1 else 0.0
            diff_stats[col] = (avg, std)

    return diff_stats


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, perf_col = read_csv(csv_path)

    # Identify setting columns (exclude performance)
    setting_cols = [c for c in rows[0].keys() if c != perf_col]

    # --------------------- APACHE-03 ---------------------
    print("== APACHE-03 ==")

    X = np.array([[r[col] for col in setting_cols] for r in rows])
    y = np.array([r[perf_col] for r in rows])

    model, y_true = fit_regression(rows, perf_col, setting_cols)
    mae, r2 = evaluate_model(model, X, y_true)

    # Show a few predictions
    predictions = model.predict(X)
    sample_lines = []
    for i in range(min(5, len(rows))):
        cfg = ", ".join(f"{col}={rows[i][col]}" for col in setting_cols)
        sample_lines.append(f"Pred: {int(predictions[i])} | Actual: {int(rows[i][perf_col])} | {cfg}")

    print(f"Mean Absolute Error of predictions: {mae:.2f}")
    print(f"R^2 of the regression model: {r2:.4f}")
    print("Sample predictions (first 5 rows):")
    for line in sample_lines:
        print(line)

    # --------------------- APACHE-04 ---------------------
    print("\n== APACHE-04 ==")
    diff_stats = estimate_diff_per_setting(rows, perf_col, setting_cols)

    if not diff_stats:
        print("No comparable configuration pairs found.")
        return

    # Output per-setting difference and uncertainty
    print("Estimated performance difference for a single setting change:")
    for col, (avg_diff, std_dev) in sorted(diff_stats.items(), key=lambda kv: abs(kv[1][0]), reverse=True):
        print(f"  {col}: ΔPerf = {avg_diff:.2f} ± {std_dev:.2f}")

    # Optionally, show the setting with the largest magnitude effect
    best_col = max(diff_stats.items(), key=lambda kv: abs(kv[1][0]))[0]
    best_avg, best_std = diff_stats[best_col]
    print(f"\nMost impactful setting: {best_col}")
    print(f"  Average difference: {best_avg:.2f}")
    print(f"  Uncertainty (std dev): {best_std:.2f}")


if __name__ == "__main__":
    main()
