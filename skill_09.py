#!/usr/bin/env python3
"""
Apache configuration anomaly detection.

Implements:
  * APACHE-17 – Identify configurations whose recorded performance values are
                unusually different from what their settings and comparable
                historical configurations would lead us to expect.
  * APACHE-18 – Flag configurations whose combinations of settings differ
                substantially from the most similar configurations in our
                historical benchmark records, even when their measured
                performance is not obviously poor.

Usage:
    python skill_09.py <path_to_Apache_AllMeasurements.csv>
"""

import csv
import sys
import math
from collections import defaultdict
from statistics import mean, stdev

import numpy as np
from sklearn.linear_model import LinearRegression


# --------------------------------------------------------------------------- #
# CSV reading helper
# --------------------------------------------------------------------------- #
def read_csv(csv_path: str):
    """Read CSV and return rows (list of dicts) and performance column name."""
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
# Helper: compute prediction residuals
# --------------------------------------------------------------------------- #
def compute_residuals(rows, perf_col, setting_cols):
    """Return residuals array and model."""
    X = np.array([[r[col] for col in setting_cols] for r in rows])
    y = np.array([r[perf_col] for r in rows])

    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)
    residuals = y - preds  # positive if actual > predicted (worse)
    return residuals, model


# --------------------------------------------------------------------------- #
# APACHE-17: Unusually different performance
# --------------------------------------------------------------------------- #
def apache_17(rows, perf_col, setting_cols):
    """Find configurations with residuals beyond 2 standard deviations."""
    residuals, _ = compute_residuals(rows, perf_col, setting_cols)

    mean_res = mean(residuals)
    if len(residuals) > 1:
        std_res = stdev(residuals)
    else:
        std_res = 0.0

    threshold = mean_res + 2 * std_res
    anomalies = []
    for r, resid in zip(rows, residuals):
        if resid > threshold:
            anomalies.append((r, resid))

    # Sort by magnitude descending
    anomalies.sort(key=lambda kv: kv[1], reverse=True)

    print("\n== APACHE-17 ==")
    print(f"Residual mean: {mean_res:.2f}, std: {std_res:.2f}")
    if not anomalies:
        print("No configurations with unusually high residual performance found.")
        return

    print(f"Configurations with residual > mean + 2*std ({threshold:.2f}):")
    for r, resid in anomalies[:5]:
        cfg = ", ".join(f"{c}={r[c]}" for c in setting_cols)
        print(f"  Residual={resid:.2f} | {cfg} | Measured={r[perf_col]:.2f}")


# --------------------------------------------------------------------------- #
# Helper: Hamming distance
# --------------------------------------------------------------------------- #
def hamming(a: dict, b: dict, setting_cols):
    return sum(1 for c in setting_cols if a[c] != b[c])


# --------------------------------------------------------------------------- #
# APACHE-18: Unusually dissimilar configuration patterns
# --------------------------------------------------------------------------- #
def apache_18(rows, perf_col, setting_cols):
    """Flag configurations that are far from any other configuration."""
    n = len(rows)
    if n <= 1:
        print("\n== APACHE-18 ==")
        print("Not enough data to compute configuration similarity.")
        return

    # Precompute distances to nearest neighbor for each config
    distances = []
    for i, r in enumerate(rows):
        min_dist = n  # max possible is n-1
        for j, r2 in enumerate(rows):
            if i == j:
                continue
            d = hamming(r, r2, setting_cols)
            if d < min_dist:
                min_dist = d
                if min_dist == 0:
                    break  # identical
        distances.append(min_dist)

    mean_dist = mean(distances)
    if len(distances) > 1:
        std_dist = stdev(distances)
    else:
        std_dist = 0.0

    threshold = mean_dist + 1.5 * std_dist
    outliers = [r for r, d in zip(rows, distances) if d > threshold]

    print("\n== APACHE-18 ==")
    print(f"Mean nearest‑neighbor Hamming distance: {mean_dist:.2f}")
    print(f"Std of distances: {std_dist:.2f}")
    if not outliers:
        print("No configurations with unusually high Hamming distance found.")
        return

    print(f"Configurations with distance > mean + 1.5*std ({threshold:.2f}):")
    for r in outliers[:5]:
        cfg = ", ".join(f"{c}={r[c]}" for c in setting_cols)
        print(f"  Distance={hamming(r, rows[0], setting_cols)} | {cfg} | Measured={r[perf_col]:.2f}")


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, perf_col = read_csv(csv_path)
    setting_cols = [c for c in rows[0].keys() if c != perf_col]

    apache_17(rows, perf_col, setting_cols)
    apache_18(rows, perf_col, setting_cols)


if __name__ == "__main__":
    main()
