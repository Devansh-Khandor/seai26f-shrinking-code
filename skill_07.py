#!/usr/bin/env python3
"""
Apache configuration discovery and recommendation.

Implements:
  * APACHE-13 – Find a low‑performance configuration by examining only a subset
                 of the available data, thus keeping the number of evaluated
                 candidates small.
  * APACHE-14 – With a limited benchmarking budget, suggest which
                 configurations to evaluate first.  We use a simple linear
                 regression model trained on the full data to predict
                 performance and then recommend the best predicted
                 configurations.

Usage:
    python skill_07.py <path_to_Apache_AllMeasurements.csv>
"""

import csv
import sys
import random
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------------- #
# Helper: read and convert CSV
# --------------------------------------------------------------------------- #
def read_csv(csv_path: str):
    """Read CSV, convert numeric values, return rows and performance column name."""
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
# APACHE-13: Find low‑performance config with minimal evaluation
# --------------------------------------------------------------------------- #
def apache_13(rows, perf_col, setting_cols, sample_frac=0.1):
    """
    Scan a random subset of the data and return the configuration with the
    lowest performance metric found in that subset.
    """
    n_total = len(rows)
    sample_size = max(1, int(n_total * sample_frac))
    random.seed(42)  # reproducible sampling
    sample_indices = random.sample(range(n_total), sample_size)
    sample_rows = [rows[i] for i in sample_indices]

    # Find best in sample
    best_sample = min(sample_rows, key=lambda r: r[perf_col])

    # For context, find best in full dataset
    best_full = min(rows, key=lambda r: r[perf_col])

    return best_sample, best_full, sample_size, n_total


# --------------------------------------------------------------------------- #
# APACHE-14: Recommend configs to evaluate first
# --------------------------------------------------------------------------- #
def apache_14(rows, perf_col, setting_cols, top_n=5):
    """
    Train a linear regression model on the full dataset and predict performance
    for every configuration.  Recommend the `top_n` configurations with the
    lowest predicted performance.
    """
    X = np.array([[r[col] for col in setting_cols] for r in rows])
    y = np.array([r[perf_col] for r in rows])

    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)
    # Attach predictions to rows
    scored = [(pred, r) for pred, r in zip(preds, rows)]
    scored.sort(key=lambda t: t[0])  # ascending predicted performance

    # Take top_n unique configurations
    top_cfgs = []
    seen_keys = set()
    for pred, r in scored:
        key = tuple((col, r[col]) for col in setting_cols)
        if key not in seen_keys:
            seen_keys.add(key)
            top_cfgs.append((pred, r))
        if len(top_cfgs) >= top_n:
            break

    return top_cfgs


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, perf_col = read_csv(csv_path)
    setting_cols = [c for c in rows[0].keys() if c != perf_col]

    # ----------------- APACHE-13 -----------------
    best_sample, best_full, sample_size, total = apache_13(rows, perf_col, setting_cols)

    print("== APACHE-13 ==")
    print(f"Sampled {sample_size} of {total} configurations.")
    cfg_parts = [f"{c}={best_sample[c]}" for c in setting_cols]
    cfg_str = ", ".join(cfg_parts)
    print(f"Discovered config with lowest performance in sample:")
    print(f"  {cfg_str} ; Performance={best_sample[perf_col]:.2f}")
    print(f"Actual best performance in full dataset: {best_full[perf_col]:.2f}")

    # ----------------- APACHE-14 -----------------
    top_cfgs = apache_14(rows, perf_col, setting_cols, top_n=5)

    print("\n== APACHE-14 ==")
    print("Recommended configurations to evaluate first (based on predicted performance):")
    for idx, (pred, r) in enumerate(top_cfgs, 1):
        cfg_parts = [f"{c}={r[c]}" for c in setting_cols]
        cfg_str = ", ".join(cfg_parts)
        print(f"  {idx}. Predicted={pred:.2f} | {cfg_str}")

    print("\nNote: These recommendations are based on a linear regression model trained on all available data. "
          "Evaluating them early will help surface promising server settings with fewer benchmark runs.")


if __name__ == "__main__":
    main()
