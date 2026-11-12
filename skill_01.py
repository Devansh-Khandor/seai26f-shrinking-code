#!/usr/bin/env python3
"""
Apache configuration performance analysis.

Usage:
    python skill_01.py <path_to_Apache_AllMeasurements.csv>
"""

import csv
import math
import sys
from collections import defaultdict

import numpy as np


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Read the CSV into a list of dictionaries
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

    # Identify the performance column (any column that starts with "performance")
    perf_col = None
    for col in rows[0].keys():
        if col.lower().startswith("performance"):
            perf_col = col
            break

    if perf_col is None:
        print("Performance column not found.")
        sys.exit(1)

    # Convert numeric fields to int (or float for performance)
    for r in rows:
        for k, v in r.items():
            if k == perf_col:
                r[k] = float(v)
            else:
                r[k] = int(v)

    # Extract performance values
    performances = np.array([r[perf_col] for r in rows])

    # Summary statistics
    mean_perf = performances.mean()
    median_perf = np.median(performances)
    min_perf = performances.min()
    max_perf = performances.max()

    # --------------------- APACHE-01 ---------------------
    print("== APACHE-01 ==")
    print(f"Typical performance (mean): {mean_perf:.2f}")
    print(f"Best performance (min): {int(min_perf)}")
    print(f"Worst performance (max): {int(max_perf)}")

    # --------------------- APACHE-02 ---------------------
    # Determine the "unusually favorable" configurations:
    #   - Top 10% lowest performance values (or at least one entry)
    total_configs = len(rows)
    top_k = max(1, math.ceil(0.10 * total_configs))
    sorted_indices = np.argsort(performances)
    top_indices = sorted_indices[:top_k]
    top_configs = [rows[i] for i in top_indices]

    # Columns that define a configuration (exclude performance and any constant column)
    setting_cols = [
        col for col in rows[0].keys() if col != perf_col and rows[0][col] in ("0", "1")
    ]

    # Compute proportions of '1' for each setting across all configs
    overall_counts = defaultdict(int)
    for r in rows:
        for col in setting_cols:
            overall_counts[col] += r[col]
    overall_props = {col: overall_counts[col] / total_configs for col in setting_cols}

    # Compute proportions of '1' for each setting within the top group
    top_counts = defaultdict(int)
    for r in top_configs:
        for col in setting_cols:
            top_counts[col] += r[col]
    top_props = {col: top_counts[col] / len(top_configs) for col in setting_cols}

    # Identify settings that differ by at least 0.3 in proportion of '1'
    distinguishing = [
        (col, top_props[col], overall_props[col])
        for col in setting_cols
        if abs(top_props[col] - overall_props[col]) >= 0.3
    ]

    print("\n== APACHE-02 ==")
    print(f"Total configurations analysed: {total_configs}")
    print(f"Top {len(top_configs)} configurations with unusually favorable performance:")
    for r in top_configs:
        cfg_parts = [f"{col}={r[col]}" for col in setting_cols]
        cfg_str = ", ".join(cfg_parts)
        print(f"{cfg_str} ; Performance={int(r[perf_col])}")

    if distinguishing:
        print("\nDistinguishing settings (difference in proportion of '1' ≥ 0.30):")
        for col, top_p, overall_p in distinguishing:
            print(f"  {col}: top={top_p:.2f}, overall={overall_p:.2f}")
    else:
        print("\nNo distinguishing settings found.")


if __name__ == "__main__":
    main()
