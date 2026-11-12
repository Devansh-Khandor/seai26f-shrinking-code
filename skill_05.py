#!/usr/bin/env python3
"""
Apache configuration analysis: identify settings associated with poor performance
and frequent poor‑performance combinations.

Usage:
    python skill_05.py <path_to_Apache_AllMeasurements.csv>
"""

import csv
import sys
import math
from collections import defaultdict

import numpy as np


# --------------------------------------------------------------------------- #
# 1. CSV reading and numeric conversion
# --------------------------------------------------------------------------- #
def read_csv(csv_path: str):
    """Read CSV file and return list of rows as dictionaries with numeric values."""
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

    # Convert numeric values
    for r in rows:
        for k, v in r.items():
            if k == perf_col:
                r[k] = float(v)
            else:
                try:
                    r[k] = int(v)
                except ValueError:
                    # fallback for unexpected non‑integer values
                    r[k] = 0

    return rows, perf_col


# --------------------------------------------------------------------------- #
# 2. APACHE-09: Settings most associated with unfavorable performance
# --------------------------------------------------------------------------- #
def apache_09(rows, perf_col, setting_cols):
    """Identify settings most strongly associated with higher (worse) performance."""
    # Compute correlation of each setting with performance
    perf_arr = np.array([r[perf_col] for r in rows])
    corrs = {}
    for col in setting_cols:
        X = np.array([r[col] for r in rows])
        # Handle case where X has zero variance
        if np.all(X == X[0]):
            corr = 0.0
        else:
            corr = np.corrcoef(X, perf_arr)[0, 1]
        corrs[col] = corr

    # Worst configuration (highest performance number = worst)
    worst_row = max(rows, key=lambda r: r[perf_col])

    # Rank by absolute correlation
    sorted_corrs = sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)

    # Output
    print("== APACHE-09 ==")
    print(f"Worst configuration (performance={worst_row[perf_col]:.2f}):")
    cfg_parts = [f"{col}={worst_row[col]}" for col in setting_cols]
    print("  " + ", ".join(cfg_parts))
    print("\nTop 3 settings most strongly associated with unfavorable performance:")
    for col, corr in sorted_corrs[:3]:
        assoc = "enabled" if corr > 0 else "disabled"
        # Determine which value in the worst config is linked to higher perf
        # If corr > 0, 1 leads to higher perf; else 0 leads to higher perf
        print(f"  {col}: correlation={corr:+.3f} ({assoc} in worst config)")


# --------------------------------------------------------------------------- #
# 3. APACHE-10: Frequent combinations among poor‑performance configurations
# --------------------------------------------------------------------------- #
def apache_10(rows, perf_col, setting_cols, poor_percentile=90):
    """Identify combinations of settings that appear frequently in poor configs."""
    perf_arr = np.array([r[perf_col] for r in rows])

    # Determine threshold for poor performance (top poor_percentile%)
    threshold = np.percentile(perf_arr, poor_percentile)
    poor_rows = [r for r in rows if r[perf_col] >= threshold]

    total_configs = len(rows)
    poor_configs = len(poor_rows)

    # Count occurrences of each joint value for each pair
    total_counts = defaultdict(int)
    poor_counts = defaultdict(int)

    for r in rows:
        for i in range(len(setting_cols)):
            for j in range(i + 1, len(setting_cols)):
                c1, c2 = setting_cols[i], setting_cols[j]
                val1, val2 = r[c1], r[c2]
                key = (c1, c2, val1, val2)
                total_counts[key] += 1

    for r in poor_rows:
        for i in range(len(setting_cols)):
            for j in range(i + 1, len(setting_cols)):
                c1, c2 = setting_cols[i], setting_cols[j]
                val1, val2 = r[c1], r[c2]
                key = (c1, c2, val1, val2)
                poor_counts[key] += 1

    # Compute differences in proportions
    diff_list = []
    for key, tot in total_counts.items():
        if tot == 0:
            continue
        poor = poor_counts.get(key, 0)
        total_prop = tot / total_configs
        poor_prop = poor / poor_configs if poor_configs > 0 else 0
        diff = poor_prop - total_prop
        diff_list.append((diff, key, poor_prop, total_prop))

    # Sort by largest positive difference
    diff_list.sort(key=lambda kv: kv[0], reverse=True)

    # Output
    print("\n== APACHE-10 ==")
    if not diff_list:
        print("No pair combinations found.")
        return

    print(f"Top combinations of settings appearing most frequently in the worst "
          f"{poor_percentile}% of configurations:")
    for i, (diff, key, poor_prop, total_prop) in enumerate(diff_list[:3], 1):
        c1, c2, v1, v2 = key
        print(f"  {i}. {c1}={v1}, {c2}={v2}")
        print(f"     Poor config proportion: {poor_prop:.3f}")
        print(f"     All config proportion:  {total_prop:.3f}")
        print(f"     Difference (+poor):    {diff:.3f}")


# --------------------------------------------------------------------------- #
# 4. Main driver
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, perf_col = read_csv(csv_path)

    setting_cols = [c for c in rows[0].keys() if c != perf_col]

    apache_09(rows, perf_col, setting_cols)
    apache_10(rows, perf_col, setting_cols, poor_percentile=90)


if __name__ == "__main__":
    main()
