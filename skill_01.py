#!/usr/bin/env python3
"""
skill_01.py – Summary and anomaly detection for Apache configuration performance.

Usage:
    python skill_01.py <path-to-Apache_AllMeasurements.csv>
"""

import sys
import csv
from collections import Counter
import math

def load_csv(path):
    """Load CSV into a list of dict rows and return column names."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames
    return rows, headers

def compute_summary(perf_values):
    """Return typical (median), best (min), and worst (max) performance."""
    sorted_vals = sorted(perf_values)
    n = len(sorted_vals)
    median = sorted_vals[n // 2] if n % 2 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    best = min(sorted_vals)
    worst = max(sorted_vals)
    return median, best, worst

def find_favorable(rows, perf_values, z_thresh=2.0):
    """Return rows with unusually favorable performance."""
    mean = sum(perf_values) / len(perf_values)
    std = math.sqrt(sum((x - mean) ** 2 for x in perf_values) / len(perf_values))
    threshold = mean - z_thresh * std
    fav = [row for row, p in zip(rows, perf_values) if p <= threshold]
    return fav, threshold

def majority_value(values):
    """Return the most common value in a list; ties broken arbitrarily."""
    counter = Counter(values)
    return counter.most_common(1)[0][0]

def distinguishing_settings(fav_row, all_rows, headers):
    """Return dict of columns where fav_row differs from majority value."""
    # Compute majority per column
    majority = {}
    for h in headers:
        majority[h] = majority_value([row[h] for row in all_rows])
    # Find differences
    diff = {}
    for h in headers:
        if fav_row[h] != majority[h]:
            diff[h] = fav_row[h]
    return diff

def main(csv_path):
    rows, headers = load_csv(csv_path)
    # Assume 'Performance-' column exists and contains numeric values
    perf_col = 'Performance-'
    perf_values = [int(row[perf_col]) for row in rows]

    # APACHE-01
    median, best, worst = compute_summary(perf_values)
    print("== APACHE-01 ==")
    print(f"Typical performance (median): {median}")
    print(f"Best performance (min): {best}")
    print(f"Worst performance (max): {worst}")

    # APACHE-02
    fav_rows, threshold = find_favorable(rows, perf_values)
    print("\n== APACHE-02 ==")
    if not fav_rows:
        print("No configurations exhibit unusually favorable performance.")
    else:
        print(f"Threshold for unusually favorable performance: {threshold:.2f}")
        for idx, row in enumerate(fav_rows, start=1):
            print(f"\nConfiguration #{idx} (Performance={row[perf_col]}):")
            diff = distinguishing_settings(row, rows, headers)
            if diff:
                print("Distinguishing settings:")
                for key, val in diff.items():
                    print(f"  {key}: {val}")
            else:
                print("  No distinguishing settings found.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python skill_01.py <csv_path>")
    main(sys.argv[1])
