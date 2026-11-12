#!/usr/bin/env python3
"""
skill_01.py

Usage:
    python skill_01.py <path_to_Apache_AllMeasurements.csv>
"""

import csv
import sys
import math
from collections import defaultdict

def read_csv(path):
    """Read the CSV file and return a list of dicts with integer values."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # Convert all fields to int except the performance column
            new_row = {}
            for key, value in row.items():
                if key.strip().lower() == 'performance-':
                    new_row[key] = int(value)
                else:
                    new_row[key] = int(value)
            rows.append(new_row)
        return rows, reader.fieldnames

def compute_statistics(rows, perf_key):
    performances = [row[perf_key] for row in rows]
    count = len(performances)
    mean = sum(performances) / count
    std = math.sqrt(sum((x - mean) ** 2 for x in performances) / count)
    median = sorted(performances)[count // 2]
    best = max(performances)
    worst = min(performances)
    return {
        'count': count,
        'median': median,
        'mean': mean,
        'std': std,
        'best': best,
        'worst': worst,
    }

def find_favorable(rows, perf_key, stats):
    """Return rows whose performance is more than one std above the mean."""
    threshold = stats['mean'] + stats['std']
    return [row for row in rows if row[perf_key] > threshold]

def column_means(rows, exclude_keys):
    """Compute mean of each column (excluding specified keys)."""
    sums = defaultdict(int)
    counts = defaultdict(int)
    for row in rows:
        for key, val in row.items():
            if key in exclude_keys:
                continue
            sums[key] += val
            counts[key] += 1
    means = {k: sums[k] / counts[k] for k in sums}
    return means

def distinguish_settings(row, col_means):
    """Return settings that differ significantly from the mean (>0.1 difference)."""
    diff = {}
    for key, val in row.items():
        if key in col_means:
            mean_val = col_means[key]
            if abs(val - mean_val) > 0.1:
                diff[key] = val
    return diff

def main(csv_path):
    rows, fieldnames = read_csv(csv_path)
    perf_key = next((k for k in fieldnames if k.strip().lower() == 'performance-'), None)
    if perf_key is None:
        print("Error: No 'Performance-' column found.")
        sys.exit(1)

    stats = compute_statistics(rows, perf_key)

    # APACHE-01
    print("== APACHE-01 ==")
    print(f"Total configurations: {stats['count']}")
    print(f"Typical (median) performance: {stats['median']}")
    print(f"Best recorded performance: {stats['best']}")
    print(f"Worst recorded performance: {stats['worst']}")

    # APACHE-02
    favorable_rows = find_favorable(rows, perf_key, stats)
    print("\n== APACHE-02 ==")
    print(f"Number of unusually favorable configurations: {len(favorable_rows)}")
    if not favorable_rows:
        print("No configurations exceed one standard deviation above the mean.")
        return

    # Compute column means for distinction
    exclude_keys = {perf_key}
    col_means = column_means(rows, exclude_keys)

    for idx, row in enumerate(favorable_rows, 1):
        print(f"\nFavorable config #{idx} (Performance: {row[perf_key]})")
        diffs = distinguish_settings(row, col_means)
        if not diffs:
            print("  No settings differ noticeably from the average.")
        else:
            for k, v in diffs.items():
                print(f"  {k}: {v} (average: {col_means[k]:.2f})")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <path_to_csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
