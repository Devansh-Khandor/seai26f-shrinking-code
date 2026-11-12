#!/usr/bin/env python3
"""
skill_04.py

Usage:
    python skill_04.py <path_to_Apache_AllMeasurements.csv>
"""

import csv
import sys
import math
from collections import defaultdict

# ----------------------------------------------------------------------
# CSV helpers
# ----------------------------------------------------------------------
def read_csv(path):
    """Read the CSV file and return a list of dicts with integer values."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            new_row = {}
            for key, value in row.items():
                try:
                    new_row[key] = int(value)
                except ValueError:
                    # Skip rows that can't be parsed
                    break
            else:
                rows.append(new_row)
        return rows, reader.fieldnames

# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def describe_stats(values):
    n = len(values)
    if n == 0:
        return n, None, None, None, None
    mean = sum(values) / n
    sorted_vals = sorted(values)
    median = sorted_vals[n // 2] if n % 2 else (sorted_vals[n//2-1] + sorted_vals[n//2]) / 2
    std = math.sqrt(sum((x - mean)**2 for x in values) / n)
    return n, mean, median, std, max(values), min(values)

# ----------------------------------------------------------------------
# APACHE-07: Real‑time alert for poor performance
# ----------------------------------------------------------------------
def apache_07(rows, perf_key):
    """
    For each configuration, compare its performance against the running mean
    and standard deviation of all previously processed configurations.
    """
    alerts = []
    prior_perfs = []

    for idx, row in enumerate(rows, 1):
        perf = row[perf_key]
        if prior_perfs:
            mean = sum(prior_perfs) / len(prior_perfs)
            std = math.sqrt(sum((x - mean)**2 for x in prior_perfs) / len(prior_perfs))
            threshold = mean - std
            if perf < threshold:
                alerts.append((idx, perf, threshold, mean, std))
        prior_perfs.append(perf)

    return alerts

# ----------------------------------------------------------------------
# APACHE-08: Updated summaries and recommendation shortlist
# ----------------------------------------------------------------------
def apache_08(rows, perf_key):
    """
    Compute overall statistics and generate a shortlist of recommended
    configurations that outperform the mean by more than one standard
    deviation.
    """
    perf_values = [row[perf_key] for row in rows]
    n, mean, median, std, best, worst = describe_stats(perf_values)

    # Identify favorable configurations
    threshold = mean + std
    favorable = [(idx, row[perf_key]) for idx, row in enumerate(rows, 1) if row[perf_key] > threshold]
    favorable.sort(key=lambda t: t[1], reverse=True)
    shortlist = favorable[:5]  # top 5

    return {
        'n': n,
        'mean': mean,
        'median': median,
        'std': std,
        'best': best,
        'worst': worst,
        'shortlist': shortlist,
    }

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path):
    rows, fieldnames = read_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        sys.exit(1)

    perf_key = next((k for k in fieldnames if k.strip().lower() == 'performance-'), None)
    if perf_key is None:
        print("Error: No 'Performance-' column found.")
        sys.exit(1)

    # APACHE-07
    print("== APACHE-07 ==")
    alerts = apache_07(rows, perf_key)
    print(f"Number of unusually poor configurations detected: {len(alerts)}")
    if alerts:
        first = alerts[0]
        print(f"First alert at configuration #{first[0]}: performance={first[1]} "
              f"(threshold={first[2]:.2f} based on mean={first[3]:.2f}, std={first[4]:.2f})")
        for idx, perf, threshold, mean, std in alerts[:5]:
            print(f"  Config #{idx}: perf={perf}, threshold={threshold:.2f}")
    else:
        print("No configuration was unusually poor relative to prior data.")
    print()

    # APACHE-08
    print("== APACHE-08 ==")
    stats = apache_08(rows, perf_key)
    print(f"Total configurations: {stats['n']}")
    print(f"Mean performance: {stats['mean']:.2f}")
    print(f"Median performance: {stats['median']:.2f}")
    print(f"Best performance: {stats['best']}")
    print(f"Worst performance: {stats['worst']}")
    print(f"Standard deviation: {stats['std']:.2f}")
    print("\nRecommended configurations (top 5 > mean + std):")
    if stats['shortlist']:
        for idx, perf in stats['shortlist']:
            print(f"  Config #{idx}: performance={perf}")
    else:
        print("  No configuration exceeds mean + std threshold.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <path_to_csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
