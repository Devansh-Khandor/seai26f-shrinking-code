#!/usr/bin/env python3
"""
skill_06.py

Usage:
    python skill_06.py <path_to_Apache_AllMeasurements.csv>

This script implements:
  - APACHE-11: Explanation of recommended configurations by comparing
    them to the average configuration.
  - APACHE-12: Explanation of pairs of configurations that received
    different performance assessments, highlighting the settings
    that differ and how those differences may explain the
    performance gap.
"""

import csv
import sys
import math
from collections import defaultdict

# ----------------------------------------------------------------------
# CSV helpers
# ----------------------------------------------------------------------
def read_csv(path):
    """Read the CSV file and return a list of dicts with integer values
    and the fieldnames."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            new_row = {}
            for key, value in row.items():
                try:
                    new_row[key] = int(value)
                except ValueError:
                    # skip malformed rows
                    break
            else:
                rows.append(new_row)
        return rows, reader.fieldnames

# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def compute_global_stats(rows, perf_key):
    """Return mean, std, and list of all performance values."""
    values = [row[perf_key] for row in rows]
    n = len(values)
    if n == 0:
        return 0.0, 0.0, []
    mean = sum(values) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / n)
    return mean, std, values

def compute_mean_config(rows, feature_keys):
    """Return a dict of the average value for each setting,
    rounded to nearest integer (0 or 1)."""
    sums = defaultdict(int)
    for row in rows:
        for key in feature_keys:
            sums[key] += row[key]
    mean_config = {}
    for key in feature_keys:
        mean_config[key] = 1 if sums[key] >= len(rows) / 2 else 0
    return mean_config

# ----------------------------------------------------------------------
# APACHE-11 helpers
# ----------------------------------------------------------------------
def find_recommended(rows, perf_key, mean, std, top_n=5):
    """Return top_n configurations with performance > mean + std."""
    threshold = mean + std
    rec = [(row[perf_key], row) for row in rows if row[perf_key] > threshold]
    rec.sort(reverse=True, key=lambda t: t[0])
    return rec[:top_n]

def explain_recommendations(recs, mean_config, perf_key):
    """Print explanations for each recommended configuration."""
    if not recs:
        print("No configurations exceed one standard deviation above the mean.")
        return

    print(f"Number of recommended configurations: {len(recs)}")
    for idx, (perf, row) in enumerate(recs, 1):
        print(f"\nConfig #{idx} (Perf: {perf})")
        # Compare to mean configuration
        diff_settings = []
        for key, val in row.items():
            if key == perf_key:
                continue
            if val != mean_config[key]:
                diff_settings.append((key, val, mean_config[key]))

        if not diff_settings:
            print("  No settings differ from the average configuration.")
            continue

        print("  Settings differing from the average configuration:")
        for key, val, avg in diff_settings:
            direction = "increased" if val > avg else "decreased"
            print(f"    {key} {direction} from {avg} to {val}")
        # Summarise potential impact
        print(f"  These deviations likely contribute to the above performance advantage "
              f"compared to the mean of {mean}.")


# ----------------------------------------------------------------------
# APACHE-12 helpers
# ----------------------------------------------------------------------
def find_largest_gap_pair(rows, perf_key, feature_keys):
    """Return the pair of configurations with the largest absolute
    performance difference, along with the difference value."""
    if len(rows) < 2:
        return None, None, None
    max_diff = -1
    best_pair = (None, None)
    for i, r1 in enumerate(rows):
        for r2 in rows[i+1:]:
            diff = abs(r1[perf_key] - r2[perf_key])
            if diff > max_diff:
                max_diff = diff
                best_pair = (r1, r2)
    return best_pair[0], best_pair[1], max_diff

def explain_difference(r1, r2, perf_key, diff, feature_keys):
    """Print explanation for why two configurations differ."""
    print("\nAPACHE-12: Explanation of the largest performance gap")
    p1 = r1[perf_key]
    p2 = r2[perf_key]
    print(f"Config A performance: {p1}")
    print(f"Config B performance: {p2}")
    print(f"Absolute difference: {diff}")

    # Find differing settings
    differing = []
    for key in feature_keys:
        if r1[key] != r2[key]:
            differing.append((key, r1[key], r2[key]))

    if not differing:
        print("  No settings differ between the two configurations.")
        return

    print("  Differing settings:")
    for key, val1, val2 in differing:
        print(f"    {key}: Config A={val1}, Config B={val2}")

    # Brief interpretation
    print("\n  Interpretation:")
    for key, val1, val2 in differing:
        # Assume higher setting value tends to improve performance
        better = "A" if val1 > val2 else "B"
        print(f"    Setting {key} is {better} for the configuration with higher performance "
              f"({better}={max(val1, val2)}).")

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

    feature_keys = [k for k in fieldnames if k.strip().lower() != 'performance-']

    # Global stats
    mean_perf, std_perf, _ = compute_global_stats(rows, perf_key)

    # ------------------------------------------------------------------
    # APACHE-11
    # ------------------------------------------------------------------
    print("== APACHE-11 ==")
    mean_config = compute_mean_config(rows, feature_keys)
    recs = find_recommended(rows, perf_key, mean_perf, std_perf)
    explain_recommendations(recs, mean_config, perf_key)

    # ------------------------------------------------------------------
    # APACHE-12
    # ------------------------------------------------------------------
    print("\n== APACHE-12 ==")
    r1, r2, diff = find_largest_gap_pair(rows, perf_key, feature_keys)
    if r1 is None:
        print("Not enough configurations to compare.")
    else:
        explain_difference(r1, r2, perf_key, diff, feature_keys)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <path_to_csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
