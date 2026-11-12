#!/usr/bin/env python3
"""
skill_05.py

Usage:
    python skill_05.py <path_to_Apache_AllMeasurements.csv>

This script implements:
  - APACHE-09: For each poorly performing configuration, report the
    settings that are most strongly correlated with low performance.
  - APACHE-10: Identify frequent setting/value combinations that appear
    in poor configurations and compare their occurrence rates to
    better‑performing configurations.
"""

import csv
import sys
import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

# ----------------------------------------------------------------------
# CSV helpers
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[Dict[str, int]], List[str]]:
    """Read the CSV file and return a list of dicts with integer values and fieldnames."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, int]] = []
        for row in reader:
            new_row: Dict[str, int] = {}
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
def percentile(values: List[float], p: float) -> float:
    """Return the p-th percentile (0‑100) of a list of numbers."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[int(f)] * (c - k)
    d1 = sorted_vals[int(c)] * (k - f)
    return d0 + d1

# ----------------------------------------------------------------------
# Correlation helpers
# ----------------------------------------------------------------------
def compute_correlations(rows: List[Dict[str, int]],
                         perf_key: str) -> Dict[str, float]:
    """Return Pearson correlation for each binary setting with performance."""
    n = len(rows)
    perf_vals = [row[perf_key] for row in rows]
    mean_perf = sum(perf_vals) / n
    std_perf = math.sqrt(sum((x - mean_perf) ** 2 for x in perf_vals) / n)

    corr: Dict[str, float] = {}
    for key in rows[0].keys():
        if key == perf_key:
            continue
        setting_vals = [row[key] for row in rows]
        mean_setting = sum(setting_vals) / n
        std_setting = math.sqrt(sum((x - mean_setting) ** 2 for x in setting_vals) / n)
        if std_setting == 0 or std_perf == 0:
            continue
        cov = sum((setting_vals[i] - mean_setting) * (perf_vals[i] - mean_perf)
                  for i in range(n)) / n
        corr[key] = cov / (std_setting * std_perf)
    return corr

# ----------------------------------------------------------------------
# APACHE-09 helpers
# ----------------------------------------------------------------------
def get_top_settings_for_config(row: Dict[str, int],
                                corr: Dict[str, float],
                                top_n: int = 3) -> List[str]:
    """Return up to top_n setting=value pairs that are most strongly
    associated with low performance for the given configuration."""
    # For each setting, compute the sign‑adjusted association:
    # if corr is negative and value is 1 → contributes to low perf
    # if corr is positive and value is 0 → contributes to low perf
    assoc: List[Tuple[float, str]] = []
    for key, value in row.items():
        if key == 'Performance-':
            continue
        if key not in corr:
            continue
        r = corr[key]
        if (r < 0 and value == 1) or (r > 0 and value == 0):
            assoc.append((abs(r), f"{key}={value} (r={r:.3f})"))
    assoc.sort(reverse=True, key=lambda x: x[0])
    return [a[1] for a in assoc[:top_n]]

# ----------------------------------------------------------------------
# APACHE-10 helpers
# ----------------------------------------------------------------------
def pair_frequencies(rows: List[Dict[str, int]],
                     perf_key: str,
                     is_poor: bool) -> Counter:
    """Return Counter of setting=value pair strings for either poor or good configs."""
    pairs = Counter()
    for row in rows:
        if (is_poor and row[perf_key] <= poor_threshold) or \
           (not is_poor and row[perf_key] >= good_threshold):
            for key, value in row.items():
                if key == perf_key:
                    continue
                pairs[f"{key}={value}"] += 1
    return pairs

def find_top_pairs(poor_rows: List[Dict[str, int]],
                   good_rows: List[Dict[str, int]],
                   perf_key: str,
                   top_n: int = 3) -> List[Tuple[str, int, int, int]]:
    """Return top_n pairs that are most overrepresented in poor configs.
    Each entry: (pair_string, freq_poor, freq_good, diff)."""
    # Build pair counts for poor and good separately
    poor_counts = Counter()
    good_counts = Counter()
    for row in poor_rows:
        for key, value in row.items():
            if key == perf_key:
                continue
            poor_counts[f"{key}={value}"] += 1
    for row in good_rows:
        for key, value in row.items():
            if key == perf_key:
                continue
            good_counts[f"{key}={value}"] += 1
    # Find pairs that appear in both
    common_pairs = set(poor_counts) & set(good_counts)
    diffs = []
    for pair in common_pairs:
        diff = poor_counts[pair] - good_counts[pair]
        if diff > 0:
            diffs.append((pair, poor_counts[pair], good_counts[pair], diff))
    # Sort by difference descending
    diffs.sort(key=lambda x: x[3], reverse=True)
    return diffs[:top_n]

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path: str):
    rows, fieldnames = read_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        sys.exit(1)

    # Identify Performance- column
    perf_key = None
    for k in fieldnames:
        if k.strip().lower() == 'performance-':
            perf_key = k
            break
    if perf_key is None:
        print("Error: No 'Performance-' column found.")
        sys.exit(1)

    perf_values = [row[perf_key] for row in rows]
    n = len(perf_values)
    mean_perf = sum(perf_values) / n
    std_perf = math.sqrt(sum((x - mean_perf) ** 2 for x in perf_values) / n)
    q1 = percentile(perf_values, 25)
    q3 = percentile(perf_values, 75)

    # Identify poor and good rows
    poor_rows = [row for row in rows if row[perf_key] <= q1]
    good_rows = [row for row in rows if row[perf_key] >= q3]

    # Compute correlations
    corr = compute_correlations(rows, perf_key)

    # ------------------------------------------------------------------
    # APACHE-09
    # ------------------------------------------------------------------
    print("== APACHE-09 ==")
    if not poor_rows:
        print("No poorly performing configurations detected.")
    else:
        # Show up to first 3 poor configs
        for idx, row in enumerate(poor_rows[:3], 1):
            perf = row[perf_key]
            top_settings = get_top_settings_for_config(row, corr, top_n=3)
            print(f"Config #{idx} (Perf: {perf})")
            if top_settings:
                print("  Most associated settings:")
                for s in top_settings:
                    print(f"    {s}")
            else:
                print("  No strongly correlated settings found.")
        # Summary: list overall top correlations
        print("\nOverall strongest negative correlations:")
        neg_corr = [(k, v) for k, v in corr.items() if v < 0]
        neg_corr.sort(key=lambda x: x[1])
        for k, v in neg_corr[:5]:
            print(f"  {k}: {v:.3f}")

    # ------------------------------------------------------------------
    # APACHE-10
    # ------------------------------------------------------------------
    print("\n== APACHE-10 ==")
    if not poor_rows:
        print("No poorly performing configurations to analyze combinations.")
    else:
        top_pairs = find_top_pairs(poor_rows, good_rows, perf_key, top_n=3)
        if not top_pairs:
            print("No common pairs overrepresented in poor configs.")
        else:
            print(f"{'Pair':<25}{'Poor Count':>12}{'Good Count':>12}{'Diff':>10}")
            for pair, poor_cnt, good_cnt, diff in top_pairs:
                print(f"{pair:<25}{poor_cnt:>12}{good_cnt:>12}{diff:>10}")
            # Show overall counts
            print(f"\nTotal poor configs: {len(poor_rows)}")
            print(f"Total good configs: {len(good_rows)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <path_to_csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
