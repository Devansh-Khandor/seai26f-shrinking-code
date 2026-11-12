#!/usr/bin/env python3
"""
Skill 05: Apache configuration blame analysis

Usage:
    python skill_05.py <csv path>

This script implements two new reports requested by the client:

  - APACHE-09  Identify which individual settings are most strongly
               associated with unfavorable performance.

  - APACHE-10  Find frequent combinations of settings that appear in
               poorly performing configurations and compare them
               to the overall average performance.

All calculations are performed with the Python standard library only.
"""

import csv
import sys
import math
from collections import defaultdict, Counter

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def read_csv(path):
    """Read CSV and return list of rows as dicts of ints plus fieldnames."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: int(v) for k, v in r.items()})
    return rows, reader.fieldnames


def get_performance_column(fieldnames):
    """Return the name of the performance column (last column or ending in '-')"""
    for name in fieldnames:
        if name.lower() == 'performance' or name.endswith('-'):
            return name
    return fieldnames[-1]


def mean_std(values):
    """Return (mean, std_dev) for a list of numbers (sample std)."""
    n = len(values)
    if n == 0:
        return (None, None)
    mean = sum(values) / n
    if n == 1:
        return (mean, 0.0)
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return (mean, math.sqrt(var))


# --------------------------------------------------------------------------- #
# Report APACHE-09
# --------------------------------------------------------------------------- #

def report_apache_09(rows, perf_col, config_cols):
    """Identify which settings are most strongly associated with poor performance."""
    print("== APACHE-09 ==")

    # Compute mean performance for each setting value
    diff_stats = {}
    for col in config_cols:
        vals1 = [row[perf_col] for row in rows if row[col] == 1]
        vals0 = [row[perf_col] for row in rows if row[col] == 0]
        if not vals1 or not vals0:
            continue  # skip if one of the groups is empty

        m1, _ = mean_std(vals1)
        m0, _ = mean_std(vals0)

        # Positive diff => value 1 is associated with higher (worse) perf
        diff = m1 - m0
        diff_stats[col] = diff

    if not diff_stats:
        print("Insufficient data to compute associations.")
        return

    # Sort by absolute diff descending; keep top 3
    top = sorted(diff_stats.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    for col, diff in top:
        direction = "enabled" if diff > 0 else "disabled"
        print(f"{col} ({direction}) mean perf diff: {diff:.2f}")

# --------------------------------------------------------------------------- #
# Report APACHE-10
# --------------------------------------------------------------------------- #

def report_apache_10(rows, perf_col, config_cols):
    """Find frequent combinations in poor configurations and compare to overall average."""
    print("== APACHE-10 ==")

    all_perf = [row[perf_col] for row in rows]
    if not all_perf:
        print("No performance data available.")
        return

    overall_mean, _ = mean_std(all_perf)

    # Define poorly performing configurations as those >= mean + std
    mean_perf, std_perf = mean_std(all_perf)
    threshold = mean_perf + std_perf

    poor_rows = [row for row in rows if row[perf_col] >= threshold]
    if not poor_rows:
        print("No poor configurations detected by threshold.")
        return

    # Count combination frequencies
    combo_counter = Counter()
    combo_perf_sums = defaultdict(float)
    combo_counts = defaultdict(int)

    for row in poor_rows:
        combo = tuple(row[col] for col in config_cols)
        combo_counter[combo] += 1
        combo_perf_sums[combo] += row[perf_col]
        combo_counts[combo] += 1

    # Take top 5 most frequent combinations
    top_combos = combo_counter.most_common(5)

    if not top_combos:
        print("No frequent combinations found among poor configurations.")
        return

    for combo, freq in top_combos:
        combo_mean_perf = combo_perf_sums[combo] / combo_counts[combo]
        diff_from_overall = combo_mean_perf - overall_mean
        config_str = ",".join(f"{col}={val}" for col, val in zip(config_cols, combo))
        print(f"Config: {config_str} | freq={freq} | mean_perf={combo_mean_perf:.2f} "
              f"| diff_from_overall={diff_from_overall:+.2f}")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)
    perf_col = get_performance_column(fieldnames)
    config_cols = [c for c in fieldnames if c != perf_col]

    report_apache_09(rows, perf_col, config_cols)
    report_apache_10(rows, perf_col, config_cols)


if __name__ == "__main__":
    main()
