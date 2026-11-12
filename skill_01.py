#!/usr/bin/env python3
"""
Skill 01: Basic Apache performance analysis

Usage:
    python skill_01.py <csv path>

The script reads the CSV file containing Apache server configuration
measurements and prints two reports requested by the client.

The CSV is assumed to contain only integer values and a header row.
The performance column is the one that is *not* a configuration option.
"""

import csv
import sys
import math
from collections import defaultdict

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def read_csv(path):
    """Read the CSV and return a list of rows as dicts with integer values."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Convert all values to int
            row = {k: int(v) for k, v in r.items()}
            rows.append(row)
    return rows, reader.fieldnames

def get_performance_column(fieldnames):
    """Return the name of the performance column (the one that is not a config)."""
    # Assume config columns are all except 'Performance' or something ending with '-'
    # We detect the column that does not match typical config names (all lowercase or containing digits)
    # For this dataset, the performance column ends with '-'
    for name in fieldnames:
        if name.lower() == 'performance' or name.endswith('-'):
            return name
    # fallback: last column
    return fieldnames[-1]

def median(values):
    """Return median of a list of numbers."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    return sorted_vals[mid]

def column_medians(rows, columns):
    """Return a dict mapping column to median value among rows."""
    med = {}
    for col in columns:
        med[col] = median([row[col] for row in rows])
    return med

# --------------------------------------------------------------------------- #
# Main reporting functions
# --------------------------------------------------------------------------- #

def report_apache_01(rows, perf_col):
    """Print summary of typical, best, worst performance values."""
    perf_vals = [row[perf_col] for row in rows]
    typical = median(perf_vals)
    best = min(perf_vals)
    worst = max(perf_vals)

    print("== APACHE-01 ==")
    print(f"Typical performance: {typical}")
    print(f"Best performance:    {best}")
    print(f"Worst performance:   {worst}")

def report_apache_02(rows, perf_col, config_cols):
    """Identify unusually favorable configurations and show distinguishing settings."""
    perf_vals = [row[perf_col] for row in rows]
    mean = sum(perf_vals) / len(perf_vals)
    std = math.sqrt(sum((x - mean) ** 2 for x in perf_vals) / len(perf_vals))

    threshold = mean - 2 * std  # arbitrary definition of unusually favorable
    favorable = [row for row in rows if row[perf_col] <= threshold]

    print("== APACHE-02 ==")
    if not favorable:
        print("No unusually favorable configurations found.")
        return

    # Compute median config for comparison
    medians = column_medians(favorable, config_cols)

    for i, row in enumerate(favorable, start=1):
        # Print the configuration as CSV row
        vals = [str(row[col]) for col in config_cols]
        print(f"Config {i}: " + ",".join(vals))

        # Find distinguishing settings
        diff = [col for col in config_cols if row[col] != medians[col]]
        if diff:
            diff_str = ", ".join(f"{col}={row[col]}" for col in diff)
            print(f"  Distinguishing settings: {diff_str}")
        else:
            print("  No distinguishing settings.")

# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)
    perf_col = get_performance_column(fieldnames)

    # All columns except performance are configuration options
    config_cols = [col for col in fieldnames if col != perf_col]

    report_apache_01(rows, perf_col)
    report_apache_02(rows, perf_col, config_cols)

if __name__ == "__main__":
    main()
