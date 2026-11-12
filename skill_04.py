#!/usr/bin/env python3
"""
Skill 04: Incremental Apache performance monitoring and summary

Usage:
    python skill_04.py <csv path>

This script implements two new reports requested by the client:

  - APACHE-07  Detect unusually poor configurations as new benchmark data arrives.
  - APACHE-08  Keep performance summaries and recommended configuration shortlist up to date.

The script reads the CSV file, processes each row in the order it appears
and prints the results for each report.

The performance column is assumed to be the last column or the one that ends
with a dash ('-'), matching the conventions used in earlier skills.
"""

import csv
import sys
import math

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
    """Return the name of the performance column."""
    for name in fieldnames:
        if name.lower() == 'performance' or name.endswith('-'):
            return name
    # Fallback to last column
    return fieldnames[-1]


def mean_std(values):
    """Return (mean, std_dev) for a list of numbers. std_dev is sample std."""
    n = len(values)
    if n == 0:
        return (None, None)
    mean = sum(values) / n
    if n == 1:
        return (mean, 0.0)
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return (mean, math.sqrt(var))


def median(values):
    """Return median of a list of numbers."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    return sorted_vals[mid]


# --------------------------------------------------------------------------- #
# Reporting functions
# --------------------------------------------------------------------------- #

def report_apache_07(rows, perf_col, config_cols):
    """Detect unusually poor configurations as data arrives."""
    print("== APACHE-07 ==")
    prev_perf = []
    alerts = []

    for idx, row in enumerate(rows, start=1):
        curr_val = row[perf_col]
        if prev_perf:
            mean, std = mean_std(prev_perf)
            threshold = mean + 2 * std
            if curr_val > threshold:
                # Build alert message
                config_vals = ",".join(str(row[col]) for col in config_cols)
                alerts.append(
                    f"Alert {idx}: config={config_vals} | perf={curr_val} "
                    f"| mean(prev)={mean:.2f} std(prev)={std:.2f} threshold={threshold:.2f}"
                )
        prev_perf.append(curr_val)

    if alerts:
        for a in alerts:
            print(a)
    else:
        print("No unusually poor configurations found.")


def report_apache_08(rows, perf_col, config_cols):
    """Print performance summaries and recommended configuration shortlist."""
    print("== APACHE-08 ==")

    perf_vals = [row[perf_col] for row in rows]
    if not perf_vals:
        print("No performance data available.")
        return

    typical = median(perf_vals)
    best = min(perf_vals)
    worst = max(perf_vals)

    print(f"Typical performance (median): {typical:.2f}")
    print(f"Best performance (lowest value): {best:.2f}")
    print(f"Worst performance (highest value): {worst:.2f}")

    # Recommended configuration(s): those with best performance
    recommended = [row for row in rows if row[perf_col] == best]
    print(f"Recommended configuration(s) ({len(recommended)} found):")
    for i, row in enumerate(recommended, start=1):
        config_vals = ",".join(str(row[col]) for col in config_cols)
        print(f"  {i}. config={config_vals} | perf={row[perf_col]}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)
    perf_col = get_performance_column(fieldnames)
    config_cols = [c for c in fieldnames if c != perf_col]

    report_apache_07(rows, perf_col, config_cols)
    report_apache_08(rows, perf_col, config_cols)


if __name__ == "__main__":
    main()
