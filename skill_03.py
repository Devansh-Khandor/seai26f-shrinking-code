#!/usr/bin/env python3
"""
Skill 03: Apache configuration impact analysis

Usage:
    python skill_03.py <csv path>

This script produces two new reports requested by the client:

  - APACHE-05  Compare KeepAlive enabled vs disabled configurations.
  - APACHE-06  Compare access logging enabled vs disabled configurations.

All calculations are performed with the Python standard library only.
"""

import csv
import sys
import math
from collections import defaultdict

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def read_csv(path):
    """Read CSV and return a list of rows as dicts of integers plus fieldnames."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: int(v) for k, v in r.items()})
    return rows, reader.fieldnames

def get_performance_column(fieldnames):
    """Return the name of the performance column."""
    # Prefer a column ending with '-' or named 'Performance' (case‑insensitive)
    for name in fieldnames:
        if name.lower() == 'performance' or name.endswith('-'):
            return name
    # Fallback: last column
    return fieldnames[-1]

def compute_stats(group):
    """Return (mean, std_dev, count) for a list of integer performance values."""
    n = len(group)
    if n == 0:
        return (None, None, 0)
    mean = sum(group) / n
    if n == 1:
        return (mean, 0.0, n)
    variance = sum((x - mean) ** 2 for x in group) / (n - 1)
    return (mean, math.sqrt(variance), n)

def t_statistic(mean1, mean2, std1, std2, n1, n2):
    """
    Compute a simple two‑sample t‑statistic (assuming equal variances).
    If either sample size is 1, returns None.
    """
    if n1 < 2 or n2 < 2:
        return None
    se = math.sqrt(std1 ** 2 / n1 + std2 ** 2 / n2)
    if se == 0:
        return None
    return (mean1 - mean2) / se

# --------------------------------------------------------------------------- #
# Reporting functions
# --------------------------------------------------------------------------- #

def report_keepalive(rows, perf_col):
    """Report the impact of KeepAlive setting on performance."""
    print("== APACHE-05 ==")

    group1 = [row[perf_col] for row in rows if row.get('keepAlive') == 1]
    group0 = [row[perf_col] for row in rows if row.get('keepAlive') == 0]

    mean1, std1, n1 = compute_stats(group1)
    mean0, std0, n0 = compute_stats(group0)

    if mean1 is None or mean0 is None:
        print("Insufficient data to compare KeepAlive settings.")
        return

    diff = mean1 - mean0
    t = t_statistic(mean1, mean0, std1, std0, n1, n0)

    print(f"Mean performance (KeepAlive=1): {mean1:.2f} (n={n1})")
    print(f"Mean performance (KeepAlive=0): {mean0:.2f} (n={n0})")
    print(f"Difference (1 - 0): {diff:.2f}")

    if t is not None:
        print(f"t‑statistic: {t:.2f}")
        significance = "Significant" if abs(t) > 2 else "Not significant"
        print(f"Interpretation: {significance} difference (|t|>2 threshold).")
    else:
        print("Cannot compute t‑statistic due to insufficient variance or sample size.")

def report_accesslog(rows, perf_col):
    """Report the impact of access logging setting on performance."""
    print("== APACHE-06 ==")

    group1 = [row[perf_col] for row in rows if row.get('accessLog') == 1]
    group0 = [row[perf_col] for row in rows if row.get('accessLog') == 0]

    mean1, std1, n1 = compute_stats(group1)
    mean0, std0, n0 = compute_stats(group0)

    if mean1 is None or mean0 is None:
        print("Insufficient data to compare access logging settings.")
        return

    diff = mean1 - mean0
    t = t_statistic(mean1, mean0, std1, std0, n1, n0)

    print(f"Mean performance (AccessLog=1): {mean1:.2f} (n={n1})")
    print(f"Mean performance (AccessLog=0): {mean0:.2f} (n={n0})")
    print(f"Difference (1 - 0): {diff:.2f}")

    if t is not None:
        print(f"t‑statistic: {t:.2f}")
        significance = "Significant" if abs(t) > 2 else "Not significant"
        print(f"Interpretation: {significance} difference (|t|>2 threshold).")
    else:
        print("Cannot compute t‑statistic due to insufficient variance or sample size.")

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)
    perf_col = get_performance_column(fieldnames)

    report_keepalive(rows, perf_col)
    report_accesslog(rows, perf_col)

if __name__ == "__main__":
    main()
