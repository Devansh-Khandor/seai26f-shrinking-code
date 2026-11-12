#!/usr/bin/env python3
"""
Apache configuration comparison.

Usage:
    python skill_03.py <path_to_Apache_AllMeasurements.csv>
"""

import csv
import sys
import math
import numpy as np


def read_csv(csv_path):
    """Read CSV and return rows as list of dicts with numeric conversion."""
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

    # Identify performance column
    perf_col = None
    for col in rows[0].keys():
        if col.lower().startswith("performance"):
            perf_col = col
            break
    if not perf_col:
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
                    r[k] = 0  # fallback for unexpected values

    return rows, perf_col


def group_stats(rows, perf_col, group_col):
    """Return mean, std, n for rows grouped by group_col values 0/1."""
    values0 = [r[perf_col] for r in rows if r[group_col] == 0]
    values1 = [r[perf_col] for r in rows if r[group_col] == 1]

    if not values0 or not values1:
        return None, None

    def stats(vals):
        arr = np.array(vals)
        return float(arr.mean()), float(arr.std(ddof=1)), len(arr)

    return stats(values0), stats(values1)


def diff_confidence_interval(mean1, std1, n1, mean2, std2, n2):
    """Return difference of means and 95% confidence interval."""
    diff = mean1 - mean2
    se = math.sqrt(std1**2 / n1 + std2**2 / n2)
    margin = 1.96 * se
    return diff, (diff - margin, diff + margin)


def print_apache_section(section_id, description, keep_alive_stats, access_log_stats):
    print(f"== {section_id} ==")
    print(description)

    # KeepAlive section
    if keep_alive_stats:
        (m0, s0, n0), (m1, s1, n1) = keep_alive_stats
        diff, (ci_low, ci_high) = diff_confidence_interval(m1, s1, n1, m0, s0, n0)
        signif = "YES" if ci_low > 0 or ci_high < 0 else "NO"
        print(f"KeepAlive=1: mean={m1:.2f}, std={s1:.2f}, n={n1}")
        print(f"KeepAlive=0: mean={m0:.2f}, std={s0:.2f}, n={n0}")
        print(f"Mean difference (1-0): {diff:.2f} (95% CI: [{ci_low:.2f}, {ci_high:.2f}])")
        print(f"Significant difference: {signif}")
    else:
        print("Insufficient data for KeepAlive comparison.")

    # AccessLog section
    if access_log_stats:
        (m0, s0, n0), (m1, s1, n1) = access_log_stats
        diff, (ci_low, ci_high) = diff_confidence_interval(m1, s1, n1, m0, s0, n0)
        signif = "YES" if ci_low > 0 or ci_high < 0 else "NO"
        print(f"accessLog=1: mean={m1:.2f}, std={s1:.2f}, n={n1}")
        print(f"accessLog=0: mean={m0:.2f}, std={s0:.2f}, n={n0}")
        print(f"Mean difference (1-0): {diff:.2f} (95% CI: [{ci_low:.2f}, {ci_high:.2f}])")
        print(f"Significant difference: {signif}")
    else:
        print("Insufficient data for accessLog comparison.")


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, perf_col = read_csv(csv_path)

    # Find column names (case-insensitive)
    def find_col(name):
        for col in rows[0].keys():
            if col.lower() == name.lower():
                return col
        return None

    keep_col = find_col("keepAlive")
    log_col = find_col("accessLog")

    if keep_col is None or log_col is None:
        print("Required columns (keepAlive, accessLog) not found.")
        sys.exit(1)

    # Compute statistics
    keep_stats = group_stats(rows, perf_col, keep_col)
    log_stats = group_stats(rows, perf_col, log_col)

    print_apache_section(
        "APACHE-05",
        "Comparison of configurations with KeepAlive enabled vs disabled:",
        keep_stats,
        log_stats,
    )
    print()  # blank line for separation
    # We reuse log_stats for APACHE-06
    print_apache_section(
        "APACHE-06",
        "Comparison of configurations with access logging enabled vs disabled:",
        keep_stats,
        log_stats,
    )


if __name__ == "__main__":
    main()
