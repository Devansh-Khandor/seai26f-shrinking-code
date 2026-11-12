#!/usr/bin/env python3
"""
Skill 01 – Apache performance analysis

This script reads the Apache_AllMeasurements.csv file, produces a summary of
performance statistics (typical, best, worst) and identifies
configurations that show unusually favorable performance values.
"""

import sys
import csv
import argparse
import math
from collections import Counter, defaultdict
from statistics import mean, median
import numpy as np

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------


def read_csv(path: str):
    """
    Reads a CSV file and returns:
        - header: list of column names
        - rows: list of dicts where every value is converted to int
    """
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = []
        for row in reader:
            int_row = {}
            for key, value in row.items():
                # Convert to int; if conversion fails keep original
                try:
                    int_row[key] = int(value)
                except ValueError:
                    int_row[key] = value
            rows.append(int_row)
    return header, rows


def find_performance_column(header):
    """
    Detects the column that represents performance.
    The column name starts with 'performance' (case‑insensitive).
    """
    for col in header:
        if col.lower().startswith('performance'):
            return col
    raise KeyError("No performance column found in header.")


def compute_stats(performance_values):
    """
    Given a list of performance values returns a dict of:
        - mean, median, std, best (min), worst (max)
    """
    stats = {}
    stats['mean'] = mean(performance_values)
    stats['median'] = median(performance_values)
    stats['std'] = np.std(performance_values, ddof=0)
    stats['best'] = min(performance_values)      # lower is better
    stats['worst'] = max(performance_values)
    return stats


def find_favorable_configs(rows, perf_col, threshold):
    """
    Returns a list of rows that are considered 'favorable'
    (performance <= threshold).
    """
    return [row for row in rows if row[perf_col] <= threshold]


def column_modes(rows, cols):
    """
    Compute the mode (most common value) for each column among rows.
    Returns a dict: col_name -> mode_value
    """
    modes = {}
    for col in cols:
        counter = Counter(row[col] for row in rows)
        # Resolve ties by choosing the smallest value
        mode_value = min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        modes[col] = mode_value
    return modes


def distinguish_settings(overall_modes, favorable_modes):
    """
    Determine which settings differ between overall modes and favorable modes.
    Returns a dict of column -> (fav_mode, overall_mode)
    """
    diff = {}
    for col, fav_mode in favorable_modes.items():
        overall_mode = overall_modes[col]
        if fav_mode != overall_mode:
            diff[col] = (fav_mode, overall_mode)
    return diff

# ----------------------------------------------------------------------
# Main skill logic
# ----------------------------------------------------------------------


def skill_apache_01(header, rows):
    """
    Implements request APACHE-01: summary of performance.
    """
    perf_col = find_performance_column(header)
    perf_vals = [row[perf_col] for row in rows]

    stats = compute_stats(perf_vals)

    print("== APACHE-01 ==")
    print(f"Typical performance (median): {stats['median']:.0f}")
    print(f"Best performance (min): {stats['best']:.0f}")
    print(f"Worst performance (max): {stats['worst']:.0f}")


def skill_apache_02(header, rows):
    """
    Implements request APACHE-02: identify favorable configurations.
    """
    perf_col = find_performance_column(header)
    perf_vals = [row[perf_col] for row in rows]
    # Define threshold as the 10th percentile (i.e., lower 10% are favorable)
    threshold = np.percentile(perf_vals, 10)

    favorable_rows = find_favorable_configs(rows, perf_col, threshold)

    total_configs = len(rows)
    fav_configs = len(favorable_rows)

    # Compute modes
    setting_cols = [col for col in header if col != perf_col]
    overall_modes = column_modes(rows, setting_cols)
    favorable_modes = column_modes(favorable_rows, setting_cols)

    distinguishing = distinguish_settings(overall_modes, favorable_modes)

    print("== APACHE-02 ==")
    print(f"Favorable configurations: {fav_configs} out of {total_configs}")
    if distinguishing:
        print("Distinguishing settings:")
        for col, (fav_mode, overall_mode) in distinguishing.items():
            print(f"  - {col}: favorable={fav_mode} (overall mode={overall_mode})")
    else:
        print("No distinguishing settings found.")


def main(csv_path: str):
    header, rows = read_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        return

    skill_apache_01(header, rows)
    print()  # Blank line between skills
    skill_apache_02(header, rows)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apache performance analysis skill."
    )
    parser.add_argument(
        "csv_path",
        help="Path to Apache_AllMeasurements.csv",
    )
    args = parser.parse_args()

    try:
        main(args.csv_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
