#!/usr/bin/env python3
"""
skill_01.py
-----------

This script reads an Apache performance CSV file and answers two
requests:

APACHE-01: Summarise typical, best and worst performance values.
APACHE-02: Identify configurations that have unusually favorable
           performance values and list the settings that distinguish
           them from the rest.

Usage:
    python skill_01.py <path_to_Apache_AllMeasurements.csv>
"""

import sys
import csv
import math
from collections import defaultdict

def read_csv(csv_path):
    """
    Reads the CSV and returns:
        feature_names: list of configuration column names
        X: list of lists of feature values (ints)
        y: list of performance values (floats)
    """
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header:
            raise ValueError("Empty CSV header")

        # The performance column is the last column
        perf_col = header[-1]
        feature_names = header[:-1]

        X = []
        y = []

        for row in reader:
            if len(row) != len(header):
                continue  # skip malformed rows
            # Convert feature values to int (they are 0/1 in the sample)
            try:
                feat_row = [int(float(v)) for v in row[:-1]]
            except ValueError:
                # skip rows with non-numeric feature values
                continue
            try:
                perf_val = float(row[-1])
            except ValueError:
                continue
            X.append(feat_row)
            y.append(perf_val)

    return feature_names, X, y

def apache_01_summary(y):
    """
    Compute typical (median), best (min) and worst (max) performance.
    Returns a dict.
    """
    n = len(y)
    if n == 0:
        return {"median": None, "min": None, "max": None}

    sorted_y = sorted(y)
    median = sorted_y[n//2] if n % 2 == 1 else (sorted_y[n//2 - 1] + sorted_y[n//2]) / 2
    best = sorted_y[0]
    worst = sorted_y[-1]
    return {"median": median, "min": best, "max": worst}

def compute_std(values):
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)

def apache_02_identify(feature_names, X, y):
    """
    Identify unusually favorable configurations.
    Criteria: performance <= median - 0.5 * std.
    Returns:
        fav_indices: list of indices of favorable configs
        group_means: dict of feature_name -> mean value in favorable group
        overall_means: dict of feature_name -> mean value overall
    """
    n = len(y)
    if n == 0:
        return [], {}, {}

    median = sorted(y)[n//2] if n % 2 == 1 else (sorted(y)[n//2 - 1] + sorted(y)[n//2]) / 2
    std = compute_std(y)
    threshold = median - 0.5 * std

    fav_indices = [i for i, val in enumerate(y) if val <= threshold]

    # If no config qualifies, pick top 2 best performances
    if not fav_indices:
        sorted_idx = sorted(range(n), key=lambda i: y[i])
        fav_indices = sorted_idx[:max(1, n//5)]  # at least one, or 20%

    # Compute means
    def mean(values):
        return sum(values) / len(values) if values else 0.0

    group_means = {}
    overall_means = {}

    # Transpose X to iterate by feature
    for j, fname in enumerate(feature_names):
        overall_vals = [row[j] for row in X]
        overall_means[fname] = mean(overall_vals)
        group_vals = [X[i][j] for i in fav_indices]
        group_means[fname] = mean(group_vals)

    return fav_indices, group_means, overall_means

def print_apache_01(summary):
    print("== APACHE-01 ==")
    if summary["median"] is None:
        print("No data available.")
        return
    print(f"Typical performance (median): {summary['median']}")
    print(f"Best performance (min): {summary['min']}")
    print(f"Worst performance (max): {summary['max']}")

def print_apache_02(feature_names, fav_indices, group_means, overall_means):
    print("== APACHE-02 ==")
    print(f"Number of unusually favorable configurations: {len(fav_indices)}")

    # Find features with a significant difference (>0.3) between group and overall
    threshold = 0.3
    distinguishing = []
    for fname in feature_names:
        diff = group_means[fname] - overall_means[fname]
        if abs(diff) >= threshold:
            distinguishing.append((fname, diff))

    if distinguishing:
        print(f"Significant distinguishing settings (diff > {threshold}):")
        for fname, diff in sorted(distinguishing, key=lambda x: -abs(x[1])):
            sign = '+' if diff >= 0 else '-'
            print(f" - {fname}: {sign}{abs(diff):.2f}")
    else:
        print("No settings differ significantly from overall average.")

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        feature_names, X, y = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # APACHE-01
    summary = apache_01_summary(y)
    print_apache_01(summary)

    # APACHE-02
    fav_indices, group_means, overall_means = apache_02_identify(feature_names, X, y)
    print_apache_02(feature_names, fav_indices, group_means, overall_means)

if __name__ == "__main__":
    main()
