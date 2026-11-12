#!/usr/bin/env python3
"""
skill_03.py

Usage:
    python skill_03.py <path_to_Apache_AllMeasurements.csv>
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
def describe_group(values):
    """Return count, mean, median, std for a list of ints."""
    n = len(values)
    if n == 0:
        return n, None, None, None
    sorted_vals = sorted(values)
    mean = sum(values) / n
    median = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / n)
    return n, mean, median, std

def cohen_d(mean1, mean2, std1, std2):
    """Compute Cohen's d effect size."""
    if std1 is None or std2 is None:
        return None
    pooled = math.sqrt((std1 ** 2 + std2 ** 2) / 2)
    if pooled == 0:
        return 0.0
    return (mean1 - mean2) / pooled

# ----------------------------------------------------------------------
# APACHE-05 & APACHE-06
# ----------------------------------------------------------------------
def analyze_setting(rows, setting_key, setting_name):
    """Analyze performance differences between setting 0 and 1."""
    perf_key = next((k for k in rows[0].keys() if k.strip().lower() == 'performance-'), None)
    if perf_key is None:
        raise ValueError("No 'Performance-' column found.")

    group0 = [row[perf_key] for row in rows if row.get(setting_key) == 0]
    group1 = [row[perf_key] for row in rows if row.get(setting_key) == 1]

    n0, mean0, med0, std0 = describe_group(group0)
    n1, mean1, med1, std1 = describe_group(group1)

    diff_mean = (mean1 - mean0) if mean0 is not None and mean1 is not None else None
    d = cohen_d(mean1, mean0, std1, std0)

    # Decision rule: |d| > 0.5 => meaningful
    conclusion = ""
    if d is None:
        conclusion = "Insufficient data to determine significance."
    elif abs(d) > 0.5:
        conclusion = f"Observed difference is large enough ({d:.2f}) to guide decisions."
    else:
        conclusion = f"Observed difference ({d:.2f}) is not large enough; evidence is inconclusive."

    return {
        'setting': setting_name,
        'group0': {'count': n0, 'mean': mean0, 'median': med0, 'std': std0},
        'group1': {'count': n1, 'mean': mean1, 'median': med1, 'std': std1},
        'diff_mean': diff_mean,
        'cohen_d': d,
        'conclusion': conclusion
    }

def main(csv_path):
    rows, fieldnames = read_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        sys.exit(1)

    # APACHE-05: KeepAlive comparison
    print("== APACHE-05 ==")
    ka_result = analyze_setting(rows, 'keepAlive', 'KeepAlive')
    print(f"Configurations with KeepAlive disabled: {ka_result['group0']['count']}")
    print(f"Configurations with KeepAlive enabled:  {ka_result['group1']['count']}")
    if ka_result['group0']['mean'] is not None:
        print(f"Mean performance (disabled): {ka_result['group0']['mean']:.2f}")
        print(f"Mean performance (enabled):  {ka_result['group1']['mean']:.2f}")
        print(f"Mean difference (enabled - disabled): {ka_result['diff_mean']:.2f}")
        print(f"Cohen's d effect size: {ka_result['cohen_d']:.2f}")
    print(ka_result['conclusion'])
    print()

    # APACHE-06: accessLog comparison
    print("== APACHE-06 ==")
    al_result = analyze_setting(rows, 'accessLog', 'AccessLog')
    print(f"Configurations with AccessLog disabled: {al_result['group0']['count']}")
    print(f"Configurations with AccessLog enabled:  {al_result['group1']['count']}")
    if al_result['group0']['mean'] is not None:
        print(f"Mean performance (disabled): {al_result['group0']['mean']:.2f}")
        print(f"Mean performance (enabled):  {al_result['group1']['mean']:.2f}")
        print(f"Mean difference (enabled - disabled): {al_result['diff_mean']:.2f}")
        print(f"Cohen's d effect size: {al_result['cohen_d']:.2f}")
    print(al_result['conclusion'])
    print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <path_to_csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
