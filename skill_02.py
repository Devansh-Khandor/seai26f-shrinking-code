#!/usr/bin/env python3
"""
Skill 02: Effort and defects estimation for new projects.

Usage:
    python skill_02.py <path_to_nasa93dem.csv>
"""

import csv
import sys
import math
from collections import defaultdict

def read_data(csv_path):
    """
    Read CSV and return list of project dictionaries containing numeric fields:
    - Kloc+
    - Effort-
    - Defects-
    Projects with non-positive Kloc+ are ignored.
    """
    projects = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                k = float(row["Kloc+"].strip())
                if k <= 0:
                    continue
                eff = float(row["Effort-"].strip())
                defc = float(row["Defects-"].strip())
                projects.append({"Kloc": k, "Effort": eff, "Defects": defc})
            except (ValueError, KeyError):
                continue
    return projects

def compute_stats(values):
    """
    Given a list of numeric values, compute mean, standard deviation and count.
    """
    n = len(values)
    if n == 0:
        return None, None, 0
    mean = sum(values) / n
    # population standard deviation
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / n)
    return mean, std, n

def confidence_interval(std, n, confidence=0.95):
    """
    Compute the 1-sided half-width of the confidence interval for the mean
    using the normal distribution (z-score). For small n a t-distribution
    would be more appropriate, but the dataset is large enough for this.
    """
    z = 1.96  # approx for 95%
    return z * std / math.sqrt(n)

def format_number(val, decimals=2):
    if math.isnan(val) or math.isinf(val):
        return "N/A"
    return f"{val:.{decimals}f}".rstrip('0').rstrip('.')

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    projects = read_data(csv_path)

    efforts = [p["Effort"] for p in projects]
    defects = [p["Defects"] for p in projects]

    # NASA-03: Effort estimate
    mean_effort, std_effort, n_effort = compute_stats(efforts)
    ci_effort = confidence_interval(std_effort, n_effort) if n_effort > 0 else None

    print("== NASA-03 ==")
    if n_effort > 0:
        print(f"Estimated effort: {format_number(mean_effort)} man-months "
              f"(± {format_number(ci_effort)})")
        print(f"Number of projects used: {n_effort}")
        print(f"Standard deviation: {format_number(std_effort)}")
    else:
        print("No effort data available.")

    # NASA-04: Defects estimate
    mean_def, std_def, n_def = compute_stats(defects)
    ci_def = confidence_interval(std_def, n_def) if n_def > 0 else None

    print("\n== NASA-04 ==")
    if n_def > 0:
        print(f"Estimated defects: {format_number(mean_def)} "
              f"(± {format_number(ci_def)})")
        print(f"Number of projects used: {n_def}")
        print(f"Standard deviation: {format_number(std_def)}")
    else:
        print("No defects data available.")

if __name__ == "__main__":
    main()
