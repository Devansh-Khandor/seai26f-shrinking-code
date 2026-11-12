#!/usr/bin/env python3
"""
skill_03.py

A self‑contained script that extends the NASA data analysis to provide
comparative insights for future planning.

- NASA-05: Compare projects with different programmer capability ratings
           and evaluate whether their development‑effort differences are
           statistically meaningful.
- NASA-06: Compare projects with different product‑complexity ratings
           and assess whether their defect outcomes differ reliably
           or whether the evidence is insufficient.

Usage:
    python skill_03.py path/to/nasa93dem.csv
"""

import csv
import sys
from pathlib import Path
import math
import statistics

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def _read_csv(path: Path):
    """
    Reads the CSV file and returns a list of dicts.  Each value is
    converted to float if it can be parsed; otherwise the original
    string is kept unchanged.
    """
    data = []
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {}
            for k, v in row.items():
                if v is None:
                    cleaned[k] = None
                    continue
                val = v.strip()
                try:
                    cleaned[k] = float(val)
                except ValueError:
                    cleaned[k] = val
            data.append(cleaned)
    return data

def _group_numeric(data, group_key: str, value_key: str):
    """
    Groups numeric values of *value_key* by the categorical value of
    *group_key*.  Non‑numeric or missing values are ignored.
    """
    groups = {}
    for row in data:
        group = row.get(group_key)
        value = row.get(value_key)
        if group is None or value is None:
            continue
        if not isinstance(value, (int, float)):
            continue
        groups.setdefault(group, []).append(value)
    return groups

def _stats(values):
    """
    Return count, mean, stddev, lower CI, upper CI for a list of numbers.
    Uses a z‑score of 2.0 (approx. 95% confidence) for the interval.
    If fewer than two samples, stddev and CI are None.
    """
    n = len(values)
    if n == 0:
        return n, None, None, None, None
    mean = statistics.mean(values)
    if n < 2:
        return n, mean, None, None, None
    stddev = statistics.stdev(values)
    se = stddev / math.sqrt(n)
    margin = 2.0 * se  # approximate 95% CI
    return n, mean, stddev, mean - margin, mean + margin

def _print_group_stats(title: str, group_stats: dict):
    """
    Print a table of group statistics and a brief conclusion.
    """
    print(title)
    # Sort groups for consistent output
    for group, stats in sorted(group_stats.items(), key=lambda x: str(x[0])):
        n, mean, std, low, high = stats
        if n == 0:
            continue
        print(f"  Group '{group}': n={n}, mean={mean:.2f}", end='')
        if std is not None:
            print(f", std={std:.2f}")
            print(f"    95% CI: [{low:.2f}, {high:.2f}]")
        else:
            print(", std=NA (only one sample)")

    # Determine if any pair of groups have non‑overlapping CIs
    meaningful = False
    groups = sorted(group_stats.items(), key=lambda x: str(x[0]))
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            _, (n1, mean1, _, low1, high1) = groups[i]
            _, (n2, mean2, _, low2, high2) = groups[j]
            if None in (low1, high1, low2, high2):
                continue  # cannot compare
            if high1 < low2 or high2 < low1:
                meaningful = True
                break
        if meaningful:
            break

    if meaningful:
        print("\nConclusion: The differences between groups are statistically meaningful "
              "(confidence intervals do not overlap).")
    else:
        print("\nConclusion: There is insufficient evidence that the groups differ "
              "in a statistically meaningful way. More data are recommended.")

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    data = _read_csv(csv_path)

    # ----------------------------------------------------------------------- #
    # NASA-05: Programmer capability vs. Effort
    # ----------------------------------------------------------------------- #
    print("== NASA-05 ==")
    # Group by 'acap' (programmer capability) and analyze 'Effort-'
    cap_groups = _group_numeric(data, "acap", "Effort-")
    cap_stats = {k: _stats(v) for k, v in cap_groups.items()}
    if not cap_stats:
        print("No data available for programmer capability analysis.")
    else:
        _print_group_stats("Programmer Capability Comparison (Effort-):", cap_stats)

    # ----------------------------------------------------------------------- #
    # NASA-06: Product complexity vs. Defects
    # ----------------------------------------------------------------------- #
    print("\n== NASA-06 ==")
    # Group by 'pcon' (product complexity) and analyze 'Defects-'
    con_groups = _group_numeric(data, "pcon", "Defects-")
    con_stats = {k: _stats(v) for k, v in con_groups.items()}
    if not con_stats:
        print("No data available for complexity vs. defect analysis.")
    else:
        _print_group_stats("Product Complexity Comparison (Defects-):", con_stats)

if __name__ == "__main__":
    main()
