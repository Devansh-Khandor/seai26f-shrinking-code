#!/usr/bin/env python3
"""
skill_03.py

Implements NASA-05 and NASA-06 analyses on the NASA software project dataset.

Usage:
    python skill_03.py <csv_path>
"""

import sys
import csv
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------

def read_csv(csv_path: Path):
    """Read CSV and return a list of rows (dict)."""
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def parse_value(val: str):
    """Convert a string to float or return np.nan."""
    if val.strip() == '':
        return np.nan
    try:
        return float(val.replace(',', ''))
    except ValueError:
        return np.nan


def group_stats(rows, group_col, value_col):
    """
    Group rows by `group_col` and compute mean, std, count for the numeric `value_col`.
    Returns a dict: {group_value: (mean, std, count)}
    """
    groups = {}
    for r in rows:
        key = r.get(group_col, '').strip()
        if key == '':
            continue  # skip rows without group info
        val = parse_value(r.get(value_col, ''))
        if np.isnan(val):
            continue  # skip rows with missing target
        groups.setdefault(key, []).append(val)

    stats = {}
    for key, values in groups.items():
        arr = np.array(values, dtype=float)
        stats[key] = (np.mean(arr), np.std(arr, ddof=1), len(arr))
    return stats


def extreme_pair(stats):
    """
    Find pair of groups with the largest absolute difference in mean.
    Returns (group1, mean1, group2, mean2, diff, effect_size)
    If fewer than 2 groups, returns None.
    """
    if len(stats) < 2:
        return None

    # Prepare sorted list by mean
    sorted_groups = sorted(stats.items(), key=lambda kv: kv[1][0])
    # Find min and max
    min_key, (min_mean, min_std, min_n) = sorted_groups[0]
    max_key, (max_mean, max_std, max_n) = sorted_groups[-1]
    diff = max_mean - min_mean

    # Compute pooled std
    pooled_std = np.sqrt(
        ((min_n - 1) * min_std ** 2 + (max_n - 1) * max_std ** 2) / (min_n + max_n - 2)
    )
    effect_size = diff / pooled_std if pooled_std > 0 else np.nan

    return min_key, min_mean, max_key, max_mean, diff, effect_size


def print_nasa_05(rows):
    print("== NASA-05 ==")

    stats = group_stats(rows, 'pcap', 'Effort-')
    if not stats:
        print("No programmer capability groups with valid effort data.")
        print()
        return

    # Header
    print("Programmer capability groups and mean effort (person‑months):")
    for key, (mean, std, n) in sorted(stats.items()):
        print(f"  {key}: mean={mean:.2f} ({n} projects)")

    pair = extreme_pair(stats)
    if pair is None:
        print("Insufficient groups to compare.")
        print()
        return

    min_k, min_m, max_k, max_m, diff, d = pair
    print(f"\nHighest mean effort: {max_k} ({max_m:.2f})")
    print(f"Lowest  mean effort: {min_k} ({min_m:.2f})")
    print(f"Difference: {diff:.2f} person‑months")
    print(f"Cohen's d effect size: {d:.2f}")
    print()


def print_nasa_06(rows):
    print("== NASA-06 ==")

    stats = group_stats(rows, 'cplx', 'Defects-')
    if not stats:
        print("No complexity groups with valid defect data.")
        print()
        return

    # Header
    print("Complexity groups and mean defects:")
    for key, (mean, std, n) in sorted(stats.items()):
        print(f"  {key}: mean={mean:.2f} ({n} projects)")

    pair = extreme_pair(stats)
    if pair is None:
        print("Insufficient groups to compare.")
        print()
        return

    min_k, min_m, max_k, max_m, diff, d = pair
    print(f"\nHighest mean defects: {max_k} ({max_m:.2f})")
    print(f"Lowest  mean defects: {min_k} ({min_m:.2f})")
    print(f"Difference: {diff:.2f} defects")
    print(f"Cohen's d effect size: {d:.2f}")
    print()


# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: {csv_path} does not exist or is not a file.")
        sys.exit(1)

    rows, _ = read_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        sys.exit(1)

    print_nasa_05(rows)
    print_nasa_06(rows)


if __name__ == "__main__":
    main()
