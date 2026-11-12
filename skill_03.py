#!/usr/bin/env python3
"""
Skill 03: NASA-05 and NASA-06 – Comparative analysis of effort and defects
Author: Your Name
"""

import sys
import csv
import math

# ----------------------------------------------------------------------
# Utility: read CSV and convert numeric values where possible
# ----------------------------------------------------------------------
def read_csv(path):
    """
    Reads a CSV file and returns:
        header: list of column names
        rows: list of dicts mapping column -> value (numeric where possible)
    """
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = []
        for r in reader:
            row = {}
            for k, v in r.items():
                v = v.strip()
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return header, rows

# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def group_stats(rows, group_col, target_col):
    """
    Compute mean, std, and count for each group value in group_col.
    Returns a dict: {group_value: (mean, std, n)}.
    """
    groups = {}
    for r in rows:
        if group_col not in r or target_col not in r:
            continue
        g = r[group_col]
        val = r[target_col]
        if not isinstance(val, (int, float)):
            continue
        groups.setdefault(g, []).append(val)

    stats = {}
    for g, vals in groups.items():
        n = len(vals)
        mean = sum(vals) / n
        if n > 1:
            std = math.sqrt(sum((x - mean) ** 2 for x in vals) / (n - 1))
        else:
            std = 0.0
        stats[g] = (mean, std, n)
    return stats

def pairwise_ci(stats, g1, g2, alpha=0.05):
    """
    Compute difference of means and 95% confidence interval between two groups.
    Returns (diff, lower, upper, significant).
    """
    mean1, std1, n1 = stats[g1]
    mean2, std2, n2 = stats[g2]
    diff = mean1 - mean2
    # Standard error of difference
    se = math.sqrt(std1 ** 2 / n1 + std2 ** 2 / n2)
    # z critical value for two-sided
    z = 1.96  # approximate for 95% CI
    lower = diff - z * se
    upper = diff + z * se
    significant = not (lower <= 0 <= upper)
    return diff, lower, upper, significant

# ----------------------------------------------------------------------
# NASA-05: Programmer capability vs effort
# ----------------------------------------------------------------------
def nasa_05(rows):
    """
    Compare projects with different programmer capability ratings
    and evaluate if their development-effort differences are statistically meaningful.
    """
    print("== NASA-05 ==")
    stats = group_stats(rows, 'pcap', 'Effort-')
    if not stats:
        print("No valid data for NASA-05.")
        print()
        return

    # Print group summaries
    for g in sorted(stats.keys()):
        mean, std, n = stats[g]
        print(f"Programmer capability: {g}, mean effort={mean:.2f}, std={std:.2f}, n={n}")

    # Pairwise comparisons
    groups = sorted(stats.keys())
    if len(groups) >= 2:
        print()
        print("Pairwise comparison of mean effort (95% CI):")
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1, g2 = groups[i], groups[j]
                diff, lower, upper, sig = pairwise_ci(stats, g1, g2)
                sig_str = "YES" if sig else "NO"
                print(f"{g1} vs {g2}: mean diff = {diff:.2f}, 95% CI = ({lower:.2f}, {upper:.2f}), significant? {sig_str}")

    print()  # blank line for readability

# ----------------------------------------------------------------------
# NASA-06: Complexity vs defects
# ----------------------------------------------------------------------
def nasa_06(rows):
    """
    Compare projects with different complexity ratings and determine
    if defect outcomes differ reliably or evidence is insufficient.
    """
    print("== NASA-06 ==")
    stats = group_stats(rows, 'cplx', 'Defects-')
    if not stats:
        print("No valid data for NASA-06.")
        print()
        return

    # Print group summaries
    for g in sorted(stats.keys()):
        mean, std, n = stats[g]
        print(f"Complexity rating: {g}, mean defects={mean:.2f}, std={std:.2f}, n={n}")

    # Pairwise comparisons
    groups = sorted(stats.keys())
    if len(groups) >= 2:
        print()
        print("Pairwise comparison of mean defects (95% CI):")
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1, g2 = groups[i], groups[j]
                diff, lower, upper, sig = pairwise_ci(stats, g1, g2)
                sig_str = "YES" if sig else "NO"
                print(f"{g1} vs {g2}: mean diff = {diff:.2f}, 95% CI = ({lower:.2f}, {upper:.2f}), significant? {sig_str}")

    print()  # blank line for readability

# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        header, rows = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        nasa_05(rows)
    except Exception as e:
        print(f"NASA-05 error: {e}", file=sys.stderr)

    try:
        nasa_06(rows)
    except Exception as e:
        print(f"NASA-06 error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
