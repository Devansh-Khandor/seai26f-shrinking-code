#!/usr/bin/env python3
"""
Skill 03: Compare effort and defects across programmer capability and complexity ratings.

Usage:
    python skill_03.py <csv_path>
"""

import csv
import math
import sys
from collections import defaultdict

# ---------- Helper functions ----------

def parse_float(val: str):
    """Parse a string to float, return None on failure."""
    try:
        return float(val.strip())
    except (AttributeError, ValueError):
        return None

def mean(values):
    return sum(values) / len(values) if values else 0.0

def stddev(values, mean_val):
    n = len(values)
    if n < 2:
        return 0.0
    return math.sqrt(sum((x - mean_val) ** 2 for x in values) / (n - 1))

# ---------- Statistical comparison utilities ----------

def diff_confidence_interval(mean1, sd1, n1, mean2, sd2, n2, z=1.96):
    """Return (difference, lower, upper) for 95% CI of difference between two means."""
    diff = mean1 - mean2
    se = math.sqrt((sd1 ** 2) / n1 + (sd2 ** 2) / n2)
    lower = diff - z * se
    upper = diff + z * se
    return diff, lower, upper

def is_significant(ci_lower, ci_upper):
    """Return True if CI does not include zero."""
    return ci_lower > 0 or ci_upper < 0

# ---------- Analysis functions ----------

def group_by_category(rows, category_col, numeric_col):
    """Group rows by category column and compute stats for the numeric column."""
    groups = defaultdict(list)
    for r in rows:
        key = r.get(category_col, "Unknown")
        val = r.get(numeric_col)
        if val is not None:
            groups[key].append(val)

    stats = {}
    for key, values in groups.items():
        n = len(values)
        m = mean(values)
        s = stddev(values, m)
        stats[key] = {'n': n, 'mean': m, 'sd': s}
    return stats

def compare_groups(stats):
    """Return list of comparison dicts for all unique pairs."""
    keys = sorted(stats.keys())
    comparisons = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = keys[i]
            b = keys[j]
            s1 = stats[a]
            s2 = stats[b]
            diff, low, up = diff_confidence_interval(
                s1['mean'], s1['sd'], s1['n'],
                s2['mean'], s2['sd'], s2['n']
            )
            comparisons.append({
                'group1': a,
                'group2': b,
                'diff': diff,
                'low': low,
                'up': up,
                'significant': is_significant(low, up),
                'n1': s1['n'],
                'n2': s2['n'],
                'mean1': s1['mean'],
                'mean2': s2['mean']
            })
    return comparisons

# ---------- Main entry point ----------

def main(csv_path):
    # Read CSV and collect relevant rows
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Clean numeric fields
            row['Effort-'] = parse_float(row.get('Effort-'))
            row['Defects-'] = parse_float(row.get('Defects-'))
            # Keep raw row for grouping
            rows.append(row)

    # Remove rows missing numeric values
    rows = [r for r in rows if r['Effort-'] is not None and r['Defects-'] is not None]

    if not rows:
        print("No valid data found in the CSV.")
        return

    # ====================== NASA-05 ======================
    # Compare effort by programmer capability ('pcap')
    pcap_stats = group_by_category(rows, 'pcap', 'Effort-')
    pcap_comparisons = compare_groups(pcap_stats)

    print("== NASA-05 ==")
    if pcap_stats:
        print("Programmer capability groups (n, mean Effort-):")
        for key, s in sorted(pcap_stats.items()):
            print(f"  {key:>4}: n={s['n']:>3}, mean Effort-={s['mean']:.2f}")
        print("\nPairwise comparison of Effort- between pcap groups (95% CI of difference):")
        for comp in pcap_comparisons:
            sig_text = "significant" if comp['significant'] else "not significant"
            print(f"  {comp['group1']} vs {comp['group2']}: diff={comp['diff']:.2f} "
                  f"({comp['low']:.2f} to {comp['up']:.2f}) → {sig_text}")
    else:
        print("No programmer capability groups with sufficient data.")

    print()

    # ====================== NASA-06 ======================
    # Compare defects by complexity ('cplx')
    cplx_stats = group_by_category(rows, 'cplx', 'Defects-')
    cplx_comparisons = compare_groups(cplx_stats)

    print("== NASA-06 ==")
    if cplx_stats:
        print("Complexity groups (n, mean Defects-):")
        for key, s in sorted(cplx_stats.items()):
            print(f"  {key:>4}: n={s['n']:>3}, mean Defects-={s['mean']:.2f}")
        print("\nPairwise comparison of Defects- between complexity groups (95% CI of difference):")
        for comp in cplx_comparisons:
            sig_text = "significant" if comp['significant'] else "not significant"
            print(f"  {comp['group1']} vs {comp['group2']}: diff={comp['diff']:.2f} "
                  f"({comp['low']:.2f} to {comp['up']:.2f}) → {sig_text}")
    else:
        print("No complexity groups with sufficient data.")

    print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
