#!/usr/bin/env python3
"""
Skill 03: Compare effort and defect estimates across different programmer capability
ratings and project complexity ratings.

Usage:
    python skill_03.py <csv_path>
"""

import sys
import numpy as np

# Reuse data loading from skill_01
try:
    import skill_01 as s01
except ImportError:
    # Minimal fallback if skill_01 is unavailable
    def load_data(csv_path: str):
        rows = []
        import csv
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    r['Kloc+'] = float(r['Kloc+'])
                    r['Effort-'] = float(r['Effort-'])
                    r['Defects-'] = float(r['Defects-'])
                except (ValueError, KeyError):
                    continue
                rows.append(r)
        return rows
    s01 = type('s01', (), {'load_data': staticmethod(load_data)})

def compute_group_stats(rows, group_col: str, value_col: str):
    """
    Group rows by group_col and compute mean, std, CI for value_col.
    Returns a dict mapping group value -> stats dict.
    """
    groups = {}
    for r in rows:
        key = r[group_col]
        val = r[value_col]
        groups.setdefault(key, []).append(val)

    stats_dict = {}
    for key, vals in groups.items():
        arr = np.array(vals)
        n = len(arr)
        mean = arr.mean()
        # Use sample std; handle n=1
        if n > 1:
            sd = arr.std(ddof=1)
        else:
            sd = 0.0
        se = sd / np.sqrt(n) if n > 0 else 0.0
        ci_low = mean - 1.96 * se
        ci_high = mean + 1.96 * se
        stats_dict[key] = {
            'n': n,
            'mean': mean,
            'sd': sd,
            'ci_low': ci_low,
            'ci_high': ci_high,
        }
    return stats_dict

def ci_overlap(stats1, stats2):
    """Return True if two 95% CIs overlap."""
    return not (stats1['ci_high'] < stats2['ci_low'] or stats2['ci_high'] < stats1['ci_low'])

def analyze_rating(rows, rating_col, value_col, label):
    """
    Generate report for a particular rating column and value column.
    Prints the NASA-XX header and results.
    """
    group_stats = compute_group_stats(rows, rating_col, value_col)
    # Sort groups by rating for consistent output
    sorted_groups = sorted(group_stats.items(), key=lambda x: x[0])

    print(f"== {label} ==")
    for rating, stats in sorted_groups:
        print(f"Rating '{rating}': {stats['n']} projects, mean {value_col} "
              f"{stats['mean']:.2f} ± {stats['sd']:.2f} "
              f"(95% CI: {stats['ci_low']:.2f} – {stats['ci_high']:.2f})")

    # Determine if any pairwise comparisons are meaningful
    meaningful = False
    for i in range(len(sorted_groups)):
        for j in range(i + 1, len(sorted_groups)):
            if not ci_overlap(sorted_groups[i][1], sorted_groups[j][1]):
                meaningful = True
                break
        if meaningful:
            break

    if meaningful:
        print(f"{label} indicates statistically meaningful differences between "
              f"ratings.")
    else:
        print(f"{label} shows no statistically significant differences or "
              f"insufficient evidence across ratings.")

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = s01.load_data(csv_path)
    if not rows:
        print("No valid data found.")
        sys.exit(1)

    # NASA-05: Programmer capability ratings (pcap) vs effort
    analyze_rating(rows, 'pcap', 'Effort-', 'NASA-05')

    # NASA-06: Complexity ratings (cplx) vs defect outcomes
    analyze_rating(rows, 'cplx', 'Defects-', 'NASA-06')

if __name__ == "__main__":
    main()
