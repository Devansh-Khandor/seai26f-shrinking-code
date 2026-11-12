#!/usr/bin/env python3
"""
skill_03.py

Usage:
    python skill_03.py <csv_path>

The script prints:
    * CAR-05: Compare fuel efficiency across cylinder counts.
    * CAR-06: Compare overall performance between two model-year ranges.
"""

import sys
import math
from pathlib import Path

# Import helper functions from skill_01
from skill_01 import read_csv, mean, format_number, compute_composite_scores

# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #

def stddev(values):
    """Return standard deviation of a list of numbers."""
    n = len(values)
    if n < 2:
        return float('nan')
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance)

def t_test(mean1, sd1, n1, mean2, sd2, n2):
    """
    Perform Welch's two-sample t-test.
    Returns (t_stat, df, p_value).
    """
    if n1 < 2 or n2 < 2:
        return float('nan'), float('nan'), float('nan')
    se1 = sd1 ** 2 / n1 if n1 else 0
    se2 = sd2 ** 2 / n2 if n2 else 0
    denom = math.sqrt(se1 + se2)
    if denom == 0:
        return float('nan'), float('nan'), float('nan')
    t_stat = (mean1 - mean2) / denom
    # Welch-Satterthwaite equation for df
    num = (se1 + se2) ** 2
    den = (se1 ** 2) / (n1 - 1) + (se2 ** 2) / (n2 - 1) if (n1 > 1 and n2 > 1) else float('inf')
    df = num / den if den != 0 else float('inf')
    # Two-sided p-value from standard normal approximation
    p_value = math.erfc(abs(t_stat) / math.sqrt(2))
    return t_stat, df, p_value

# --------------------------------------------------------------------------- #
# CAR-05: Compare fuel efficiency across cylinder counts
# --------------------------------------------------------------------------- #

def print_car_05_efficiency_comparison(data):
    # Group mpg by cylinder count
    cyl_groups = {}
    for v in data:
        cyl = v['Clndrs']
        mpg = v['Mpg+']
        if mpg <= 0:
            continue
        cyl_groups.setdefault(cyl, []).append(mpg)

    # Prepare statistics per group
    stats = {}
    for cyl, mpgs in cyl_groups.items():
        stats[cyl] = {
            'n': len(mpgs),
            'mean': mean(mpgs),
            'sd': stddev(mpgs),
        }

    # Perform pairwise comparisons
    comparisons = []
    cyl_keys = sorted(stats.keys())
    for i in range(len(cyl_keys)):
        for j in range(i + 1, len(cyl_keys)):
            c1, c2 = cyl_keys[i], cyl_keys[j]
            s1, s2 = stats[c1], stats[c2]
            t_stat, df, p_val = t_test(s1['mean'], s1['sd'], s1['n'],
                                       s2['mean'], s2['sd'], s2['n'])
            comparisons.append({
                'cyl1': c1,
                'cyl2': c2,
                'mean1': s1['mean'],
                'mean2': s2['mean'],
                'diff': s1['mean'] - s2['mean'],
                't': t_stat,
                'df': df,
                'p': p_val,
            })

    # Output
    print("\n== CAR-05 ==")
    print("Fuel‑efficiency (MPG) comparison across cylinder counts:")
    if not comparisons:
        print("  No sufficient data for pairwise comparison.")
        return
    for comp in comparisons:
        signif = "significant" if comp['p'] < 0.05 else "not significant"
        print(
            f"  Cylinders {comp['cyl1']} vs {comp['cyl2']}: "
            f"mean MPG {format_number(comp['mean1'])} vs {format_number(comp['mean2'])}, "
            f"diff {format_number(comp['diff'])}, "
            f"t={format_number(comp['t'])} (df={format_number(comp['df'])}), "
            f"p={format_number(comp['p'])} -> {signif}"
        )

# --------------------------------------------------------------------------- #
# CAR-06: Compare overall performance between two model-year ranges
# --------------------------------------------------------------------------- #

def print_car_06_performance_comparison(data):
    # Define two model-year groups
    # For simplicity, use a split at model year <= 71 and > 71
    group_a = [v for v in data if v['Model'] <= 71]
    group_b = [v for v in data if v['Model'] > 71]

    # Compute composite scores for each group
    scores_a = [score for score, _ in compute_composite_scores(group_a)]
    scores_b = [score for score, _ in compute_composite_scores(group_b)]

    # Statistics
    stats_a = {
        'n': len(scores_a),
        'mean': mean(scores_a),
        'sd': stddev(scores_a),
    }
    stats_b = {
        'n': len(scores_b),
        'mean': mean(scores_b),
        'sd': stddev(scores_b),
    }

    t_stat, df, p_val = t_test(stats_a['mean'], stats_a['sd'], stats_a['n'],
                               stats_b['mean'], stats_b['sd'], stats_b['n'])

    # Output
    print("\n== CAR-06 ==")
    print("Overall performance comparison between model-year ranges:")
    print(f"  Group 1 (Model <= 71): N={stats_a['n']}, mean composite {format_number(stats_a['mean'])}")
    print(f"  Group 2 (Model > 71): N={stats_b['n']}, mean composite {format_number(stats_b['mean'])}")
    if math.isnan(t_stat):
        print("  Insufficient data for statistical comparison.")
        return
    signif = "significant" if p_val < 0.05 else "not significant"
    print(
        f"  Difference in mean composite: {format_number(stats_a['mean'] - stats_b['mean'])}, "
        f"t={format_number(t_stat)} (df={format_number(df)}), p={format_number(p_val)} -> {signif}"
    )

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    data = read_csv(csv_path)
    if not data:
        print("No data found in CSV.", file=sys.stderr)
        sys.exit(1)

    print_car_05_efficiency_comparison(data)
    print_car_06_performance_comparison(data)

if __name__ == "__main__":
    main()
