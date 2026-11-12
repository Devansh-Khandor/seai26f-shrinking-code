#!/usr/bin/env python3
"""
Apache performance comparison for specific configuration flags.

This script builds on the earlier skill_01.py and skill_02.py modules
to provide two new analysis points:

- APACHE-05: Compare KeepAlive enabled vs disabled.
- APACHE-06: Compare access logging enabled vs disabled.

The output includes the mean performance for each group, the difference,
an approximate standard error, and a short verdict on whether the
observed difference is likely meaningful for decision making.

Usage:
    python skill_03.py <csv_path>
"""

import sys
import statistics
import skill_01

def load_csv(path: str):
    """Delegate to skill_01's load_csv."""
    return skill_01.load_csv(path)

def group_stats(rows, feature, value):
    """Return statistics for the subset of rows where feature==value."""
    group = [row for row in rows if row.get(feature) == value]
    if not group:
        return {'count': 0, 'mean': 0.0, 'stddev': 0.0, 'data': []}
    perf = [row['Performance-'] for row in group]
    mean = statistics.mean(perf)
    stddev = statistics.stdev(perf) if len(perf) > 1 else 0.0
    return {'count': len(group), 'mean': mean, 'stddev': stddev, 'data': perf}

def compare_feature(rows, feature):
    """Compute statistics for enabled/disabled groups and difference."""
    enabled = group_stats(rows, feature, 1)
    disabled = group_stats(rows, feature, 0)

    # If either group is missing, we cannot compare
    if enabled['count'] == 0 or disabled['count'] == 0:
        return None

    # Difference: mean_enabled - mean_disabled
    diff = enabled['mean'] - disabled['mean']

    # Standard error of the difference (approximate)
    se = ((enabled['stddev'] ** 2) / enabled['count'] +
          (disabled['stddev'] ** 2) / disabled['count']) ** 0.5

    # Determine significance: using a simple 2*SE threshold
    significant = abs(diff) > 2 * se

    return {
        'feature': feature,
        'enabled_mean': enabled['mean'],
        'disabled_mean': disabled['mean'],
        'diff': diff,
        'se': se,
        'significant': significant
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)

    # APACHE-05: KeepAlive comparison
    print("== APACHE-05 ==")
    ka_res = compare_feature(rows, 'keepAlive')
    if ka_res is None:
        print("Insufficient data for KeepAlive comparison.")
    else:
        print(f"Mean performance with KeepAlive=1: {ka_res['enabled_mean']:.2f}")
        print(f"Mean performance with KeepAlive=0: {ka_res['disabled_mean']:.2f}")
        print(f"Difference (enabled - disabled): {ka_res['diff']:.2f} ± {ka_res['se']:.2f}")
        verdict = ("Significant difference." if ka_res['significant']
                   else "Difference not statistically significant.")
        print(verdict)

    # APACHE-06: accessLog comparison
    print("\n== APACHE-06 ==")
    al_res = compare_feature(rows, 'accessLog')
    if al_res is None:
        print("Insufficient data for accessLog comparison.")
    else:
        print(f"Mean performance with accessLog=1: {al_res['enabled_mean']:.2f}")
        print(f"Mean performance with accessLog=0: {al_res['disabled_mean']:.2f}")
        print(f"Difference (enabled - disabled): {al_res['diff']:.2f} ± {al_res['se']:.2f}")
        verdict = ("Significant difference." if al_res['significant']
                   else "Difference not statistically significant.")
        print(verdict)

if __name__ == "__main__":
    main()
