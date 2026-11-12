#!/usr/bin/env python3
"""
skill_03.py – Statistical comparison of Apache configurations for KeepAlive
and access logging settings.

Usage:
    python skill_03.py <path-to-Apache_AllMeasurements.csv>
"""

import sys
import csv
import math
import numpy as np
import skill_01 as s1   # Reuse utilities from previous skills

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def load_csv(path):
    """Load CSV into a list of dict rows and return column names."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames
    return rows, headers

def group_stats(rows, group_col, target_col='Performance-'):
    """
    Compute basic statistics for configurations that share the same value
    in `group_col`. Returns a dict mapping the group value to a stats dict.
    """
    groups = {}
    for row in rows:
        key = row[group_col]
        val = int(row[target_col])
        groups.setdefault(key, []).append(val)

    stats = {}
    for key, vals in groups.items():
        np_vals = np.array(vals, dtype=int)
        stats[key] = {
            'count': len(vals),
            'mean': np_vals.mean(),
            'median': np.median(np_vals),
            'std': np_vals.std(ddof=0),   # population std
            'values': np_vals
        }
    return stats

def cohen_d(mean1, mean0, std1, std0):
    """
    Cohen's d effect size for two independent samples.
    """
    pooled_std = math.sqrt((std1**2 + std0**2) / 2)
    if pooled_std == 0:
        return 0.0
    return (mean1 - mean0) / pooled_std

def confidence_interval(mean1, mean0, std1, std0, n1, n0, z=1.96):
    """
    95% confidence interval for the difference in means between two samples.
    """
    se = math.sqrt(std1**2 / n1 + std0**2 / n0)
    diff = mean1 - mean0
    return diff - z * se, diff + z * se

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main(csv_path):
    rows, headers = load_csv(csv_path)

    # ------------------------------------------------------------------
    # APACHE-05: KeepAlive comparison
    # ------------------------------------------------------------------
    keepalive_stats = group_stats(rows, 'keepAlive')
    print("== APACHE-05 ==")
    if len(keepalive_stats) != 2:
        print("  Insufficient data for KeepAlive comparison.")
    else:
        enabled = keepalive_stats.get('1', {})
        disabled = keepalive_stats.get('0', {})
        # Print summary
        print(f"  KeepAlive enabled: {enabled['count']} configs, "
              f"mean={enabled['mean']:.2f}, median={enabled['median']:.2f}, "
              f"std={enabled['std']:.2f}")
        print(f"  KeepAlive disabled: {disabled['count']} configs, "
              f"mean={disabled['mean']:.2f}, median={disabled['median']:.2f}, "
              f"std={disabled['std']:.2f}")
        # Effect size
        d = cohen_d(enabled['mean'], disabled['mean'],
                    enabled['std'], disabled['std'])
        print(f"  Cohen's d effect size: {d:.2f}")
        # 95% CI
        ci_low, ci_high = confidence_interval(
            enabled['mean'], disabled['mean'],
            enabled['std'], disabled['std'],
            enabled['count'], disabled['count'])
        print(f"  95% CI for mean difference: ({ci_low:.2f}, {ci_high:.2f})")
        # Interpretation
        if ci_low > 0:
            print("  Interpretation: KeepAlive enabled yields significantly higher performance.")
        elif ci_high < 0:
            print("  Interpretation: KeepAlive enabled yields significantly lower performance.")
        else:
            print("  Interpretation: No statistically significant difference in performance.")

    # ------------------------------------------------------------------
    # APACHE-06: Access logging comparison
    # ------------------------------------------------------------------
    access_stats = group_stats(rows, 'accessLog')
    print("\n== APACHE-06 ==")
    if len(access_stats) != 2:
        print("  Insufficient data for access log comparison.")
    else:
        enabled = access_stats.get('1', {})
        disabled = access_stats.get('0', {})
        # Print summary
        print(f"  Access logging enabled: {enabled['count']} configs, "
              f"mean={enabled['mean']:.2f}, median={enabled['median']:.2f}, "
              f"std={enabled['std']:.2f}")
        print(f"  Access logging disabled: {disabled['count']} configs, "
              f"mean={disabled['mean']:.2f}, median={disabled['median']:.2f}, "
              f"std={disabled['std']:.2f}")
        # Effect size
        d = cohen_d(enabled['mean'], disabled['mean'],
                    enabled['std'], disabled['std'])
        print(f"  Cohen's d effect size: {d:.2f}")
        # 95% CI
        ci_low, ci_high = confidence_interval(
            enabled['mean'], disabled['mean'],
            enabled['std'], disabled['std'],
            enabled['count'], disabled['count'])
        print(f"  95% CI for mean difference: ({ci_low:.2f}, {ci_high:.2f})")
        # Interpretation
        if ci_low > 0:
            print("  Interpretation: Access logging enabled yields significantly higher performance.")
        elif ci_high < 0:
            print("  Interpretation: Access logging enabled yields significantly lower performance.")
        else:
            print("  Interpretation: Evidence inconclusive; no clear performance advantage.")

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python skill_03.py <csv_path>")
    main(sys.argv[1])
