#!/usr/bin/env python3
"""
APACHE skill 03

This script adds two new analyses:

APACHE-05: Compare configurations with KeepAlive enabled and disabled, and determine
            whether the observed performance differences are statistically meaningful.

APACHE-06: Compare configurations with access logging enabled and disabled,
            and indicate whether one group consistently performs better or whether
            the evidence is inconclusive.

Usage:
    python skill_03.py <path_to_Apache_AllMeasurements.csv>
"""

import sys
import csv
import math
from statistics import mean, stdev

# ---------------------------------------------------------------------------

def read_csv(path):
    """Read CSV and return a list of config dicts with numeric values."""
    configs = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cfg = {}
            for k, v in row.items():
                try:
                    cfg[k] = int(v)
                except ValueError:
                    cfg[k] = v
            configs.append(cfg)
    return configs

# ---------------------------------------------------------------------------

def group_stats(configs, group_col, target_col='Performance-'):
    """
    Split configs into two groups by group_col value and compute basic statistics.
    Returns a dict mapping group value to (mean, std, n, list of performances).
    """
    groups = {}
    for cfg in configs:
        key = cfg[group_col]
        groups.setdefault(key, []).append(cfg[target_col])
    stats = {}
    for key, vals in groups.items():
        stats[key] = {
            'mean': mean(vals),
            'std': stdev(vals) if len(vals) > 1 else 0.0,
            'n': len(vals),
            'vals': vals
        }
    return stats

# ---------------------------------------------------------------------------

def significance_analysis(stats_a, stats_b, group_label):
    """
    Compute difference, t-statistic, and p-value between two groups.
    Returns a formatted string with the results.
    """
    mean_a, std_a, n_a = stats_a['mean'], stats_a['std'], stats_a['n']
    mean_b, std_b, n_b = stats_b['mean'], stats_b['std'], stats_b['n']

    diff = mean_b - mean_a  # group B - group A
    se_a = std_a / math.sqrt(n_a) if n_a > 0 else 0.0
    se_b = std_b / math.sqrt(n_b) if n_b > 0 else 0.0
    se_diff = math.sqrt(se_a**2 + se_b**2)

    t_stat = diff / se_diff if se_diff != 0 else 0.0
    # Two‑tailed p‑value from normal distribution
    p_value = math.erfc(abs(t_stat) / math.sqrt(2))

    # Build output
    lines = [
        f"{group_label} Group 0: mean={mean_a:.2f}, std={std_a:.2f}, n={n_a}",
        f"{group_label} Group 1: mean={mean_b:.2f}, std={std_b:.2f}, n={n_b}",
        f"Difference (1 - 0): {diff:.2f}",
        f"t‑statistic: {t_stat:.2f}",
        f"p‑value: {p_value:.4f}"
    ]

    if p_value < 0.05:
        lines.append(f"Result: Significant difference in performance; {group_label} setting is likely impactful.")
    else:
        lines.append("Result: Not statistically significant; evidence inconclusive for a definitive recommendation.")
    return "\n".join(lines)

# ---------------------------------------------------------------------------

def main(csv_path):
    configs = read_csv(csv_path)
    if not configs:
        print("No data found.")
        return

    # APACHE-05: KeepAlive comparison
    print("== APACHE-05 ==")
    keepalive_stats = group_stats(configs, 'KeepAlive')
    if len(keepalive_stats) < 2:
        print("Insufficient data to compare KeepAlive settings.")
    else:
        print(significance_analysis(keepalive_stats.get(0, {'mean':0,'std':0,'n':0}),
                                    keepalive_stats.get(1, {'mean':0,'std':0,'n':0}),
                                    "KeepAlive"))

    # APACHE-06: accessLog comparison
    print("\n== APACHE-06 ==")
    accesslog_stats = group_stats(configs, 'accessLog')
    if len(accesslog_stats) < 2:
        print("Insufficient data to compare access logging settings.")
    else:
        print(significance_analysis(accesslog_stats.get(0, {'mean':0,'std':0,'n':0}),
                                    accesslog_stats.get(1, {'mean':0,'std':0,'n':0}),
                                    "accessLog"))

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <path_to_Apache_AllMeasurements.csv>")
        sys.exit(1)
    main(sys.argv[1])
