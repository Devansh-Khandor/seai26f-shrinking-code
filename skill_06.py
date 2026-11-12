#!/usr/bin/env python3
"""
APACHE skill 06

This script implements the following analyses:

APACHE-11: For each configuration that the system recommends (i.e.
performance ≤ mean – std), explain why it is recommended by comparing
its settings and performance to the overall mean configuration and to the
overall performance statistics.

APACHE-12: For pairs of configurations that receive different performance
assessments (one recommended, one non‑recommended), identify the settings
that differ and describe how those differences relate to the observed
performance difference.

Usage:
    python skill_06.py <path_to_Apache_AllMeasurements.csv>
"""

import sys
import csv
import math
from statistics import mean, stdev

# ---------------------------------------------------------------------------

def read_csv(path):
    """Read CSV and return a list of configuration dictionaries (int values)."""
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

def compute_overall_stats(configs, perf_col='Performance-'):
    """Return mean, std of performance and per‑feature mean values."""
    performances = [c[perf_col] for c in configs]
    mean_perf = mean(performances)
    std_perf = stdev(performances) if len(performances) > 1 else 0.0

    # per‑feature averages
    feature_avgs = {}
    feature_names = [k for k in configs[0].keys() if k != perf_col]
    for name in feature_names:
        vals = [c[name] for c in configs]
        feature_avgs[name] = mean(vals)

    return mean_perf, std_perf, feature_avgs

# ---------------------------------------------------------------------------

def apache_11(configs, mean_perf, std_perf, feature_avgs, perf_col='Performance-'):
    """Analysis APACHE-11."""
    print("== APACHE-11 ==")

    threshold = mean_perf - std_perf
    recommended = [ (idx, cfg) for idx, cfg in enumerate(configs, start=1)
                   if cfg[perf_col] <= threshold ]

    if not recommended:
        print("No recommended configurations found (performance <= mean - std).")
        return

    for idx, cfg in recommended:
        perf = cfg[perf_col]
        perf_gain = mean_perf - perf
        print(f"Configuration #{idx} recommended: Performance={perf} (better by {perf_gain:.1f} ms).")
        # Compare settings to overall averages
        diff_features = []
        for name, avg in feature_avgs.items():
            diff = cfg[name] - avg
            diff_features.append((abs(diff), name, diff, avg))
        # Top 3 features that deviate most from the average
        diff_features.sort(reverse=True)
        top3 = diff_features[:3]
        print("  Key settings differing from overall mean:")
        for _, name, diff, avg in top3:
            direction = "higher" if diff > 0 else "lower"
            print(f"    {name}: {cfg[name]} (overall mean: {avg:.3f}) – {direction} than average")
        print()

# ---------------------------------------------------------------------------

def apache_12(configs, mean_perf, std_perf, perf_col='Performance-'):
    """Analysis APACHE-12."""
    print("== APACHE-12 ==")

    good_thresh = mean_perf - std_perf
    bad_thresh = mean_perf + std_perf

    good_cfgs = [ (idx, cfg) for idx, cfg in enumerate(configs, start=1)
                  if cfg[perf_col] <= good_thresh ]
    bad_cfgs = [ (idx, cfg) for idx, cfg in enumerate(configs, start=1)
                 if cfg[perf_col] >= bad_thresh ]

    if not good_cfgs or not bad_cfgs:
        print("Not enough good/bad configurations to compare.")
        return

    # Compute pairs sorted by absolute performance difference
    pairs = []
    for g_idx, g_cfg in good_cfgs:
        for b_idx, b_cfg in bad_cfgs:
            perf_diff = g_cfg[perf_col] - b_cfg[perf_col]   # negative (good < bad)
            pairs.append((abs(perf_diff), g_idx, b_idx, g_cfg, b_cfg, perf_diff))

    pairs.sort(reverse=True)  # largest abs difference first
    top_pairs = pairs[:3]  # show top 3 pairs

    for _, g_idx, b_idx, g_cfg, b_cfg, perf_diff in top_pairs:
        print(f"Pair: Config #{g_idx} (good) vs Config #{b_idx} (bad) – performance diff = {perf_diff:.1f} ms")
        # Identify differing settings
        differing = [name for name in g_cfg if name != perf_col and g_cfg[name] != b_cfg[name]]
        if not differing:
            print("  No differing settings found.")
            print()
            continue
        print("  Settings that differ:")
        for name in differing:
            print(f"    {name}: good={g_cfg[name]}, bad={b_cfg[name]}")
        # Summarize impact
        print(f"  Interpretation: Differences in these settings likely contribute to the {abs(perf_diff):.1f} ms performance gap.")
        print()

# ---------------------------------------------------------------------------

def main(csv_path):
    configs = read_csv(csv_path)
    if not configs:
        print("No data found.")
        return

    perf_col = 'Performance-'
    mean_perf, std_perf, feature_avgs = compute_overall_stats(configs, perf_col)

    apache_11(configs, mean_perf, std_perf, feature_avgs, perf_col)
    apache_12(configs, mean_perf, std_perf, perf_col)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <path_to_Apache_AllMeasurements.csv>")
        sys.exit(1)
    main(sys.argv[1])
