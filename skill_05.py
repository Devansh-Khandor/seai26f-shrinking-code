#!/usr/bin/env python3
"""
APACHE skill 05

This script implements two new analyses for the Apache benchmark data:

APACHE-09: When a configuration performs poorly, identify which of its
enabled or disabled settings are most strongly associated with
unfavorable performance in our historical measurements.

APACHE-10: Identify combinations of server settings that frequently
appear among poorly performing configurations and show how they
differ from better-performing combinations.

Usage:
    python skill_05.py <path_to_Apache_AllMeasurements.csv>
"""

import sys
import csv
import math
from collections import defaultdict
from itertools import combinations

# ---------------------------------------------------------------------------

def read_csv(path):
    """Read CSV and return a list of configuration dictionaries with int values."""
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

def compute_perf_stats(configs, perf_col='Performance-'):
    """Return overall mean, std, and threshold for poor configurations."""
    performances = [c[perf_col] for c in configs]
    mean_perf = sum(performances) / len(performances)
    if len(performances) > 1:
        std_perf = math.sqrt(sum((p - mean_perf) ** 2 for p in performances) / (len(performances) - 1))
    else:
        std_perf = 0.0
    threshold = mean_perf + std_perf   # Poor if >= mean + std
    return mean_perf, std_perf, threshold

# ---------------------------------------------------------------------------

def format_config(cfg):
    """Return a string representation of a configuration."""
    return ', '.join(f'{k}={v}' for k, v in cfg.items())

# ---------------------------------------------------------------------------

def apache_09(configs, perf_col='Performance-'):
    """Analysis APACHE-09."""
    print("== APACHE-09 ==")

    mean_perf, std_perf, threshold = compute_perf_stats(configs, perf_col)

    # Identify poor configurations
    poor_cfgs = [cfg for cfg in configs if cfg[perf_col] >= threshold]
    if not poor_cfgs:
        print("No poorly performing configurations detected.")
        return

    # Compute per‑feature performance difference
    feature_diffs = {}   # feature -> (mean_1 - mean_0)
    for col in configs[0].keys():
        if col == perf_col:
            continue
        vals1 = [c[perf_col] for c in configs if c[col] == 1]
        vals0 = [c[perf_col] for c in configs if c[col] == 0]
        if not vals1 or not vals0:
            # Skip features that never vary
            continue
        mean1 = sum(vals1) / len(vals1)
        mean0 = sum(vals0) / len(vals0)
        feature_diffs[col] = mean1 - mean0

    # Sort features by absolute difference (most impactful first)
    sorted_features = sorted(feature_diffs.items(), key=lambda kv: abs(kv[1]), reverse=True)

    # Report top 5 features overall
    print("\nTop 5 features most strongly associated with performance:")
    for feat, diff in sorted_features[:5]:
        direction = "1 worse" if diff > 0 else "0 worse"
        print(f"  {feat:15s} : {diff:+.2f} ({direction})")

    # For each poor config, list the features that match the "bad" side
    print("\nDetailed analysis for each poorly performing configuration:")
    for idx, cfg in enumerate(poor_cfgs, start=1):
        bad_features = []
        for feat, diff in sorted_features[:10]:  # consider top 10 for matching
            if diff > 0 and cfg[feat] == 1:
                bad_features.append(feat)
            elif diff < 0 and cfg[feat] == 0:
                bad_features.append(feat)
        if bad_features:
            print(f"Configuration #{idx} ({format_config(cfg)}):")
            print(f"  Bad features: {', '.join(bad_features)}")
        else:
            print(f"Configuration #{idx} ({format_config(cfg)}):")
            print("  No clearly bad features detected among top indicators.")

# ---------------------------------------------------------------------------

def apache_10(configs, perf_col='Performance-'):
    """Analysis APACHE-10."""
    print("\n== APACHE-10 ==")

    mean_perf, std_perf, threshold = compute_perf_stats(configs, perf_col)

    # Identify poor configurations
    poor_cfgs = [cfg for cfg in configs if cfg[perf_col] >= threshold]
    if not poor_cfgs:
        print("No poorly performing configurations detected.")
        return

    feature_names = [c for c in configs[0].keys() if c != perf_col]
    # Count pair support among poor configs
    pair_counts = defaultdict(int)
    pair_perf_sum = defaultdict(float)

    for cfg in poor_cfgs:
        # Get set of features that are enabled (value == 1)
        enabled = [f for f in feature_names if cfg[f] == 1]
        # Consider all unique pairs of enabled features
        for f1, f2 in combinations(sorted(enabled), 2):
            pair = (f1, f2)
            pair_counts[pair] += 1
            pair_perf_sum[pair] += cfg[perf_col]

    if not pair_counts:
        print("No pairs of enabled settings found in poor configurations.")
        return

    # Prepare results sorted by support
    pair_stats = []
    for pair, count in pair_counts.items():
        avg_perf = pair_perf_sum[pair] / count
        pair_stats.append((count, avg_perf, pair))

    # Sort by support descending, then by higher avg performance
    pair_stats.sort(key=lambda kv: (-kv[0], -kv[1]))

    # Show top 5 pairs
    print("\nTop 5 frequently occurring pairs of enabled settings in poor configs:")
    for count, avg_perf, pair in pair_stats[:5]:
        print(f"  Pair {pair[0]} & {pair[1]}: support={count}, avg_performance={avg_perf:.2f}")

    # Compare with overall performance
    overall_mean = mean_perf
    print(f"\nOverall mean performance: {overall_mean:.2f}")
    print("Pairs with average performance worse than overall mean:")
    for count, avg_perf, pair in pair_stats:
        if avg_perf > overall_mean:
            print(f"  {pair[0]} & {pair[1]} (support={count}) -> avg_perf={avg_perf:.2f}")

# ---------------------------------------------------------------------------

def main(csv_path):
    configs = read_csv(csv_path)
    if not configs:
        print("No data found.")
        return

    apache_09(configs)
    apache_10(configs)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <path_to_Apache_AllMeasurements.csv>")
        sys.exit(1)
    main(sys.argv[1])
