#!/usr/bin/env python3
"""
APACHE skill 01

This script analyses Apache server configuration data and answers two
specific questions:

1. APACHE-01: Show a summary of the recorded performance values (typical,
   best and worst).

2. APACHE-02: Identify configurations with unusually favorable performance
   values and highlight the settings that distinguish them from the
   average configuration.

Usage:
    python skill_01.py <path_to_Apache_AllMeasurements.csv>
"""

import sys
import csv
import math
from statistics import mean, stdev

def read_csv(path):
    """Read the CSV file and return a list of configuration dicts."""
    configs = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert all numeric fields to int
            cfg = {}
            for k, v in row.items():
                try:
                    cfg[k] = int(v)
                except ValueError:
                    cfg[k] = v  # keep as string if not numeric
            configs.append(cfg)
    return configs

def performance_stats(configs, perf_col='Performance-'):
    """Return mean, min, max of the performance column."""
    performances = [cfg[perf_col] for cfg in configs]
    return mean(performances), min(performances), max(performances)

def average_config(configs, numeric_cols):
    """Return the mean value of each numeric column across all configs."""
    avg = {}
    for col in numeric_cols:
        values = [cfg[col] for cfg in configs]
        avg[col] = mean(values)
    return avg

def differentiate(config, avg):
    """Return a list of (column, value) pairs where config differs from avg."""
    diff = []
    for col, val in config.items():
        if isinstance(val, (int, float)):
            if abs(val - avg[col]) > 0.5:  # tolerance
                diff.append((col, val))
    return diff

def main(csv_path):
    configs = read_csv(csv_path)

    # Identify numeric columns (exclude the performance column)
    if not configs:
        print("No data found.")
        return
    all_cols = configs[0].keys()
    perf_col = 'Performance-'
    numeric_cols = [c for c in all_cols if c != perf_col]

    # APACHE-01
    print("== APACHE-01 ==")
    mean_perf, min_perf, max_perf = performance_stats(configs, perf_col)
    print(f"Typical performance (mean): {mean_perf:.2f}")
    print(f"Best performance: {min_perf}")
    print(f"Worst performance: {max_perf}")

    # APACHE-02
    print("\n== APACHE-02 ==")

    # Compute threshold: mean - std (unusually favorable)
    performances = [cfg[perf_col] for cfg in configs]
    if len(performances) > 1:
        std_perf = stdev(performances)
    else:
        std_perf = 0
    threshold = mean_perf - std_perf

    fav_configs = [cfg for cfg in configs if cfg[perf_col] <= threshold]

    if not fav_configs:
        print("No unusually favorable configurations found.")
        return

    avg_cfg = average_config(configs, numeric_cols)

    print(f"Unusually favorable configurations (threshold <= {threshold:.2f}):")
    for cfg in fav_configs:
        # Format configuration string
        cfg_str = ', '.join(f"{k}={v}" for k, v in cfg.items())
        print(f"- {cfg_str}")
        # Show distinguishing settings
        diff = differentiate(cfg, avg_cfg)
        if diff:
            print("  Distinct settings:")
            for k, v in diff:
                print(f"    {k}: {v} (avg: {avg_cfg[k]:.2f})")
        else:
            print("  No distinct settings compared to average configuration.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <path_to_csv>")
        sys.exit(1)
    main(sys.argv[1])
