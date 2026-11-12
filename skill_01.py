#!/usr/bin/env python3
"""
skill_01.py

This script performs the requested analyses on an Apache configuration CSV file.

Usage:
    python skill_01.py <path_to_csv>

The script prints two sections:
    == APACHE-01 ==  -- Summary of performance (typical, best, worst)
    == APACHE-02 ==  -- Configurations with unusually favorable performance
"""

import csv
import sys
import math
from collections import Counter, defaultdict

def read_configurations(csv_path):
    """Read CSV and return list of configuration dicts."""
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        configs = []
        for row in reader:
            # Convert numeric fields to float for analysis
            config = {k: float(v) if v.replace('.', '', 1).isdigit() else v
                      for k, v in row.items()}
            configs.append(config)
    return configs

def performance_stats(configs):
    """Return typical (mean), best (max), worst (min) performance values."""
    performances = [c['Performance-'] for c in configs]
    mean = sum(performances) / len(performances)
    best = max(performances)
    worst = min(performances)
    # Standard deviation
    variance = sum((p - mean) ** 2 for p in performances) / len(performances)
    stddev = math.sqrt(variance)
    return mean, best, worst, stddev

def compute_mode_per_setting(configs):
    """Return mode (most common value) for each setting (excluding Performance-)."""
    modes = {}
    # get all setting keys except performance
    setting_keys = [k for k in configs[0].keys() if k != 'Performance-']
    for key in setting_keys:
        counter = Counter(c[key] for c in configs)
        mode_value, _ = counter.most_common(1)[0]
        modes[key] = mode_value
    return modes

def find_unusually_favorable(configs, mean, stddev, threshold_factor=1.5):
    """Return list of configs whose performance is >= mean + threshold_factor*stddev."""
    threshold = mean + threshold_factor * stddev
    return [c for c in configs if c['Performance-'] >= threshold]

def format_differences(config, modes):
    """Return string of settings that differ from the mode."""
    diffs = []
    for key, val in config.items():
        if key == 'Performance-':
            continue
        if val != modes[key]:
            diffs.append(f"{key}={int(val) if val.is_integer() else val}")
    return ", ".join(diffs) if diffs else "No distinguishing settings"

def main(csv_path):
    configs = read_configurations(csv_path)
    mean, best, worst, stddev = performance_stats(configs)
    modes = compute_mode_per_setting(configs)

    # APACHE-01
    print("== APACHE-01 ==")
    print(f"Typical performance (mean): {mean:.2f}")
    print(f"Best performance: {best:.2f}")
    print(f"Worst performance: {worst:.2f}")

    # APACHE-02
    high_perf_configs = find_unusually_favorable(configs, mean, stddev, threshold_factor=1.5)
    print("\n== APACHE-02 ==")
    if not high_perf_configs:
        print("No configurations exceed the unusual threshold.")
    else:
        print(f"Found {len(high_perf_configs)} unusually favorable configuration(s):")
        for idx, cfg in enumerate(high_perf_configs, 1):
            # Show full config and distinguishing settings
            cfg_str = ", ".join(f"{k}={int(v) if isinstance(v, float) and v.is_integer() else v}"
                                for k, v in cfg.items())
            diffs = format_differences(cfg, modes)
            print(f"{idx}. {cfg_str}")
            print(f"   Distinguishing settings: {diffs}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <path_to_csv>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
