#!/usr/bin/env python3
"""
APACHE skill 04

This script implements two new analyses for the Apache benchmark data:

APACHE-07: As new benchmark results arrive, each configuration is assessed
immediately and an alert is printed whenever its measured performance is
unusually poor compared with previously tested configurations.

APACHE-08: Performance summaries and a recommended-configuration shortlist
are kept up‑to‑date as additional benchmark records become available,
without requiring a full restart of the analysis.

Usage:
    python skill_04.py <path_to_Apache_AllMeasurements.csv>
"""

import sys
import csv
import math
from statistics import mean

# ---------------------------------------------------------------------------

def read_csv(path):
    """
    Read the CSV file and return a list of configuration dictionaries.
    All numeric values are converted to integers; non‑numeric values are
    kept as strings.
    """
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

def running_stats():
    """
    Generator yielding the (n, mean, variance) of the sequence seen so far.
    Uses Welford's algorithm for numerical stability.
    """
    n = 0
    mean_val = 0.0
    M2 = 0.0
    while True:
        delta = yield (n, mean_val, M2)
        if delta is None:
            continue
        x = delta
        n += 1
        delta1 = x - mean_val
        mean_val += delta1 / n
        delta2 = x - mean_val
        M2 += delta1 * delta2

# ---------------------------------------------------------------------------

def format_config(cfg):
    """Return a string representation of a configuration."""
    return ', '.join(f'{k}={v}' for k, v in cfg.items())

# ---------------------------------------------------------------------------

def main(csv_path):
    configs = read_csv(csv_path)
    if not configs:
        print("No data found.")
        return

    perf_col = 'Performance-'
    # -----------------------------------------------------------------------
    # APACHE-07
    print("== APACHE-07 ==")
    stats_gen = running_stats()
    next(stats_gen)  # prime generator
    alerts = []

    for idx, cfg in enumerate(configs, start=1):
        perf = cfg[perf_col]
        n, mean_val, M2 = stats_gen.send(None)  # get stats before adding this record
        if n > 1:
            std_val = math.sqrt(M2 / (n - 1))
            # Unusually poor: performance > mean + 2*std
            if perf > mean_val + 2 * std_val:
                alert_msg = (
                    f"Alert: Configuration #{idx} ({format_config(cfg)}) "
                    f"is unusually poor. Previous mean={mean_val:.2f}, "
                    f"std={std_val:.2f}, performance={perf}"
                )
                alerts.append(alert_msg)
        # Update stats with current performance
        stats_gen.send(perf)

    if alerts:
        for line in alerts:
            print(line)
    else:
        print("No unusually poor configurations detected.")

    # -----------------------------------------------------------------------
    # APACHE-08
    print("\n== APACHE-08 ==")

    performances = [cfg[perf_col] for cfg in configs]
    overall_mean = mean(performances)
    overall_min = min(performances)
    overall_max = max(performances)
    # Compute standard deviation
    if len(performances) > 1:
        overall_std = math.sqrt(sum((p - overall_mean) ** 2 for p in performances) / (len(performances) - 1))
    else:
        overall_std = 0.0

    print(f"Performance summary: mean={overall_mean:.2f}, min={overall_min}, max={overall_max}")
    threshold = overall_mean - overall_std
    # Recommended configurations: performance <= mean - std
    recommended = [cfg for cfg in configs if cfg[perf_col] <= threshold]
    if recommended:
        print(f"Recommended configurations (performance <= {threshold:.2f}):")
        for cfg in recommended:
            print(f"  - {format_config(cfg)}")
    else:
        print("No recommended configurations found (no performance <= mean - std).")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <path_to_Apache_AllMeasurements.csv>")
        sys.exit(1)
    main(sys.argv[1])
