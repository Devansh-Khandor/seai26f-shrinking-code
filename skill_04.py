#!/usr/bin/env python3
"""
skill_04.py

Implements the following analyses for Apache configuration data:

APACHE-07: As new benchmark results are read, alert when a configuration's
           measured performance is unusually poor compared with all previously
           tested configurations.

APACHE-08: Keeps the performance summaries and the recommended configuration
           shortlist up‑to‑date as records are added, without requiring a
           manual restart of the analysis.

The script processes the CSV file incrementally, updating statistics on the
fly.  No external dependencies beyond the Python 3 standard library are used.
"""

import csv
import sys
import math
from collections import defaultdict

# --------------------------------------------------------------------------- #
# Helper functions for incremental statistics
# --------------------------------------------------------------------------- #

def update_running_stats(count, mean, M2, new_value):
    """
    Update mean and running sum of squares of differences (M2) using Welford's
    algorithm.  Returns new count, mean, M2.
    """
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2
    return count, mean, M2

def running_stddev(count, M2):
    """
    Compute standard deviation from running M2.
    """
    if count < 2:
        return 0.0
    return math.sqrt(M2 / (count - 1))

def format_config(cfg):
    """
    Return a human‑readable string for a configuration dict.
    """
    return ", ".join(
        f"{k}={int(v) if isinstance(v, float) and v.is_integer() else v}"
        for k, v in cfg.items()
    )

# --------------------------------------------------------------------------- #
# APACHE-07: Unusually poor alert
# --------------------------------------------------------------------------- #

def process_row_for_alert(cfg, count, mean, M2):
    """
    Decide whether the current configuration is unusually poor.  If so,
    print an alert.  The threshold is mean - 1.5 * stddev.
    """
    perf = cfg['Performance-']
    # After updating stats for this row, we have the new mean/M2,
    # but we want to compare against previous stats (excluding this row).
    if count > 1:
        # mean and M2 are from previous rows only
        stddev = running_stddev(count - 1, M2)
        threshold = mean - 1.5 * stddev
        if perf < threshold:
            print(f"== APACHE-07 ==")
            print(f"Unusually poor performance detected.")
            print(f"Configuration: {format_config(cfg)}")
            print(f"Measured Performance: {perf:.2f}")
            print(f"Previous Mean: {mean:.2f}, Stddev: {stddev:.2f}")
            print(f"Threshold (mean - 1.5*stddev): {threshold:.2f}")
            print()

# --------------------------------------------------------------------------- #
# APACHE-08: Updated summaries and shortlist
# --------------------------------------------------------------------------- #

def update_best_configs(best_list, cfg, best_perf):
    """
    Maintain a list of the best configurations seen so far (top 3).
    """
    perf = cfg['Performance-']
    if perf > best_perf[0]:
        best_list.append((perf, cfg))
        best_list.sort(reverse=True, key=lambda x: x[0])
        if len(best_list) > 3:
            best_list.pop()  # keep only top 3
    elif len(best_list) < 3:
        best_list.append((perf, cfg))
        best_list.sort(reverse=True, key=lambda x: x[0])

def print_summary(count, mean, stddev, worst_perf, best_list):
    """
    Print the APACHE-08 summary after processing all rows.
    """
    print("== APACHE-08 ==")
    print(f"Total configurations processed: {count}")
    print(f"Mean performance: {mean:.2f}")
    print(f"Standard deviation: {stddev:.2f}")
    print(f"Worst performance: {worst_perf:.2f}")
    print(f"Top {len(best_list)} best configurations:")
    for rank, (perf, cfg) in enumerate(best_list, 1):
        print(f"  {rank}. {format_config(cfg)} (Performance: {perf:.2f})")
    print()

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main(csv_path):
    # Initialise running statistics
    count = 0
    mean = 0.0
    M2 = 0.0          # sum of squares of differences
    best_list = []    # list of tuples (performance, config)
    worst_perf = None

    # Open CSV and read line by line
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields to float
            cfg = {k: float(v) if v.replace('.', '', 1).isdigit() else v
                   for k, v in row.items()}

            perf = cfg['Performance-']

            # Update worst performance
            if worst_perf is None or perf < worst_perf:
                worst_perf = perf

            # Before updating stats with this row, check for unusually poor
            if count > 0:
                process_row_for_alert(cfg, count, mean, M2)

            # Update running statistics
            count, mean, M2 = update_running_stats(count, mean, M2, perf)

            # Update best configuration shortlist
            update_best_configs(best_list, cfg, (best_list[0][0] if best_list else -math.inf))

    # After all rows processed, compute final stddev
    stddev = running_stddev(count, M2)

    # Print APACHE-08 summary
    print_summary(count, mean, stddev, worst_perf, best_list)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <path_to_csv>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
