#!/usr/bin/env python3
"""
Skill 04: Incremental assessment of new benchmark results and dynamic summary updates.

Usage:
    python skill_04.py <csv_path>
"""

import sys
from pathlib import Path
import csv

# Re-use helpers from earlier skills
from skill_01 import (
    read_csv,
    compute_statistics,
    find_distinguishing_settings,
    format_config,
    best_worst_configs,
    top_percentile_threshold,
)

from skill_02 import standard_deviation


def assess_new_configurations(rows, performance):
    """
    Iterate over the rows as if they arrive sequentially.
    For each configuration, compare its performance to the
    bottom 10 % of all previously seen configurations.
    Return a list of alerts: (index, row, perf, threshold).
    """
    alerts = []
    prev_perf = []

    for idx, (row, perf) in enumerate(zip(rows, performance)):
        if prev_perf:
            # Compute threshold for the worst 10 % of the previous data
            threshold = top_percentile_threshold(prev_perf, percent=10)
            if perf <= threshold:
                alerts.append((idx, row, perf, threshold))
        # Update the history
        prev_perf.append(perf)

    return alerts


def print_apache_07(alerts):
    """Print results for APACHE-07."""
    print("== APACHE-07 ==")
    if not alerts:
        print("No unusually poor configurations detected.")
    else:
        print(f"Number of alerts: {len(alerts)}")
        # Show first 5 alerts for reference
        print("Sample alerts (first 5):")
        for i, (idx, row, perf, thresh) in enumerate(alerts[:5], 1):
            print(f"  Alert {i}: Config {idx+1} with performance {perf:.2f} ms "
                  f"is below threshold {thresh:.2f} ms")
    print()


def print_apache_08(rows, performance, top_rows):
    """Print updated summaries for APACHE-08."""
    print("== APACHE-08 ==")

    # Summary statistics
    mean, median, min_perf, max_perf = compute_statistics(performance)
    std_perf = standard_deviation(performance)

    print(f"Total configurations examined: {len(rows)}")
    print(f"Mean performance: {mean:.2f} ms")
    print(f"Median performance: {median:.2f} ms")
    print(f"Min performance: {min_perf:.2f} ms")
    print(f"Max performance: {max_perf:.2f} ms")
    print(f"Std deviation: {std_perf:.2f} ms")

    # Recommended shortlist
    print(f"\nTop 10% configurations (threshold >= {top_percentile_threshold(performance, percent=90):.2f} ms):")
    print(f"Number of top configs: {len(top_rows)}")
    print("Sample top configs (first 5):")
    for i, row in enumerate(top_rows[:5], 1):
        print(f"  {i}. {format_config(row)}")

    # Distinguishing settings
    distinguishing = find_distinguishing_settings(rows, top_rows)
    if distinguishing:
        print("\nDistinguishing settings among top 10%:")
        for col, (top_val, mode_val) in distinguishing.items():
            print(f"  - {col}: always '{top_val}' vs majority '{mode_val}'")
    else:
        print("\nNo distinguishing settings identified.")

    print()


def main(csv_path_str):
    csv_path = Path(csv_path_str)
    if not csv_path.is_file():
        print(f"Error: file '{csv_path}' not found.", file=sys.stderr)
        sys.exit(1)

    rows, performance = read_csv(csv_path)

    # APACHE-07: Assess each configuration as it arrives
    alerts = assess_new_configurations(rows, performance)
    print_apache_07(alerts)

    # APACHE-08: Updated performance summaries and recommended shortlist
    threshold = top_percentile_threshold(performance, percent=90)
    top_rows = [row for row, perf in zip(rows, performance) if perf >= threshold]
    print_apache_08(rows, performance, top_rows)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
