#!/usr/bin/env python3
"""
Skill 01: Apache performance summaries and distinguishing configurations.

Usage:
    python skill_01.py <csv_path>
"""

import csv
import sys
import math
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(csv_path: Path):
    """Read CSV into a list of dicts and collect performance values."""
    rows = []
    performance = []

    with csv.DictReader(open(csv_path, newline='')) as reader:
        for r in reader:
            # Convert performance column to float
            perf_val = float(r.get("Performance-") or 0)
            performance.append(perf_val)
            rows.append(r)

    return rows, performance


def compute_statistics(performance):
    """Return mean, median, min, max of performance values."""
    n = len(performance)
    if n == 0:
        return (0, 0, 0, 0)

    mean = sum(performance) / n
    sorted_perf = sorted(performance)
    if n % 2:
        median = sorted_perf[n // 2]
    else:
        median = (sorted_perf[n // 2 - 1] + sorted_perf[n // 2]) / 2

    return mean, median, min(performance), max(performance)


def best_worst_configs(rows, performance):
    """Return best and worst configuration rows."""
    if not rows:
        return None, None

    best_idx = performance.index(max(performance))
    worst_idx = performance.index(min(performance))
    return rows[best_idx], rows[worst_idx]


def top_percentile_threshold(performance, percent=90):
    """Return performance threshold for top percentile."""
    if not performance:
        return 0
    sorted_perf = sorted(performance)
    idx = int(math.ceil(len(sorted_perf) * percent / 100.0)) - 1
    idx = max(0, min(idx, len(sorted_perf) - 1))
    return sorted_perf[idx]


def find_distinguishing_settings(rows, top_rows):
    """
    Find settings that are constant within top_rows and differ from the overall mode.
    Returns a dict of column -> (top_value, mode_value).
    """
    if not rows or not top_rows:
        return {}

    # Determine overall mode for each column
    overall_modes = {}
    for key in rows[0].keys():
        if key == "Performance-":
            continue
        values = [r[key] for r in rows]
        most_common = Counter(values).most_common(1)[0][0]
        overall_modes[key] = most_common

    # Check top rows
    distinguishing = {}
    for key in rows[0].keys():
        if key == "Performance-":
            continue
        top_values = {r[key] for r in top_rows}
        if len(top_values) == 1:
            top_value = next(iter(top_values))
            if top_value != overall_modes[key]:
                distinguishing[key] = (top_value, overall_modes[key])
    return distinguishing


def format_config(conf):
    """Return a readable string representation of a configuration."""
    # Exclude Performance- column
    return ", ".join(f"{k}={v}" for k, v in conf.items() if k != "Performance-")


def main(csv_path_str):
    csv_path = Path(csv_path_str)
    if not csv_path.is_file():
        print(f"Error: file '{csv_path}' not found.", file=sys.stderr)
        sys.exit(1)

    rows, performance = read_csv(csv_path)

    # === APACHE-01 ===
    print("== APACHE-01 ==")
    num_configs = len(rows)
    mean, median, min_perf, max_perf = compute_statistics(performance)
    best_conf, worst_conf = best_worst_configs(rows, performance)

    print(f"Configurations examined: {num_configs}")
    print(f"Typical performance (mean): {mean:.2f}")
    if best_conf:
        print(f"Best performance: {max_perf:.2f} ({format_config(best_conf)})")
    if worst_conf:
        print(f"Worst performance: {min_perf:.2f} ({format_config(worst_conf)})")
    print()

    # === APACHE-02 ===
    print("== APACHE-02 ==")

    # Determine threshold for top 10% favorable performance
    threshold = top_percentile_threshold(performance, percent=90)
    top_rows = [row for row, perf in zip(rows, performance) if perf >= threshold]
    num_top = len(top_rows)

    print(f"Top 10% configurations with unusually favorable performance (threshold >= {threshold:.2f})")
    print(f"Number of top configs: {num_top}")

    distinguishing = find_distinguishing_settings(rows, top_rows)
    if distinguishing:
        print("Distinguishing settings:")
        for col, (top_val, mode_val) in distinguishing.items():
            print(f" - {col}: always '{top_val}' vs majority '{mode_val}'")
    else:
        print("No distinguishing settings identified.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
