#!/usr/bin/env python3
"""
Apache performance summary and anomaly detection.

Usage:
    python skill_01.py <csv_path>
"""

import sys
import csv
import statistics
from typing import List, Dict, Any, Tuple


def load_csv(path: str) -> List[Dict[str, Any]]:
    """
    Load the CSV file and return a list of dictionaries with numeric values
    converted to int where possible.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row: Dict[str, Any] = {}
            for k, v in raw.items():
                # try to convert everything to int (performance and config flags)
                try:
                    row[k] = int(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return rows


def compute_summary(rows: List[Dict[str, Any]]) -> Tuple[float, int, int]:
    """
    Compute typical (mean), best (max), and worst (min) performance values.
    Returns (mean, best, worst)
    """
    performances = [row['Performance-'] for row in rows]
    mean_perf = statistics.mean(performances) if performances else 0
    best_perf = max(performances) if performances else 0
    worst_perf = min(performances) if performances else 0
    return mean_perf, best_perf, worst_perf


def find_favorable_configs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Identify configurations with unusually favorable performance values.
    Here we consider a configuration favorable if its performance is at least
    one standard deviation above the mean.
    """
    if not rows:
        return []

    performances = [row['Performance-'] for row in rows]
    mean_perf = statistics.mean(performances)
    stdev_perf = statistics.stdev(performances) if len(performances) > 1 else 0
    threshold = mean_perf + stdev_perf  # one std dev above mean

    favorable = [row for row in rows if row['Performance-'] >= threshold]
    return favorable


def format_config(row: Dict[str, Any]) -> str:
    """
    Return a formatted string representation of a configuration row.
    """
    # Exclude the performance column from the config representation
    cfg_items = [f"{k}={v}" for k, v in row.items() if k != 'Performance-']
    return "{" + ", ".join(cfg_items) + "}"


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)

    # APACHE-01: summary
    print("== APACHE-01 ==")
    mean_perf, best_perf, worst_perf = compute_summary(rows)
    print(f"Typical performance: {mean_perf:.2f}")
    print(f"Best performance: {best_perf}")
    print(f"Worst performance: {worst_perf}")

    # APACHE-02: unusually favorable configurations
    print("\n== APACHE-02 ==")
    favorable = find_favorable_configs(rows)
    if not favorable:
        print("No unusually favorable configurations found.")
    else:
        print(f"Found {len(favorable)} unusually favorable configuration(s):")
        for cfg in favorable:
            print(f"- Config: {format_config(cfg)}")
            print(f"  Performance: {cfg['Performance-']}")


if __name__ == "__main__":
    main()
