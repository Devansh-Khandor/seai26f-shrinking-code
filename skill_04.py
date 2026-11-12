#!/usr/bin/env python3
"""
Skill 04 – Apache real‑time alerting and dynamic summary

This script implements two new requests:

* APACHE-07: As new benchmark results arrive, immediately assess each
  configuration and alert when its measured performance is unusually
  poor compared with the rest of the data set.
* APACHE-08: Keep performance summaries and a recommended‑configuration
  shortlist updated each time the script is run, without requiring a full
  manual restart.

The implementation re‑uses utilities from skill_01.py wherever
possible.  No third‑party dependencies beyond the standard library,
NumPy and scikit‑learn (used in skill_02) are required.
"""

import sys
import argparse
import numpy as np

# ----------------------------------------------------------------------
# Import utilities from skill_01
# ----------------------------------------------------------------------
try:
    from skill_01 import (
        read_csv,
        find_performance_column,
        compute_stats,
    )
except Exception:
    # Fallback minimal implementations if skill_01 is missing
    import csv
    from statistics import mean, median

    def read_csv(path: str):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            rows = []
            for row in reader:
                int_row = {}
                for k, v in row.items():
                    try:
                        int_row[k] = int(v)
                    except ValueError:
                        int_row[k] = v
                rows.append(int_row)
        return header, rows

    def find_performance_column(header):
        for col in header:
            if col.lower().startswith("performance"):
                return col
        raise KeyError("No performance column found in header.")

    def compute_stats(performance_values):
        from statistics import mean, median
        import numpy as np
        return {
            "mean": mean(performance_values),
            "median": median(performance_values),
            "std": np.std(performance_values, ddof=0),
            "best": min(performance_values),
            "worst": max(performance_values),
        }

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def alert_poor_performance(rows, perf_col):
    """
    For each configuration that has unusually poor performance (>= 90th
    percentile), print an alert line.
    """
    perf_vals = np.array([row[perf_col] for row in rows])
    threshold = np.percentile(perf_vals, 90)

    alerts = []
    for idx, row in enumerate(rows):
        perf = row[perf_col]
        if perf >= threshold:
            alerts.append((idx, perf, row, threshold))

    # Print results
    print("== APACHE-07 ==")
    if not alerts:
        print("No unusually poor performances detected.")
        return

    print(f"Threshold for poor performance: {threshold:.0f}")
    print(f"Found {len(alerts)} alertable configuration(s):")
    for idx, perf, row, _ in alerts:
        # Show key settings for clarity
        key_settings = {k: row[k] for k in ("hostnameLookups", "keepAlive",
                                            "enableSendfile", "followSymLinks",
                                            "accessLog", "extendedStatus",
                                            "inMemory", "handle")}
        print(f"  - idx={idx}, perf={perf} ms, settings={key_settings}")


def update_summary_and_shortlist(rows, perf_col, shortlist_size=5):
    """
    Compute overall performance statistics and produce a shortlist of
    the best configurations.
    """
    perf_vals = np.array([row[perf_col] for row in rows])
    stats = compute_stats(perf_vals)

    print("== APACHE-08 ==")
    print(f"Median performance: {stats['median']:.0f} ms")
    print(f"Best (min) performance: {stats['best']:.0f} ms")
    print(f"Worst (max) performance: {stats['worst']:.0f} ms")
    print(f"Standard deviation: {stats['std']:.0f} ms\n")

    # Shortlist the best configurations
    sorted_rows = sorted(rows, key=lambda r: r[perf_col])
    print(f"Top {shortlist_size} best configurations:")
    for rank, row in enumerate(sorted_rows[:shortlist_size], start=1):
        perf = row[perf_col]
        # Show all configuration columns except performance
        config = {k: v for k, v in row.items() if k != perf_col}
        print(f"  {rank}. perf={perf} ms, config={config}")


# ----------------------------------------------------------------------
# Main skill logic
# ----------------------------------------------------------------------
def main(csv_path: str):
    header, rows = read_csv(csv_path)
    if not rows:
        print("No data found in CSV.")
        return

    perf_col = find_performance_column(header)

    # APACHE-07: alert on poor performance
    alert_poor_performance(rows, perf_col)

    print()  # blank line between skills

    # APACHE-08: summary and shortlist
    update_summary_and_shortlist(rows, perf_col)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apache alerting and dynamic summary skill."
    )
    parser.add_argument(
        "csv_path",
        help="Path to Apache_AllMeasurements.csv",
    )
    args = parser.parse_args()

    try:
        main(args.csv_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
