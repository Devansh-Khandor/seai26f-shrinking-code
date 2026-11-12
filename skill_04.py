#!/usr/bin/env python3
"""
Apache performance monitoring and incremental analysis.

This script extends the earlier skill modules with two new analysis points:

- APACHE-07: When a new benchmark record arrives, it immediately checks if
  its performance is unusually poor compared with all previously tested
  configurations.  An alert is printed for each such configuration.
- APACHE-08: Keeps performance summaries (mean, best, worst) and a
  recommended configuration shortlist up to date as new records are processed.
  The results are printed after all records have been examined.

Usage:
    python skill_04.py <csv_path>
"""

import sys
import statistics
import skill_01

def load_csv(path: str):
    """Delegate to skill_01's load_csv."""
    return skill_01.load_csv(path)

def alert_unusually_poor(rows):
    """
    Iterate through rows sequentially and alert on configurations whose
    performance is lower than (mean of previous records - stddev of previous records).
    Returns a list of tuples: (row, mean_prev, std_prev)
    """
    prev_perfs = []
    alerts = []

    for idx, row in enumerate(rows):
        # Skip the very first record: no previous data to compare to
        if idx == 0:
            prev_perfs.append(row['Performance-'])
            continue

        # Compute statistics of all previous configurations
        mean_prev = statistics.mean(prev_perfs)
        std_prev = statistics.stdev(prev_perfs) if len(prev_perfs) > 1 else 0.0
        threshold = mean_prev - std_prev  # unusual poor threshold

        if row['Performance-'] < threshold:
            alerts.append((row, mean_prev, std_prev))

        prev_perfs.append(row['Performance-'])

    return alerts

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)

    # ---------- APACHE-07 ----------
    print("== APACHE-07 ==")
    alerts = alert_unusually_poor(rows)
    if not alerts:
        print("No unusually poor configurations found.")
    else:
        for row, mean_prev, std_prev in alerts:
            # Print at least three lines of detail
            print(f"- Config: {skill_01.format_config(row)}")
            print(f"  Performance: {row['Performance-']}")
            print(f"  Previous mean: {mean_prev:.2f}, stddev: {std_prev:.2f}")

    # ---------- APACHE-08 ----------
    print("\n== APACHE-08 ==")
    # Performance summary
    mean_perf, best_perf, worst_perf = skill_01.compute_summary(rows)
    print(f"Typical performance (mean): {mean_perf:.2f}")
    print(f"Best performance: {best_perf}")
    print(f"Worst performance: {worst_perf}")

    # Recommended configurations (unusually favorable)
    favorable = skill_01.find_favorable_configs(rows)
    if not favorable:
        print("No unusually favorable configurations found.")
    else:
        print(f"Found {len(favorable)} unusually favorable configuration(s):")
        for cfg in favorable:
            print(f"- Config: {skill_01.format_config(cfg)}")
            print(f"  Performance: {cfg['Performance-']}")

if __name__ == "__main__":
    main()
