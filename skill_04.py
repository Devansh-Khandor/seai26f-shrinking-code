#!/usr/bin/env python3
"""
skill_04.py – Real‑time anomaly detection and incremental summary maintenance.

Usage:
    python skill_04.py <path-to-Apache_AllMeasurements.csv>
"""

import sys
import math
import skill_01 as s1   # Reuse utilities from the first skill

# ----------------------------------------------------------------------
# Helper for incremental anomaly detection
# ----------------------------------------------------------------------
def alert_unusually_poor(rows, headers, z_thresh=2.0):
    """
    Iterate over the rows in order, keeping a running mean and std of the
    performance values seen so far. When a new configuration's performance
    falls below mean - z_thresh * std, yield an alert.
    """
    perf_values = []
    alerts = []

    for idx, row in enumerate(rows):
        perf = int(row['Performance-'])
        if len(perf_values) >= 2:  # need at least two to compute std
            mean = sum(perf_values) / len(perf_values)
            std = math.sqrt(sum((x - mean) ** 2 for x in perf_values) / len(perf_values))
            threshold = mean - z_thresh * std
            if perf <= threshold:
                alerts.append((idx + 1, perf, threshold))

        perf_values.append(perf)

    return alerts

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main(csv_path):
    # Load data once (the incremental logic is simulated over this list)
    rows, headers = s1.load_csv(csv_path)

    # --------------------------------------------------------------
    # APACHE-07: Real‑time anomaly detection
    # --------------------------------------------------------------
    print("== APACHE-07 ==")
    alerts = alert_unusually_poor(rows, headers)

    if alerts:
        for num, perf, thresh in alerts:
            print(f"  Alert: Configuration #{num} has unusually poor performance "
                  f"({perf}) below threshold {thresh:.2f}.")
    else:
        # Provide at least three lines of output even if there are no alerts
        print("  No unusually poor configurations detected.")
        print(f"  Total configurations processed: {len(rows)}")
        print("  Thresholds were computed for each configuration based on prior data.")

    # --------------------------------------------------------------
    # APACHE-08: Incremental summary & recommended shortlist
    # --------------------------------------------------------------
    print("\n== APACHE-08 ==")

    # Compute summary statistics for the entire dataset
    perf_values = [int(row['Performance-']) for row in rows]
    median, best, worst = s1.compute_summary(perf_values)
    print(f"  Median performance: {median:.2f}")
    print(f"  Best performance (min): {best:.2f}")
    print(f"  Worst performance (max): {worst:.2f}")

    # Identify favorable configurations (those significantly better than the mean)
    fav_rows, threshold = s1.find_favorable(rows, perf_values)
    if fav_rows:
        print("\n  Recommended configurations (favorable performance):")
        for idx, row in enumerate(fav_rows, start=1):
            perf = int(row['Performance-'])
            diff = s1.distinguishing_settings(row, rows, headers)
            print(f"\n  Config #{idx} (Performance={perf}):")
            if diff:
                print("  Distinguishing settings:")
                for k, v in diff.items():
                    print(f"    {k}: {v}")
            else:
                print("  No distinguishing settings found.")
    else:
        print("\n  No configurations exhibit unusually favorable performance.")

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python skill_04.py <csv_path>")
    main(sys.argv[1])
