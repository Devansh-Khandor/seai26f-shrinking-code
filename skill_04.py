#!/usr/bin/env python3
"""
skill_04.py
-----------

This script implements two new Apache data‑analysis requests:

  APACHE-07: Detect unusually poor configurations as new benchmark
             measurements arrive, and alert the administrator.
  APACHE-08: Maintain updated performance summaries and a short
             recommended‑configuration list as more data becomes
             available, without requiring a full restart.

Usage:
    python skill_04.py <path_to_Apache_AllMeasurements.csv>
"""

import sys
import csv
import numpy as np

# --------------------------------------------------------------------------- #
# Utility: read CSV and iterate in order
# --------------------------------------------------------------------------- #
def read_csv_iter(csv_path):
    """
    Generator yielding (row_index, feature_row, perf_value).

    Reads the CSV line by line, preserving the original order.
    """
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header:
            raise ValueError("Empty CSV header")

        feature_names = header[:-1]

        for i, row in enumerate(reader):
            if len(row) != len(header):
                continue
            try:
                feat = [int(float(v)) for v in row[:-1]]
                perf = float(row[-1])
            except ValueError:
                continue
            yield i, feat, perf

# --------------------------------------------------------------------------- #
# APACHE‑07 : Real‑time alert on unusually poor performance
# --------------------------------------------------------------------------- #
def apache_07_alerts(csv_path):
    """
    Scan the file in order, keeping a rolling list of performance values.
    For each new measurement, compare it to the median and standard
    deviation of all previous measurements and generate an alert if
    it is unusually poor.
    """
    y_prev = []
    alerts = []

    for idx, feat, perf in read_csv_iter(csv_path):
        if y_prev:
            median = np.median(y_prev)
            std = np.std(y_prev, ddof=1) if len(y_prev) > 1 else 0.0
            threshold = median - 0.5 * std if std > 0 else median
            if perf < threshold:
                alerts.append(
                    f"Record {idx}: performance {perf:.2f} "
                    f"is unusually poor (median={median:.2f}, std={std:.2f})"
                )
        y_prev.append(perf)

    return alerts

# --------------------------------------------------------------------------- #
# APACHE‑08 : Updated summaries & recommendation shortlist
# --------------------------------------------------------------------------- #
def apache_08_summary_and_recommendation(csv_path):
    """
    After all records are read, compute:
        • median, best (minimum), worst (maximum) performance
        • a shortlist of configurations that are within 5 % of the best
    """
    X = []
    y = []

    for idx, feat, perf in read_csv_iter(csv_path):
        X.append(feat)
        y.append(perf)

    if not y:
        return None, None

    y_arr = np.array(y)
    median = np.median(y_arr)
    best = np.min(y_arr)
    worst = np.max(y_arr)

    # Recommended configs: performance within 5 % of the best
    range_ = worst - best
    cutoff = best + 0.05 * range_
    rec_indices = [i for i, val in enumerate(y) if val <= cutoff]

    summary = {
        "median": median,
        "best": best,
        "worst": worst,
    }

    recommendations = [
        (i, y[i]) for i in rec_indices
    ]

    return summary, recommendations

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # -------------------- APACHE-07 -------------------- #
    alerts = apache_07_alerts(csv_path)
    print("== APACHE-07 ==")
    if alerts:
        print(f"Number of alerts: {len(alerts)}")
        # print first three alerts as required (at least three lines)
        for a in alerts[:3]:
            print(a)
        if len(alerts) > 3:
            print(f"... and {len(alerts)-3} more alerts.")
    else:
        print("No unusually poor configurations detected.")

    # -------------------- APACHE-08 -------------------- #
    summary, recs = apache_08_summary_and_recommendation(csv_path)
    print("\n== APACHE-08 ==")
    if summary is None:
        print("No data available.")
        return

    print(f"Median performance: {summary['median']:.2f}")
    print(f"Best performance: {summary['best']:.2f}")
    print(f"Worst performance: {summary['worst']:.2f}")

    if recs:
        print("Recommended configuration indices (within 5% of best):")
        # Show up to first 5 recommendations
        for idx, perf in recs[:5]:
            print(f"  Index {idx}: performance {perf:.2f}")
        if len(recs) > 5:
            print(f"... and {len(recs)-5} more.")
    else:
        print("No configurations within 5% of the best performance.")

if __name__ == "__main__":
    main()
