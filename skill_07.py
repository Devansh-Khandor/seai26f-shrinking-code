#!/usr/bin/env python3
"""
skill_07.py

Usage:
    python skill_07.py <csv_path>

Prints:
    CAR-13: Identify a vehicle with a strong balance of fuel efficiency,
            weight, and acceleration while evaluating as few candidates as
            reasonably possible.
    CAR-14: Recommend vehicles that should be inspected first to discover
            promising inventory options early, given a limited inspection budget.
"""

import csv
import sys
import math
from statistics import mean, median, stdev

# ----------------------------------------------------------------------
# Data ingestion
# ----------------------------------------------------------------------
def read_csv(path):
    """
    Read CSV and return a list of dicts with numeric conversion.
    Skip rows with any missing or invalid numeric field.
    """
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                row['Clndrs'] = int(row['Clndrs'])
                row['Volume'] = float(row['Volume'])
                row['HpX']    = float(row['HpX'])
                row['Model']  = int(row['Model'])
                row['origin'] = int(row['origin'])
                row['Lbs-']   = float(row['Lbs-'])
                row['Acc+']   = float(row['Acc+'])
                row['Mpg+']   = float(row['Mpg+'])
                row['_idx']   = idx
                rows.append(row)
            except (ValueError, KeyError):
                # Skip rows with missing/invalid data
                continue
    return rows

# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def z_score(x, mean_val, std_val):
    """Standard score, safe division."""
    return (x - mean_val) / std_val if std_val != 0 else 0.0

# ----------------------------------------------------------------------
# Core logic
# ----------------------------------------------------------------------
def main(csv_path):
    rows = read_csv(csv_path)
    if not rows:
        print("No valid data found.", file=sys.stderr)
        sys.exit(1)

    # Collect numeric columns
    weights = [r['Lbs-'] for r in rows]
    mpgs    = [r['Mpg+'] for r in rows]
    accs    = [r['Acc+'] for r in rows]

    # Compute overall statistics for z-scores
    w_mean, w_std = mean(weights), stdev(weights) if len(weights) > 1 else 0.0
    m_mean, m_std = mean(mpgs),  stdev(mpgs)  if len(mpgs)  > 1 else 0.0
    a_mean, a_std = mean(accs),  stdev(accs)  if len(accs)  > 1 else 0.0

    # Compute combined desirability score for each vehicle
    for r in rows:
        wz = z_score(r['Lbs-'], w_mean, w_std)
        mz = z_score(r['Mpg+'], m_mean, m_std)
        az = z_score(r['Acc+'], a_mean, a_std)
        # Lower weight (negative), higher mpg (positive), lower acceleration (negative)
        r['score'] = -wz + mz - az

    # Sort vehicles by descending score
    sorted_rows = sorted(rows, key=lambda r: r['score'], reverse=True)

    # ---------------------------
    # CAR-13: Balanced vehicle
    # ---------------------------
    print("== CAR-13 ==")
    # Pick top vehicle that balances all three metrics
    top = sorted_rows[0]
    print(f"1. Vehicle #{top['_idx']} (ID={top['Clndrs']})")
    print(f"   Score: {top['score']:.3f}")
    print(f"   Weight: {top['Lbs-']:.1f} lbs")
    print(f"   MPG: {top['Mpg+']:.1f} MPG")
    print(f"   Acceleration: {top['Acc+']:.1f} s")
    # Show two additional candidates to illustrate the “few” evaluation principle
    print("\nAdditional candidate vehicles:")
    for i, r in enumerate(sorted_rows[1:4], 2):
        print(f"{i}. Vehicle #{r['_idx']} (ID={r['Clndrs']})")
        print(f"   Score: {r['score']:.3f}")
        print(f"   Weight: {r['Lbs-']:.1f} lbs")
        print(f"   MPG: {r['Mpg+']:.1f} MPG")
        print(f"   Acceleration: {r['Acc+']:.1f} s")

    # ---------------------------
    # CAR-14: Inspection priority
    # ---------------------------
    print("\n== CAR-14 ==")
    print("Vehicles to inspect first (top 5 by balanced score):")
    for i, r in enumerate(sorted_rows[:5], 1):
        print(f"{i}. Vehicle #{r['_idx']} (ID={r['Clndrs']}) - Score: {r['score']:.3f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
