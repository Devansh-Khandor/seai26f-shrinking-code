#!/usr/bin/env python3
"""
skill_04.py

Usage:
    python skill_04.py <csv_path>

Prints:
    CAR-07: Immediate desirability assessment for each new vehicle.
    CAR-08: Updated inventory summaries and top‑5 shortlist.
"""

import csv
import sys
import math
import argparse
from statistics import median
from collections import defaultdict

# ----------------------------------------------------------------------
# Data ingestion
# ----------------------------------------------------------------------
def parse_csv(path):
    """Read CSV and return list of rows with numeric conversion."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                # Numeric columns
                row['Clndrs'] = int(row['Clndrs'])
                row['Volume'] = float(row['Volume'])
                row['HpX']    = float(row['HpX'])
                row['Model']  = int(row['Model'])
                row['origin'] = int(row['origin'])
                row['Lbs-']   = float(row['Lbs-'])
                row['Acc+']   = float(row['Acc+'])
                row['Mpg+']   = float(row['Mpg+'])
                row['_idx']   = i
                rows.append(row)
            except (ValueError, KeyError):
                # Skip invalid rows
                continue
    return rows

# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def compute_stats(values):
    n = len(values)
    if n == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    mean = sum(values) / n
    med = median(values)
    var = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(var)
    mn = min(values)
    mx = max(values)
    return mean, med, std, mn, mx

def z_score(x, mean, std):
    return (x - mean) / std if std != 0 else 0.0

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path):
    rows = parse_csv(csv_path)
    if not rows:
        print("No valid data found.", file=sys.stderr)
        sys.exit(1)

    # Extract columns
    weights = [r['Lbs-'] for r in rows]
    mpgs    = [r['Mpg+'] for r in rows]
    accs    = [r['Acc+'] for r in rows]

    # Compute overall statistics
    w_mean, w_med, w_std, w_min, w_max = compute_stats(weights)
    m_mean, m_med, m_std, m_min, m_max = compute_stats(mpgs)
    a_mean, a_med, a_std, a_min, a_max = compute_stats(accs)

    # Compute combined desirability score for each row
    for r in rows:
        wz = z_score(r['Lbs-'], w_mean, w_std)
        mz = z_score(r['Mpg+'], m_mean, m_std)
        az = z_score(r['Acc+'], a_mean, a_std)
        r['score'] = -wz + mz - az

    # Sort by score descending
    sorted_rows = sorted(rows, key=lambda x: x['score'], reverse=True)
    top_n = 5
    top_rows = sorted_rows[:top_n]
    top_ids = {r['_idx'] for r in top_rows}

    # ================= CAR-07 =================
    print("== CAR-07 ==")
    print("Immediate desirability assessment (top 5 vehicles):")
    for i, r in enumerate(top_rows, 1):
        print(f"{i}. Vehicle #{r['_idx']} (ID={r['Clndrs']}) -> Score: {r['score']:.3f}")
    print()
    # Provide at least three lines of further insight
    # Show 3 non‑top vehicles (if available)
    non_top = [r for r in rows if r['_idx'] not in top_ids]
    if non_top:
        print("Three example vehicles not in top 5:")
        for i, r in enumerate(non_top[:3], 1):
            print(f"{i}. Vehicle #{r['_idx']} (ID={r['Clndrs']}) -> Score: {r['score']:.3f}")
    else:
        print("(No vehicles outside top 5 to display)")

    # ================= CAR-08 =================
    print("\n== CAR-08 ==")
    print("Inventory summaries and updated shortlist:")
    print(f"Weight - mean: {w_mean:.2f} lbs, median: {w_med:.2f} lbs, std: {w_std:.2f} lbs, min: {w_min:.2f} lbs, max: {w_max:.2f} lbs")
    print(f"MPG   - mean: {m_mean:.2f} MPG, median: {m_med:.2f} MPG, std: {m_std:.2f} MPG, min: {m_min:.2f} MPG, max: {m_max:.2f} MPG")
    print(f"Acc   - mean: {a_mean:.2f} s, median: {a_med:.2f} s, std: {a_std:.2f} s, min: {a_min:.2f} s, max: {a_max:.2f} s")
    print("\nTop {0} desirable vehicles (by combined score):".format(top_n))
    for i, r in enumerate(top_rows, 1):
        print(f"{i}. Vehicle #{r['_idx']} (ID={r['Clndrs']}) "
              f"Weight={r['Lbs-']:.1f} lbs, MPG={r['Mpg+']:.1f}, Acc={r['Acc+']:.1f}s, "
              f"Score={r['score']:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inventory desirability and summary analysis")
    parser.add_argument("csv_path", help="Path to auto93.csv")
    args = parser.parse_args()
    main(args.csv_path)
