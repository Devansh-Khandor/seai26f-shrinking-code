#!/usr/bin/env python3
"""
Skill 01: Inventory summary and combined‑score ranking for used‑car dealership.

Usage:
    python skill_01.py <csv_path>
"""

import csv
import sys
import statistics

def read_data(csv_path):
    """Read the CSV and return a list of rows with numeric fields converted."""
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            try:
                row['Lbs-']   = float(row['Lbs-'])
                row['Acc+']   = float(row['Acc+'])
                row['Mpg+']   = float(row['Mpg+'])
            except (ValueError, KeyError):
                # Skip rows with bad data
                continue
            rows.append(row)
    return rows

def compute_stats(rows):
    """Return mean, min and max for weight, mpg and acceleration."""
    weights = [r['Lbs-'] for r in rows]
    mpgs   = [r['Mpg+'] for r in rows]
    accs   = [r['Acc+'] for r in rows]

    stats = {
        'weight_mean': statistics.mean(weights),
        'weight_min':  min(weights),
        'weight_max':  max(weights),
        'mpg_mean':    statistics.mean(mpcs := mpgs),
        'mpg_min':     min(mpcs),
        'mpg_max':     max(mpcs),
        'acc_mean':    statistics.mean(accs),
        'acc_min':     min(accs),
        'acc_max':     max(accs)
    }
    return stats

def score_rows(rows, stats):
    """Add a combined score to each row based on weight, mpg and acceleration."""
    wmin, wmax = stats['weight_min'], stats['weight_max']
    mmin, mmax = stats['mpg_min'], stats['mpg_max']
    amin, amax = stats['acc_min'], stats['acc_max']

    for r in rows:
        # Normalized components: lower weight is better, higher mpg is better, lower acceleration is better
        if wmax != wmin:
            w_score = (wmax - r['Lbs-']) / (wmax - wmin)
        else:
            w_score = 1.0
        if mmax != mmin:
            m_score = (r['Mpg+'] - mmin) / (mmax - mmin)
        else:
            m_score = 1.0
        if amax != amin:
            a_score = (amax - r['Acc+']) / (amax - amin)
        else:
            a_score = 1.0

        r['overall_score'] = w_score + m_score + a_score

    # Sort by descending score
    rows_sorted = sorted(rows, key=lambda r: r['overall_score'], reverse=True)
    return rows_sorted

def main(csv_path):
    rows = read_data(csv_path)
    if not rows:
        print("No valid data found in CSV.")
        return

    stats = compute_stats(rows)

    # CAR-01: Inventory summary
    print("== CAR-01 ==")
    print(f"Weight: mean {stats['weight_mean']:.2f}, min {stats['weight_min']:.2f}, max {stats['weight_max']:.2f}")
    print(f"Fuel Efficiency: mean {stats['mpg_mean']:.2f}, min {stats['mpg_min']:.2f}, max {stats['mpg_max']:.2f}")
    print(f"Acceleration: mean {stats['acc_mean']:.2f}, min {stats['acc_min']:.2f}, max {stats['acc_max']:.2f}")

    # CAR-02: Combined‑score ranking
    ranked_rows = score_rows(rows, stats)
    print("\n== CAR-02 ==")
    for i, r in enumerate(ranked_rows[:3], start=1):
        print(f"{i}. Lbs- {r['Lbs-']:.1f}, Mpg+ {r['Mpg+']:.1f}, Acc+ {r['Acc+']:.1f}, Score {r['overall_score']:.3f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
