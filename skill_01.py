#!/usr/bin/env python3
"""
skill_01.py

Usage:
    python skill_01.py <path-to-auto93.csv>

This script provides:
  CAR-01  : Summary statistics of vehicle weight, fuel efficiency, and acceleration.
  CAR-02  : Ranking of vehicles by a combined score that favors low weight,
            high fuel efficiency and fast acceleration (low Acc+ time).

The output follows the requested format exactly.
"""

import csv
import sys
import math
import numpy as np

# ---------------------------------------------------------------------------

def read_data(csv_path):
    """
    Reads the CSV file and returns a list of dicts with numeric fields.
    Assumes the CSV has no header (or if header is present, it will be ignored).
    Expected columns (in order):
      Clndrs,Volume,HpX,Model,origin,Lbs-,Acc+,Mpg+
    """
    records = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            # Skip empty rows
            if not row or all(cell.strip() == '' for cell in row):
                continue
            try:
                rec = {
                    'Clndrs': int(row[0]),
                    'Volume': int(row[1]),
                    'HpX': int(row[2]),
                    'Model': int(row[3]),
                    'origin': int(row[4]),
                    'Lbs-': float(row[5]),
                    'Acc+': float(row[6]),
                    'Mpg+': float(row[7]),
                }
            except (IndexError, ValueError):
                # Skip rows that cannot be parsed
                continue
            records.append(rec)
    return records

# ---------------------------------------------------------------------------

def compute_stats(values):
    """Return mean, std, min, max for a list of numeric values."""
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))  # sample std
    minv = float(np.min(arr))
    maxv = float(np.max(arr))
    return mean, std, minv, maxv

# ---------------------------------------------------------------------------

def normalize_min_to_max(arr, invert=False):
    """Min–max normalize array to [0, 1].
       If invert=True, higher raw values become lower normalized values."""
    arr = np.array(arr, dtype=float)
    minv, maxv = np.min(arr), np.max(arr)
    if maxv == minv:
        # Avoid division by zero; return zeros
        norm = np.zeros_like(arr)
    else:
        norm = (arr - minv) / (maxv - minv)
    if invert:
        norm = 1.0 - norm
    return norm

# ---------------------------------------------------------------------------

def compute_scores(records):
    """
    Compute a combined score for each record:
      - weight: lower is better -> invert normalization
      - acceleration: lower is better -> invert normalization
      - mpg: higher is better -> normal normalization
    The score is the sum of the three normalized components.
    Returns a list of tuples (score, record).
    """
    weights = [r['Lbs-'] for r in records]
    accs   = [r['Acc+'] for r in records]
    mpgs   = [r['Mpg+'] for r in records]

    norm_w = normalize_min_to_max(weights, invert=True)
    norm_a = normalize_min_to_max(accs,   invert=True)
    norm_m = normalize_min_to_max(mpcs:=mpgs)  # python3.8+ assignment expression

    scores = []
    for idx, rec in enumerate(records):
        score = norm_w[idx] + norm_a[idx] + norm_m[idx]
        scores.append((score, rec))
    return scores

# ---------------------------------------------------------------------------

def print_car_01(records):
    """Print summary statistics for CAR-01."""
    weights = [r['Lbs-'] for r in records]
    mpgs    = [r['Mpg+'] for r in records]
    accs    = [r['Acc+'] for r in records]

    w_mean, w_std, w_min, w_max = compute_stats(weights)
    mpg_mean, mpg_std, mpg_min, mpg_max = compute_stats(mpcs:=mpgs)
    a_mean, a_std, a_min, a_max = compute_stats(accs)

    print("== CAR-01 ==")
    print(f"Weight: mean={w_mean:,.2f} lbs, std={w_std:,.2f}, min={w_min:.0f} lbs, max={w_max:.0f} lbs")
    print(f"Fuel efficiency: mean={mpg_mean:,.2f} mpg, std={mpg_std:,.2f}, min={mpg_min:.1f} mpg, max={mpg_max:.1f} mpg")
    print(f"Acceleration: mean={a_mean:,.2f} sec, std={a_std:,.2f}, min={a_min:.1f} sec, max={a_max:.1f} sec")

# ---------------------------------------------------------------------------

def print_car_02(scores, top_n=5):
    """Print top vehicles for CAR-02."""
    print("\n== CAR-02 ==")
    print(f"Top {top_n} vehicles by combined score (low weight, high mpg, fast Acc+):")
    # Sort scores descending
    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    for i, (score, rec) in enumerate(sorted_scores[:top_n], start=1):
        print(f"{i}) Weight={rec['Lbs-']:.0f} lbs, Acc={rec['Acc+']:.1f} s, Mpg={rec['Mpg+']:.1f} (Score={score:.3f})")

# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-auto93.csv>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    records = read_data(csv_path)

    if not records:
        print("No valid data found in the CSV.", file=sys.stderr)
        sys.exit(1)

    # CAR-01
    print_car_01(records)

    # CAR-02
    scores = compute_scores(records)
    print_car_02(scores, top_n=5)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
