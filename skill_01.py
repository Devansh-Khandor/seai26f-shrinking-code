#!/usr/bin/env python3
"""
Small data‑analysis script for the Auto93 dataset.

This script implements two client requests:

    CAR-01 – Show summary statistics for vehicle weight, fuel efficiency, and acceleration.
    CAR-02 – Rank vehicles on a composite score based on low weight, high mpg, and fast
             acceleration, and list the top 3.

Usage:
    python skill_01.py <path_to_auto93.csv>
"""

import csv
import sys
import statistics
from typing import List, Dict, Any

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def read_csv(path: str) -> List[Dict[str, Any]]:
    """
    Read the CSV file and return a list of dicts with numeric values converted.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for rec in reader:
            # Convert all values to float for numeric processing
            cleaned = {k: float(v) for k, v in rec.items()}
            rows.append(cleaned)
    return rows

def compute_stats(rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    """
    Compute mean, min, and max for a numeric key across all rows.
    """
    values = [row[key] for row in rows]
    return {
        'mean': statistics.mean(values),
        'min': min(values),
        'max': max(values)
    }

def compute_score(row: Dict[str, Any],
                  weight_bounds: Dict[str, float],
                  mpg_bounds: Dict[str, float],
                  accel_bounds: Dict[str, float]) -> float:
    """
    Compute a composite score for a single vehicle.
    Lower weight, higher mpg, lower acceleration time are better.
    The score is the sum of each metric's normalized value (0..1).
    """
    weight_norm = (weight_bounds['max'] - row['Lbs-']) / (weight_bounds['max'] - weight_bounds['min'])
    mpg_norm = (row['Mpg+'] - mpg_bounds['min']) / (mpg_bounds['max'] - mpg_bounds['min'])
    accel_norm = (accel_bounds['max'] - row['Acc+']) / (accel_bounds['max'] - accel_bounds['min'])
    return weight_norm + mpg_norm + accel_norm

def format_float(value: float) -> str:
    """Return a string with up to 2 decimal places."""
    return f"{value:.2f}"

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <path_to_auto93.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        rows = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # ----------------------------------------------------------------------- #
    # CAR-01: Summary statistics
    # ----------------------------------------------------------------------- #
    print("== CAR-01 ==")
    weight_stats = compute_stats(rows, 'Lbs-')
    mpg_stats = compute_stats(rows, 'Mpg+')
    accel_stats = compute_stats(rows, 'Acc+')

    print(f"Weight (lbs): mean={format_float(weight_stats['mean'])}, "
          f"min={format_float(weight_stats['min'])}, "
          f"max={format_float(weight_stats['max'])}")

    print(f"Fuel efficiency (mpg): mean={format_float(mpg_stats['mean'])}, "
          f"min={format_float(mpg_stats['min'])}, "
          f"max={format_float(mpg_stats['max'])}")

    print(f"Acceleration (s): mean={format_float(accel_stats['mean'])}, "
          f"min={format_float(accel_stats['min'])}, "
          f"max={format_float(accel_stats['max'])}")

    # ----------------------------------------------------------------------- #
    # CAR-02: Composite ranking
    # ----------------------------------------------------------------------- #
    print("\n== CAR-02 ==")

    # Precompute bounds for normalization
    weight_bounds = {'min': min(row['Lbs-'] for row in rows),
                     'max': max(row['Lbs-'] for row in rows)}
    mpg_bounds = {'min': min(row['Mpg+'] for row in rows),
                  'max': max(row['Mpg+'] for row in rows)}
    accel_bounds = {'min': min(row['Acc+'] for row in rows),
                    'max': max(row['Acc+'] for row in rows)}

    # Compute scores
    for row in rows:
        row['score'] = compute_score(row, weight_bounds, mpg_bounds, accel_bounds)

    # Sort descending by score
    top_rows = sorted(rows, key=lambda r: r['score'], reverse=True)[:3]

    for i, v in enumerate(top_rows, start=1):
        print(f"{i}. Vehicle: Clndrs={int(v['Clndrs'])}, Volume={int(v['Volume'])}, "
              f"HpX={int(v['HpX'])}, Model={int(v['Model'])}, origin={int(v['origin'])}, "
              f"Lbs-={int(v['Lbs-'])}, Acc+= {format_float(v['Acc+'])}, "
              f"Mpg+= {int(v['Mpg+'])}, score={format_float(v['score'])}")

if __name__ == "__main__":
    main()
