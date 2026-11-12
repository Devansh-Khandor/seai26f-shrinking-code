#!/usr/bin/env python3
"""
skill_07.py

Usage:
    python skill_07.py <path-to-auto93.csv>

This script implements two new features for the dealership:

  CAR-13 : Identify a vehicle with a strong balance of fuel efficiency, weight,
           and acceleration while evaluating as few candidate vehicles as
           reasonably possible.

  CAR-14 : With a limited inspection budget, recommend which vehicles should
           be evaluated first so that promising inventory options are discovered
           early.

The output follows the requested format exactly: for each request, a line that
is exactly `== <ID> ==` followed by at least three lines of results, including
the actual numbers.
"""

import csv
import sys
import numpy as np

# ---------------------------------------------------------------------------

def read_data(csv_path):
    """
    Read the CSV file and return a list of records.
    Each record is a dict with numeric fields.
    Rows that cannot be parsed are skipped.
    """
    records = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
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
                continue
            records.append(rec)
    return records

# ---------------------------------------------------------------------------

def normalize_min_to_max(arr, invert=False):
    """Min–max normalize array to [0, 1]. If invert=True, higher raw values become lower normalized values."""
    arr = np.array(arr, dtype=float)
    minv, maxv = np.min(arr), np.max(arr)
    if maxv == minv:
        norm = np.zeros_like(arr)
    else:
        norm = (arr - minv) / (maxv - minv)
    if invert:
        norm = 1.0 - norm
    return norm

# ---------------------------------------------------------------------------

def compute_scores(records):
    """
    Compute a combined balanced score for each record:
      * Lower weight is better -> invert normalization
      * Lower acceleration time is better -> invert normalization
      * Higher mpg is better -> normal normalization
    The score is the sum of the three normalized components.
    Returns a list of tuples (score, record_index, record).
    """
    weights = [r['Lbs-'] for r in records]
    accs    = [r['Acc+'] for r in records]
    mpgs    = [r['Mpg+'] for r in records]

    norm_w = normalize_min_to_max(weights, invert=True)
    norm_a = normalize_min_to_max(accs, invert=True)
    norm_m = normalize_min_to_max(mpgs)

    scores = []
    for idx, (nw, na, nm) in enumerate(zip(norm_w, norm_a, norm_m)):
        score = nw + na + nm
        scores.append((score, idx, records[idx]))
    return scores

# ---------------------------------------------------------------------------

def car_13(scores):
    """
    Identify a vehicle with the best balanced score.
    """
    if not scores:
        print("== CAR-13 ==")
        print("No valid vehicle data available.")
        return

    # Highest score is best balanced
    best = max(scores, key=lambda x: x[0])
    score, idx, rec = best
    print("== CAR-13 ==")
    print(f"Best balanced vehicle is row {idx + 1}:")
    print(f"  Score = {score:.3f}")
    print(f"  Weight = {rec['Lbs-']:.0f} lbs, Acceleration = {rec['Acc+']:.1f} s, MPG = {rec['Mpg+']:.1f}")

# ---------------------------------------------------------------------------

def car_14(scores, budget=5):
    """
    Recommend vehicles to inspect first given a limited budget.
    """
    if not scores:
        print("== CAR-14 ==")
        print("No valid vehicle data available.")
        return

    # Sort by score descending; higher is better
    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    print("== CAR-14 ==")
    print(f"With an inspection budget of {budget} vehicle(s), recommend inspecting first:")
    for i, (score, idx, rec) in enumerate(sorted_scores[:budget], start=1):
        print(f"{i}) Row {idx + 1}: Score={score:.3f}, Weight={rec['Lbs-']:.0f} lbs, Acc={rec['Acc+']:.1f} s, MPG={rec['Mpg+']:.1f}")

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

    scores = compute_scores(records)

    # CAR-13
    car_13(scores)

    # CAR-14
    car_14(scores, budget=5)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
