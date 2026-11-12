#!/usr/bin/env python3
"""
skill_04.py

Usage:
    python skill_04.py <path-to-auto93.csv>

This script implements two new features for the dealership:

  CAR-07 : As new vehicle records arrive, assess each one immediately and
           tell me whether it belongs among our more desirable inventory options.

  CAR-08 : Keep our inventory summaries and recommended-vehicle shortlist
           up to date as additional vehicle records are received,
           without requiring staff to restart the entire analysis manually.

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
    Reads the CSV file and returns a list of dicts with numeric fields.
    Missing values are ignored (rows that cannot be parsed are skipped).
    Expected columns (in order):
      Clndrs,Volume,HpX,Model,origin,Lbs-,Acc+,Mpg+
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
    accs    = [r['Acc+'] for r in records]
    mpgs    = [r['Mpg+'] for r in records]

    norm_w = normalize_min_to_max(weights, invert=True)
    norm_a = normalize_min_to_max(accs, invert=True)
    norm_m = normalize_min_to_max(mpcs:=mpgs)

    scores = []
    for idx, rec in enumerate(records):
        score = norm_w[idx] + norm_a[idx] + norm_m[idx]
        scores.append((score, rec))
    return scores

# ---------------------------------------------------------------------------

def print_car_07(records, scores):
    """Print CAR-07 results: assessment of each vehicle."""
    print("== CAR-07 ==")
    # Sort scores descending for ranking
    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    # Map record to its rank (1-based)
    rank_dict = {}
    for rank, (_, rec) in enumerate(sorted_scores, start=1):
        rank_dict[id(rec)] = rank

    top_n = 5
    for idx, (score, rec) in enumerate(scores, start=1):
        rank = rank_dict[id(rec)]
        desirable = "Yes" if rank <= top_n else "No"
        print(f"{idx}) Vehicle row {idx}: score={score:.3f}, rank={rank}, desirable={desirable}")

# ---------------------------------------------------------------------------

def print_car_08(records, scores):
    """Print CAR-08 results: inventory summary and top vehicles."""
    print("\n== CAR-08 ==")

    # Summary statistics (CAR-01 style)
    weights = [r['Lbs-'] for r in records]
    mpgs    = [r['Mpg+'] for r in records]
    accs    = [r['Acc+'] for r in records]

    w_mean, w_std, w_min, w_max = compute_stats(weights)
    mpg_mean, mpg_std, mpg_min, mpg_max = compute_stats(mpcs:=mpgs)
    a_mean, a_std, a_min, a_max = compute_stats(accs)

    print(f"Weight: mean={w_mean:,.2f} lbs, std={w_std:,.2f}, min={w_min:.0f} lbs, max={w_max:.0f} lbs")
    print(f"Fuel efficiency: mean={mpg_mean:,.2f} mpg, std={mpg_std:,.2f}, min={mpg_min:.1f} mpg, max={mpg_max:.1f} mpg")
    print(f"Acceleration: mean={a_mean:,.2f} sec, std={a_std:,.2f}, min={a_min:.1f} sec, max={a_max:.1f} sec")

    # Top vehicles (CAR-02 style)
    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    top_n = 5
    print(f"\nTop {top_n} vehicles by combined score (low weight, high mpg, fast Acc+):")
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

    # Compute scores for all records
    scores = compute_scores(records)

    # CAR-07: assess each vehicle
    print_car_07(records, scores)

    # CAR-08: summary and top list
    print_car_08(records, scores)

if __name__ == "__main__":
    main()
