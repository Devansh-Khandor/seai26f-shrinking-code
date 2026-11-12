#!/usr/bin/env python3
"""
skill_08.py

Usage:
    python skill_08.py <path-to-auto93.csv>

This script implements two new features for the dealership:

  CAR-15 : For a poorly rated vehicle, show alternative vehicles in the historical
           inventory with similar characteristics but better recorded performance,
           and explain which differences could account for the improvement.

  CAR-16 : For a vehicle that does not meet our preferred performance targets,
           identify the closest available alternative that meets more of those targets
           while changing as few vehicle characteristics as possible.

The output follows the requested format exactly: for each request, a line that
is exactly `== <ID> ==` followed by at least three lines of results, including
the actual numbers.
"""

import csv
import sys
import numpy as np
from math import sqrt


# ---------------------------------------------------------------------------

def read_data(csv_path):
    """Read CSV and return list of records with numeric fields."""
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

def score_record(rec):
    """Return combined score for a record."""
    # Normalizations
    # weight and acc are better when lower -> invert
    # mpg is better when higher
    # We'll normalise on the fly using the entire dataset later
    return rec  # placeholder; actual score computed in compute_scores()


def compute_scores(records):
    """Return list of (score, idx, rec) tuples."""
    weights = np.array([r['Lbs-'] for r in records], dtype=float)
    accs    = np.array([r['Acc+'] for r in records], dtype=float)
    mpgs    = np.array([r['Mpg+'] for r in records], dtype=float)

    # Min-max normalisation
    def norm(arr, invert=False):
        minv, maxv = arr.min(), arr.max()
        if maxv == minv:
            return np.zeros_like(arr)
        normed = (arr - minv) / (maxv - minv)
        return 1.0 - normed if invert else normed

    norm_w = norm(weights, invert=True)
    norm_a = norm(accs, invert=True)
    norm_m = norm(mpgs, invert=False)

    scores = []
    for i, (w, a, m) in enumerate(zip(norm_w, norm_a, norm_m)):
        scores.append((w + a + m, i, records[i]))
    return scores


# ---------------------------------------------------------------------------

def find_worst(scores):
    """Return the record with the lowest score."""
    worst = min(scores, key=lambda x: x[0])
    return worst  # (score, idx, rec)


# ---------------------------------------------------------------------------

def car_15(scores, worst):
    """Implement CAR-15."""
    print("== CAR-15 ==")
    worst_score, worst_idx, worst_rec = worst
    # Tolerances: 10% of worst vehicle's values
    tol_w = 0.10 * worst_rec['Lbs-']
    tol_a = 0.10 * worst_rec['Acc+']
    tol_m = 0.10 * worst_rec['Mpg+']

    alternatives = []
    for score, idx, rec in scores:
        if idx == worst_idx:
            continue
        if (abs(rec['Lbs-'] - worst_rec['Lbs-']) <= tol_w and
            abs(rec['Acc+'] - worst_rec['Acc+']) <= tol_a and
            abs(rec['Mpg+'] - worst_rec['Mpg+']) <= tol_m):
            if score > worst_score:   # better performance
                alternatives.append((score, idx, rec))

    if not alternatives:
        print("  No similar vehicles with better performance found.")
        return

    # Sort by score descending
    alternatives.sort(key=lambda x: x[0], reverse=True)

    # Show top 3
    for i, (score, idx, rec) in enumerate(alternatives[:3], start=1):
        weight_diff = rec['Lbs-'] - worst_rec['Lbs-']
        acc_diff    = rec['Acc+'] - worst_rec['Acc+']
        mpg_diff    = rec['Mpg+'] - worst_rec['Mpg+']
        print(f"{i}) Vehicle row {idx + 1}: Score={score:.3f}")
        print(f"   - Weight {rec['Lbs-']:.0f} lbs ({weight_diff:+.0f} vs worst)")
        print(f"   - Acceleration {rec['Acc+']:.1f} s ({acc_diff:+.1f} vs worst)")
        print(f"   - MPG {rec['Mpg+']:.1f} ({mpg_diff:+.1f} vs worst)")
        print(f"   -> This vehicle improves by Δscore={score - worst_score:+.3f}")


# ---------------------------------------------------------------------------

def car_16(scores, worst):
    """Implement CAR-16."""
    print("\n== CAR-16 ==")
    worst_score, worst_idx, worst_rec = worst

    # Define preferred targets based on median of dataset
    weights = np.array([r['Lbs-'] for r in records], dtype=float)
    accs    = np.array([r['Acc+'] for r in records], dtype=float)
    mpgs    = np.array([r['Mpg+'] for r in records], dtype=float)

    weight_target = np.median(weights)
    acc_target    = np.median(accs)
    mpg_target    = np.median(mpgs)

    # Helper to count how many targets met
    def target_count(rec):
        count = 0
        if rec['Lbs-'] <= weight_target:
            count += 1
        if rec['Acc+'] <= acc_target:
            count += 1
        if rec['Mpg+'] >= mpg_target:
            count += 1
        return count

    worst_target_hits = target_count(worst_rec)

    # Find candidates that meet strictly more targets and have higher score
    candidates = []
    for score, idx, rec in scores:
        if idx == worst_idx:
            continue
        hits = target_count(rec)
        if hits > worst_target_hits and score > worst_score:
            # compute Euclidean distance in feature space
            dist = sqrt((rec['Lbs-'] - worst_rec['Lbs-'])**2 +
                        (rec['Acc+'] - worst_rec['Acc+'])**2 +
                        (rec['Mpg+'] - worst_rec['Mpg+'])**2)
            candidates.append((hits, dist, score, idx, rec))

    if not candidates:
        print("  No better alternative meets more targets.")
        return

    # Choose candidate with most targets, then minimal distance, then higher score
    candidates.sort(key=lambda x: (-x[0], x[1], -x[2]))
    hits, dist, score, idx, rec = candidates[0]

    print(f"  Target vehicle row {worst_idx + 1} meets {worst_target_hits}/3 performance targets.")
    print(f"  Closest better alternative row {idx + 1} meets {hits}/3 targets:")
    print(f"    - Score: {score:.3f} (Δscore={score - worst_score:+.3f})")
    print(f"    - Weight: {rec['Lbs-']:.0f} lbs (Δ={rec['Lbs-'] - worst_rec['Lbs-']:+.0f})")
    print(f"    - Acceleration: {rec['Acc+']:.1f} s (Δ={rec['Acc+'] - worst_rec['Acc+']:+.1f})")
    print(f"    - MPG: {rec['Mpg+']:.1f} (Δ={rec['Mpg+'] - worst_rec['Mpg+']:+.1f})")
    print(f"    - Euclidean distance in feature space: {dist:.1f}")
    print(f"  This vehicle achieves a higher score while changing only {int(round(dist))} points in combined characteristics.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-auto93.csv>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    records = read_data(csv_path)

    if not records:
        print("No valid data found in the CSV.", file=sys.stderr)
        sys.exit(1)

    # Compute scores
    scores = compute_scores(records)

    # Identify worst vehicle
    worst = find_worst(scores)

    # Run CAR-15
    car_15(scores, worst)

    # Run CAR-16
    car_16(scores, worst)
