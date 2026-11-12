#!/usr/bin/env python3
"""
Skill 07: Candidate vehicle identification and inspection prioritisation for used‑car dealership.

CAR-13: Identify a vehicle with a strong balance of fuel efficiency, weight, and acceleration while evaluating as few candidate vehicles as reasonably possible.

CAR-14: With a limited inspection budget, recommend which vehicles should be evaluated first to discover promising inventory options early.
"""

import csv
import sys

# ----------------------------------------------------------------------
# Data parsing
# ----------------------------------------------------------------------
NUMERIC_KEYS = [
    'Clndrs', 'Volume', 'HpX', 'Model', 'origin',
    'Lbs-', 'Acc+', 'Mpg+'
]

def read_data(csv_path):
    """Read the CSV and convert numeric columns to float or None."""
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in NUMERIC_KEYS:
                try:
                    val = row.get(k, '')
                    row[k] = float(val) if val != '' else None
                except (ValueError, KeyError):
                    row[k] = None
            rows.append(row)
    return rows

# ----------------------------------------------------------------------
# Score calculation helpers
# ----------------------------------------------------------------------
def compute_score_stats(rows):
    """Return min/max values for weight, mpg and acceleration."""
    weights = [r['Lbs-'] for r in rows if r['Lbs-'] is not None]
    mpgs   = [r['Mpg+'] for r in rows if r['Mpg+'] is not None]
    accs   = [r['Acc+'] for r in rows if r['Acc+'] is not None]
    return {
        'wmin': min(weights) if weights else 0.0,
        'wmax': max(weights) if weights else 0.0,
        'mmin': min(mpgs) if mpgs else 0.0,
        'mmax': max(mpgs) if mpgs else 0.0,
        'amin': min(accs) if accs else 0.0,
        'amax': max(accs) if accs else 0.0
    }

def overall_score(row, stats):
    """Return score in range 0–3 or None if any component is missing."""
    w, m, a = row.get('Lbs-'), row.get('Mpg+'), row.get('Acc+')
    if w is None or m is None or a is None:
        return None
    w_score = (stats['wmax'] - w) / (stats['wmax'] - stats['wmin']) if stats['wmax'] != stats['wmin'] else 1.0
    m_score = (m - stats['mmin']) / (stats['mmax'] - stats['mmin']) if stats['mmax'] != stats['mmin'] else 1.0
    a_score = (stats['amax'] - a) / (stats['amax'] - stats['amin']) if stats['amax'] != stats['amin'] else 1.0
    return w_score + m_score + a_score

# ----------------------------------------------------------------------
# CAR-13: Balanced vehicle selection
# ----------------------------------------------------------------------
def car_13(rows, stats):
    """Print the single vehicle with the highest overall score."""
    print("== CAR-13 ==")
    scored = [(idx + 1, r, overall_score(r, stats))
              for idx, r in enumerate(rows) if overall_score(r, stats) is not None]
    if not scored:
        print("No vehicle has complete data for scoring.")
        return

    best = max(scored, key=lambda x: x[2])
    idx, r, score = best
    print(f"Selected vehicle: row #{idx}")
    print(f"  Weight: {r['Lbs-']:.1f} lbs")
    print(f"  MPG: {r['Mpg+']:.1f} mpg")
    print(f"  Acceleration: {r['Acc+']:.1f} s")
    print(f"  Overall score: {score:.3f}")

# ----------------------------------------------------------------------
# CAR-14: Inspection prioritisation
# ----------------------------------------------------------------------
def car_14(rows, stats, top_n=3):
    """Print the top N vehicles to inspect first, ranked by overall score."""
    print("\n== CAR-14 ==")
    scored = [(idx + 1, r, overall_score(r, stats))
              for idx, r in enumerate(rows) if overall_score(r, stats) is not None]
    if not scored:
        print("No vehicle has complete data for scoring.")
        return
    scored.sort(key=lambda x: x[2], reverse=True)
    print(f"Top {top_n} vehicles to inspect first (by overall score):")
    for i, (idx, r, score) in enumerate(scored[:top_n], start=1):
        print(f"{i}. Row #{idx} – Score {score:.3f}, Weight {r['Lbs-']:.1f} lbs, MPG {r['Mpg+']:.1f} mpg, Acc {r['Acc+']:.1f} s")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main(csv_path):
    rows = read_data(csv_path)
    if not rows:
        print("No data found in CSV.")
        return
    stats = compute_score_stats(rows)
    car_13(rows, stats)
    car_14(rows, stats, top_n=3)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
