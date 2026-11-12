#!/usr/bin/env python3
"""
Skill 05: Advanced diagnostics for used‑car dealership.

CAR-09: When a vehicle has unusually poor fuel efficiency, identify which of its
recorded characteristics are most strongly associated with that outcome.

CAR-10: Find the common characteristics of vehicles with unfavorable overall
performance and help understand how they differ from better‑performing vehicles.
"""

import csv
import sys
import math
from collections import defaultdict

# ----------------------------------------------------------------------
# Data parsing
# ----------------------------------------------------------------------
NUMERIC_KEYS = [
    'Clndrs', 'Volume', 'HpX', 'Model', 'origin',
    'Lbs-', 'Acc+', 'Mpg+'
]

def read_data(csv_path):
    """Read CSV and convert numeric columns to float or None."""
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in NUMERIC_KEYS:
                try:
                    val = row.get(key, '')
                    row[key] = float(val) if val != '' else None
                except (ValueError, KeyError):
                    row[key] = None
            rows.append(row)
    return rows

# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def mean(values):
    return sum(values) / len(values) if values else 0.0

def stddev(values, ddof=1):
    n = len(values)
    if n <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (n - ddof))

def pearson_r(xs, ys):
    """Compute Pearson correlation coefficient for two lists of numbers."""
    if len(xs) != len(ys) or not xs:
        return 0.0
    n = len(xs)
    mx = mean(xs)
    my = mean(ys)
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    sp = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    denom = math.sqrt(sx * sy)
    return sp / denom if denom != 0 else 0.0

# ----------------------------------------------------------------------
# Overall score (weight, mpg, acceleration)
# ----------------------------------------------------------------------
def compute_score_stats(rows):
    """Return min/max for weight, mpg and acceleration."""
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
    """Return score in range 0–3 or None if missing data."""
    w, m, a = row.get('Lbs-'), row.get('Mpg+'), row.get('Acc+')
    if w is None or m is None or a is None:
        return None
    w_score = (stats['wmax'] - w) / (stats['wmax'] - stats['wmin']) if stats['wmax'] != stats['wmin'] else 1.0
    m_score = (m - stats['mmin']) / (stats['mmax'] - stats['mmin']) if stats['mmax'] != stats['mmin'] else 1.0
    a_score = (stats['amax'] - a) / (stats['amax'] - stats['amin']) if stats['amax'] != stats['amin'] else 1.0
    return w_score + m_score + a_score

# ----------------------------------------------------------------------
# CAR-09 logic
# ----------------------------------------------------------------------
def car_09(rows):
    print("== CAR-09 ==")

    # Find vehicle with worst MPG
    worst_rows = [r for r in rows if r['Mpg+'] is not None]
    if not worst_rows:
        print("No MPG data available.")
        return
    worst = min(worst_rows, key=lambda r: r['Mpg+'])
    print(f"Worst vehicle (row #{rows.index(worst)+1}): Mpg+ = {worst['Mpg+']:.2f}")

    # Compute correlations of all numeric attributes with MPG
    attr_correlations = {}
    for attr in NUMERIC_KEYS:
        if attr == 'Mpg+':
            continue
        xs = [r[attr] for r in rows if r[attr] is not None and r['Mpg+'] is not None]
        ys = [r['Mpg+'] for r in rows if r[attr] is not None and r['Mpg+'] is not None]
        r = pearson_r(xs, ys)
        attr_correlations[attr] = r

    # Sort by correlation descending (most negative first)
    sorted_attrs = sorted(attr_correlations.items(), key=lambda x: x[1])

    print("\nMost strongly negatively correlated attributes with MPG:")
    for i, (attr, r) in enumerate(sorted_attrs[:3], start=1):
        print(f"{i}. {attr:>7} (r = {r:+.3f})")

# ----------------------------------------------------------------------
# CAR-10 logic
# ----------------------------------------------------------------------
def car_10(rows, stats):
    print("== CAR-10 ==")

    # Compute overall scores for all rows
    for r in rows:
        r['overall_score'] = overall_score(r, stats)

    # Split into favorable/unfavorable by median score
    scores = [r['overall_score'] for r in rows if r['overall_score'] is not None]
    if len(scores) < 2:
        print("Insufficient data for overall score comparison.")
        return
    median = sorted(scores)[len(scores)//2]

    fav_group = [r for r in rows if r.get('overall_score') is not None and r['overall_score'] >= median]
    unfav_group = [r for r in rows if r.get('overall_score') is not None and r['overall_score'] < median]

    def group_mean(group, attr):
        vals = [r[attr] for r in group if r[attr] is not None]
        return mean(vals) if vals else 0.0

    print("\nMean characteristics by performance group:")
    header = f"{'Attribute':>10} | {'Unfavorable':>12} | {'Favorable':>10} | {'Difference':>10}"
    print(header)
    print("-" * len(header))
    for attr in NUMERIC_KEYS:
        if attr == 'Mpg+':  # MPG is part of overall but we still show
            pass
        unfav = group_mean(unfav_group, attr)
        fav = group_mean(fav_group, attr)
        diff = fav - unfav
        print(f"{attr:>10} | {unfav:12.2f} | {fav:10.2f} | {diff:10.2f}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main(csv_path):
    rows = read_data(csv_path)
    if not rows:
        print("No data found.")
        return

    # CAR-09
    car_09(rows)

    # CAR-10
    stats = compute_score_stats(rows)
    car_10(rows, stats)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
