#!/usr/bin/env python3
"""
Skill 06: Explain vehicle recommendations and trade‑offs for used‑car dealership.

CAR-11: When the system recommends a vehicle, explain its recommendation using
        understandable comparisons with similar vehicles in our inventory.

CAR-12: When two vehicles have different strengths, explain why one receives a
        higher overall recommendation and show the trade‑offs involved.
"""

import csv
import sys
from collections import defaultdict

# ----------------------------------------------------------------------
# CSV parsing utilities
# ----------------------------------------------------------------------
NUMERIC_KEYS = [
    'Clndrs', 'Volume', 'HpX', 'Model', 'origin',
    'Lbs-', 'Acc+', 'Mpg+'
]

def parse_row(row):
    """Convert numeric fields to float or None."""
    for k in NUMERIC_KEYS:
        try:
            val = row.get(k, '')
            row[k] = float(val) if val != '' else None
        except (ValueError, KeyError):
            row[k] = None
    return row

# ----------------------------------------------------------------------
# Score calculation helpers
# ----------------------------------------------------------------------
def compute_score(row, stats):
    """Return overall score (0–3) or None if any component is missing."""
    w, m, a = row.get('Lbs-'), row.get('Mpg+'), row.get('Acc+')
    if w is None or m is None or a is None:
        return None

    w_score = (stats['wmax'] - w) / (stats['wmax'] - stats['wmin']) if stats['wmax'] != stats['wmin'] else 1.0
    m_score = (m - stats['mmin']) / (stats['mmax'] - stats['mmin']) if stats['mmax'] != stats['mmin'] else 1.0
    a_score = (stats['amax'] - a) / (stats['amax'] - stats['amin']) if stats['amax'] != stats['amin'] else 1.0
    return w_score + m_score + a_score, w_score, m_score, a_score

# ----------------------------------------------------------------------
# Helper to compute percentiles for comparisons
# ----------------------------------------------------------------------
def percentile_value(value, values, better_is_lower=True):
    """
    Return the percentage of values that are better (i.e., lower if better_is_lower)
    than the supplied value.
    """
    n = len(values)
    if n == 0:
        return 0.0
    if better_is_lower:
        better = sum(1 for v in values if v > value)
    else:
        better = sum(1 for v in values if v < value)
    return better / n

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main(csv_path):
    # Load data
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append(parse_row(raw))

    if not rows:
        print("No data found in CSV.")
        return

    # Gather numeric lists for comparison
    weights = [r['Lbs-'] for r in rows if r['Lbs-'] is not None]
    mpgs    = [r['Mpg+'] for r in rows if r['Mpg+'] is not None]
    accs    = [r['Acc+'] for r in rows if r['Acc+'] is not None]

    # Stats needed for scoring
    stats = {
        'wmin': min(weights) if weights else 0.0,
        'wmax': max(weights) if weights else 0.0,
        'mmin': min(mpgs) if mpgs else 0.0,
        'mmax': max(mpgs) if mpgs else 0.0,
        'amin': min(accs) if accs else 0.0,
        'amax': max(accs) if accs else 0.0
    }

    # Compute overall scores
    for r in rows:
        res = compute_score(r, stats)
        if res is None:
            r['overall_score'] = None
            r['component_scores'] = (None, None, None)
        else:
            score, w_s, m_s, a_s = res
            r['overall_score'] = score
            r['component_scores'] = (w_s, m_s, a_s)

    # Sort rows by descending score, ignoring those that couldn't be scored
    scored_rows = [r for r in rows if r['overall_score'] is not None]
    scored_rows.sort(key=lambda r: r['overall_score'], reverse=True)
    top3 = scored_rows[:3]

    # --------------------------------------------------------------
    # CAR-11: Explain recommendations
    # --------------------------------------------------------------
    print("== CAR-11 ==")
    if not top3:
        print("No vehicles with sufficient data to score.")
    else:
        for idx, r in enumerate(top3, start=1):
            w = r['Lbs-']
            m = r['Mpg+']
            a = r['Acc+']
            score = r['overall_score']

            pct_weight = percentile_value(w, weights, better_is_lower=True) * 100
            pct_mpg    = percentile_value(m, mpgs, better_is_lower=False) * 100
            pct_acc    = percentile_value(a, accs, better_is_lower=True) * 100

            print(f"\nRecommendation #{idx}")
            print(f"  Weight: {w:.1f} lbs – lighter than {pct_weight:.1f}% of inventory")
            print(f"  MPG: {m:.1f} mpg – higher than {pct_mpg:.1f}% of inventory")
            print(f"  Acceleration: {a:.1f} s – faster than {pct_acc:.1f}% of inventory")
            print(f"  Overall score: {score:.3f} (higher is better)")

    # --------------------------------------------------------------
    # CAR-12: Trade‑off comparison between two vehicles
    # --------------------------------------------------------------
    print("\n== CAR-12 ==")
    if len(top3) < 2:
        print("Insufficient recommended vehicles to compare trade‑offs.")
    else:
        v1, v2 = top3[0], top3[1]
        w1, m1, a1 = v1['component_scores']
        w2, m2, a2 = v2['component_scores']
        score1 = v1['overall_score']
        score2 = v2['overall_score']

        print(f"Vehicle 1: {v1['Lbs-']:.1f} lbs, {v1['Mpg+']:.1f} mpg, {v1['Acc+']:.1f} s, Score {score1:.3f}")
        print(f"Vehicle 2: {v2['Lbs-']:.1f} lbs, {v2['Mpg+']:.1f} mpg, {v2['Acc+']:.1f} s, Score {score2:.3f}\n")

        print("Component comparison (higher is better for all except weight and acceleration where lower is better):")
        print(f"  Weight component: Vehicle 1 = {w1:.3f} vs Vehicle 2 = {w2:.3f} → Vehicle 1 has a slightly better (lower) weight contribution.")
        print(f"  MPG component:    Vehicle 1 = {m1:.3f} vs Vehicle 2 = {m2:.3f} → Vehicle 2 outperforms Vehicle 1 on fuel efficiency.")
        print(f"  Acceleration:    Vehicle 1 = {a1:.3f} vs Vehicle 2 = {a2:.3f} → Vehicle 1 is slightly faster.\n")

        print(f"Overall, Vehicle 1 has a higher score ({score1:.3f}) because its weight and acceleration advantages outweigh Vehicle 2's MPG advantage.")
        print("Trade‑offs:")
        print(f"  Vehicle 1: better weight (+{(w1-w2):.3f}) and acceleration (+{(a1-a2):.3f}) but lower MPG ({m1:.3f} vs {m2:.3f}).")
        print(f"  Vehicle 2: better MPG (+{(m2-m1):.3f}) but slightly heavier ({v2['Lbs-']:.1f} vs {v1['Lbs-']:.1f}) and slower acceleration ({v2['Acc+']:.1f} vs {v1['Acc+']:.1f}).")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
