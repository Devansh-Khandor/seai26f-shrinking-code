#!/usr/bin/env python3
"""
Skill 08: Vehicle comparison and alternative recommendation for used‑car dealership.

CAR-15: For a poorly rated vehicle, show alternative vehicles in our historical inventory
with similar characteristics but better recorded performance, and explain which differences
could account for the improvement.

CAR-16: For a vehicle that does not meet our preferred performance targets,
identify the closest available alternative that meets more of those targets while
changing as few vehicle characteristics as possible.
"""

import csv
import sys
import math

# ----------------------------------------------------------------------
# Data handling utilities
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
        for raw in reader:
            row = {}
            for k in NUMERIC_KEYS:
                try:
                    val = raw.get(k, '')
                    row[k] = float(val) if val != '' else None
                except (ValueError, KeyError):
                    row[k] = None
            rows.append(row)
    return rows

# ----------------------------------------------------------------------
# Scoring helpers
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

def normalize(value, min_val, max_val):
    """Safe normalization to [0,1]."""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)

def similarity(row1, row2, stats):
    """Euclidean distance in normalized (weight, mpg, acc) space."""
    w1, m1, a1 = row1['Lbs-'], row1['Mpg+'], row1['Acc+']
    w2, m2, a2 = row2['Lbs-'], row2['Mpg+'], row2['Acc+']
    if None in (w1, m1, a1, w2, m2, a2):
        return math.inf
    dw = normalize(w1, stats['wmin'], stats['wmax']) - normalize(w2, stats['wmin'], stats['wmax'])
    dm = normalize(m1, stats['mmin'], stats['mmax']) - normalize(m2, stats['mmin'], stats['mmax'])
    da = normalize(a1, stats['amin'], stats['amax']) - normalize(a2, stats['amin'], stats['amax'])
    return math.sqrt(dw*dw + dm*dm + da*da)

# ----------------------------------------------------------------------
# CAR‑15 implementation
# ----------------------------------------------------------------------
def car_15(rows, stats):
    """Show alternatives for the poorest‑rated vehicle."""
    print("== CAR-15 ==")

    # Compute overall scores where possible
    for r in rows:
        r['overall_score'] = overall_score(r, stats)

    # Find the vehicle with the lowest valid overall score
    valid_rows = [r for r in rows if r['overall_score'] is not None]
    if not valid_rows:
        print("No vehicle with complete data to evaluate.")
        return

    worst = min(valid_rows, key=lambda r: r['overall_score'])

    print(f"Selected poorly rated vehicle (row #{rows.index(worst)+1}):")
    print(f"  Weight: {worst['Lbs-']:.1f} lbs")
    print(f"  MPG:    {worst['Mpg+']:.1f} mpg")
    print(f"  Acc.:   {worst['Acc+']:.1f} s")
    print(f"  Overall Score: {worst['overall_score']:.3f}")

    # Find alternatives: better score AND similar characteristics
    alternatives = []
    for r in valid_rows:
        if r is worst:
            continue
        if r['overall_score'] <= worst['overall_score']:
            continue
        dist = similarity(r, worst, stats)
        alternatives.append((dist, r))

    alternatives.sort(key=lambda x: x[0])
    alternatives = alternatives[:3]  # top 3

    if not alternatives:
        print("\nNo better vehicles with similar characteristics found.")
        return

    print("\nAlternative vehicles with better performance:")
    for idx, (dist, alt) in enumerate(alternatives, start=1):
        print(f"{idx}. Row #{rows.index(alt)+1} – Score {alt['overall_score']:.3f}, Dist {dist:.3f}")
        print(f"   Weight: {alt['Lbs-']:.1f} lbs vs {worst['Lbs-']:.1f} lbs")
        print(f"   MPG:    {alt['Mpg+']:.1f} mpg vs {worst['Mpg+']:.1f} mpg")
        print(f"   Acc.:   {alt['Acc+']:.1f} s  vs {worst['Acc+']:.1f} s")
        # Explain differences
        weight_diff = alt['Lbs-'] - worst['Lbs-']
        mpg_diff    = alt['Mpg+'] - worst['Mpg+']
        acc_diff    = alt['Acc+'] - worst['Acc+']
        reasons = []
        if weight_diff < 0:
            reasons.append(f"lighter by {abs(weight_diff):.1f} lbs")
        if mpg_diff > 0:
            reasons.append(f"higher MPG by {mpg_diff:.1f}")
        if acc_diff < 0:
            reasons.append(f"faster acceleration by {abs(acc_diff):.1f} s")
        if reasons:
            print(f"   Improvements: {', '.join(reasons)}.")
        else:
            print(f"   No clear numeric advantage over the reference vehicle.")
    print()

# ----------------------------------------------------------------------
# CAR‑16 implementation
# ----------------------------------------------------------------------
def car_16(rows, stats):
    """Find the closest alternative that meets more performance targets."""
    print("== CAR-16 ==")

    # Define preferred thresholds
    weight_target = 4000.0   # lbs
    mpg_target    = 20.0     # mpg
    acc_target    = 15.0     # s

    # Compute overall scores for all rows
    for r in rows:
        r['overall_score'] = overall_score(r, stats)

    # Pick the vehicle that fails at least one target and has the lowest overall score
    def targets_met(r):
        score = 0
        if r['Lbs-'] is not None and r['Lbs-'] <= weight_target:
            score += 1
        if r['Mpg+'] is not None and r['Mpg+'] >= mpg_target:
            score += 1
        if r['Acc+'] is not None and r['Acc+'] <= acc_target:
            score += 1
        return score

    candidates = [r for r in rows if r['overall_score'] is not None]
    if not candidates:
        print("No vehicle with complete data to evaluate.")
        return

    # Find the worst vehicle in terms of overall score
    worst = min(candidates, key=lambda r: r['overall_score'])
    worst_targets = targets_met(worst)

    print(f"Selected vehicle that does not meet many targets (row #{rows.index(worst)+1}):")
    print(f"  Weight: {worst['Lbs-']:.1f} lbs (target ≤ {weight_target})")
    print(f"  MPG:    {worst['Mpg+']:.1f} mpg (target ≥ {mpg_target})")
    print(f"  Acc.:   {worst['Acc+']:.1f} s (target ≤ {acc_target})")
    print(f"  Targets met: {worst_targets}/3")
    print(f"  Overall Score: {worst['overall_score']:.3f}")

    # Find alternatives that meet more targets than worst
    better_alts = []
    for r in rows:
        if r is worst or r['overall_score'] is None:
            continue
        met = targets_met(r)
        if met > worst_targets:
            dist = similarity(r, worst, stats)
            better_alts.append((dist, met, r))

    if not better_alts:
        print("\nNo vehicle meets more targets than the selected one.")
        return

    # Choose the one with most targets met, and if tie, smallest distance
    better_alts.sort(key=lambda x: (-x[1], x[0]))
    best_dist, best_met, best = better_alts[0]

    print("\nClosest alternative meeting more performance targets:")
    print(f"  Row #{rows.index(best)+1} – Targets met: {best_met}/3, Distance: {best_dist:.3f}")
    print(f"  Weight: {best['Lbs-']:.1f} lbs")
    print(f"  MPG:    {best['Mpg+']:.1f} mpg")
    print(f"  Acc.:   {best['Acc+']:.1f} s")
    print(f"  Overall Score: {best['overall_score']:.3f}")

    # Explain which characteristics changed
    print("\nKey differences compared to the original vehicle:")
    changes = []
    if best['Lbs-'] is not None and worst['Lbs-'] is not None:
        d = best['Lbs-'] - worst['Lbs-']
        if d < 0:
            changes.append(f"lighter by {abs(d):.1f} lbs")
        elif d > 0:
            changes.append(f"heavier by {d:.1f} lbs")
    if best['Mpg+'] is not None and worst['Mpg+'] is not None:
        d = best['Mpg+'] - worst['Mpg+']
        if d > 0:
            changes.append(f"higher MPG by {d:.1f}")
        elif d < 0:
            changes.append(f"lower MPG by {abs(d):.1f}")
    if best['Acc+'] is not None and worst['Acc+'] is not None:
        d = best['Acc+'] - worst['Acc+']
        if d < 0:
            changes.append(f"faster acceleration by {abs(d):.1f} s")
        elif d > 0:
            changes.append(f"slower acceleration by {d:.1f} s")
    if changes:
        print(f"  {', '.join(changes)}.")
    else:
        print("  No numeric difference detected.")

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main(csv_path):
    rows = read_data(csv_path)
    if not rows:
        print("No data found in CSV.")
        return

    stats = compute_score_stats(rows)

    car_15(rows, stats)
    car_16(rows, stats)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
