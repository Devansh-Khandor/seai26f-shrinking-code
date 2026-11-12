#!/usr/bin/env python3
"""
skill_08.py

Usage:
    python skill_08.py <csv_path>

Prints:
    CAR-15: For a poorly rated vehicle, show alternative vehicles in the inventory
            with similar characteristics but better recorded performance,
            and explain which differences could account for the improvement.
    CAR-16: For a vehicle that does not meet the preferred performance targets,
            identify the closest available alternative that meets more of those
            targets while changing as few vehicle characteristics as possible.
"""

import csv
import sys
import math
from statistics import mean, stdev, median
from collections import Counter

# ----------------------------------------------------------------------
# Data ingestion
# ----------------------------------------------------------------------
def read_csv(path):
    """Read CSV, convert to list of dicts; skip rows with missing or invalid numeric data."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                r = {}
                r['Clndrs'] = int(row['Clndrs'])
                r['Volume'] = float(row['Volume'])
                r['HpX']    = float(row['HpX'])
                r['Model']  = int(row['Model'])
                r['origin'] = int(row['origin'])
                r['Lbs-']   = float(row['Lbs-'])
                r['Acc+']   = float(row['Acc+'])
                r['Mpg+']   = float(row['Mpg+'])
                r['_idx']   = idx
                rows.append(r)
            except (ValueError, KeyError):
                continue
    return rows

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def compute_stats(values):
    n = len(values)
    if n == 0:
        return float('nan'), float('nan')
    return mean(values), stdev(values) if n > 1 else 0.0

def z_score(x, mean_val, std_val):
    return (x - mean_val) / std_val if std_val != 0 else 0.0

def euclidean_dist(v1, v2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

# ----------------------------------------------------------------------
# Core logic
# ----------------------------------------------------------------------
def main(csv_path):
    rows = read_csv(csv_path)
    if not rows:
        print("No valid data found.", file=sys.stderr)
        sys.exit(1)

    # Compute overall stats for standardized features
    weight_vals = [r['Lbs-'] for r in rows]
    mpg_vals    = [r['Mpg+'] for r in rows]
    acc_vals    = [r['Acc+'] for r in rows]

    w_mean, w_std = compute_stats(weight_vals)
    m_mean, m_std = compute_stats(mpg_vals)
    a_mean, a_std = compute_stats(acc_vals)

    # Compute combined desirability score and standardized vectors
    for r in rows:
        wz = z_score(r['Lbs-'], w_mean, w_std)
        mz = z_score(r['Mpg+'], m_mean, m_std)
        az = z_score(r['Acc+'], a_mean, a_std)
        r['z_vec'] = (wz, mz, az)
        r['score'] = -wz + mz - az  # lower weight, higher mpg, lower acc

    # Sort by score ascending (worst first) and descending (best first)
    rows_by_score = sorted(rows, key=lambda r: r['score'])
    worst_vehicle = rows_by_score[0]

    # ------------------------------------------------------------------
    # CAR-15: Alternatives for a poorly rated vehicle
    # ------------------------------------------------------------------
    print("== CAR-15 ==")
    print(f"Target vehicle (poorly rated): Vehicle #{worst_vehicle['_idx']} (ID={worst_vehicle['Clndrs']})")
    print(f"  Score: {worst_vehicle['score']:.3f}")
    print(f"  Weight: {worst_vehicle['Lbs-']:.1f} lbs")
    print(f"  MPG: {worst_vehicle['Mpg+']:.1f} MPG")
    print(f"  Acceleration: {worst_vehicle['Acc+']:.1f}s")
    print()

    # Find better vehicles within similar standardized distance
    better_rows = [r for r in rows if r['score'] > worst_vehicle['score']]
    # Compute distance to target
    better_rows = sorted(better_rows,
                         key=lambda r: euclidean_dist(r['z_vec'], worst_vehicle['z_vec']))
    # Take top 3
    top_alternatives = better_rows[:3]

    if not top_alternatives:
        print("No better alternatives found.")
    else:
        for i, alt in enumerate(top_alternatives, 1):
            print(f"{i}. Vehicle #{alt['_idx']} (ID={alt['Clndrs']})")
            print(f"   Score: {alt['score']:.3f}")
            print(f"   Weight: {alt['Lbs-']:.1f} lbs")
            print(f"   MPG: {alt['Mpg+']:.1f} MPG")
            print(f"   Acceleration: {alt['Acc+']:.1f}s")
            # Differences
            dw = alt['Lbs-'] - worst_vehicle['Lbs-']
            dm = alt['Mpg+'] - worst_vehicle['Mpg+']
            da = alt['Acc+'] - worst_vehicle['Acc+']
            print(f"   ΔWeight: {dw:+.1f} lbs, ΔMPG: {dm:+.1f}, ΔAcc: {da:+.1f}s")
            # Simple explanation
            explanation = []
            if dm > 0:
                explanation.append("higher fuel efficiency")
            if dw < 0:
                explanation.append("lighter weight")
            if da < 0:
                explanation.append("faster acceleration")
            if explanation:
                print(f"   Improvement factors: {', '.join(explanation)}")
            else:
                print("   No clear improvement factors detected.")
            print()

    # ------------------------------------------------------------------
    # CAR-16: Alternative that meets preferred targets with minimal changes
    # ------------------------------------------------------------------
    # Define preferred targets (median values as typical)
    weight_target = median(weight_vals)  # lower than or equal to median is preferable
    mpg_target    = median(mpg_vals)     # higher than or equal to median
    acc_target    = median(acc_vals)     # lower than or equal to median

    # Find a vehicle that violates at least one target
    violating = [r for r in rows if (r['Lbs-'] > weight_target or
                                    r['Mpg+'] < mpg_target or
                                    r['Acc+'] > acc_target)]
    if not violating:
        # If none, just pick the worst again
        viol = worst_vehicle
    else:
        # Pick the one with lowest score among violators
        viol = min(violating, key=lambda r: r['score'])

    print("\n== CAR-16 ==")
    print(f"Target vehicle (does not meet preferred targets): Vehicle #{viol['_idx']} (ID={viol['Clndrs']})")
    print(f"  Score: {viol['score']:.3f}")
    print(f"  Weight: {viol['Lbs-']:.1f} lbs")
    print(f"  MPG: {viol['Mpg+']:.1f} MPG")
    print(f"  Acceleration: {viol['Acc+']:.1f}s")
    print()

    # Determine which targets are unmet
    unmet = []
    if viol['Lbs-'] > weight_target:
        unmet.append('weight')
    if viol['Mpg+'] < mpg_target:
        unmet.append('mpg')
    if viol['Acc+'] > acc_target:
        unmet.append('acc')
    print(f"Unmet targets: {', '.join(unmet)}")
    print()

    # Find alternatives that satisfy more of the targets
    candidates = []
    for r in rows:
        if r['_idx'] == viol['_idx']:
            continue
        met = 0
        if r['Lbs-'] <= weight_target:
            met += 1
        if r['Mpg+'] >= mpg_target:
            met += 1
        if r['Acc+'] <= acc_target:
            met += 1
        # Only consider those that meet at least one more target than viol
        if met > len(unmet):
            # Compute raw feature distance (absolute differences)
            dist = euclidean_dist((r['Lbs-'], r['Mpg+'], r['Acc+']),
                                  (viol['Lbs-'], viol['Mpg+'], viol['Acc+']))
            candidates.append((r, met, dist))
    # Sort by number of met targets (descending) then by distance (ascending)
    candidates.sort(key=lambda x: (-x[1], x[2]))

    if not candidates:
        print("No suitable alternative found that meets more targets.")
    else:
        alt, met_count, dist = candidates[0]
        print(f"Recommended alternative (meets {met_count} more target(s)):")
        print(f"  Vehicle #{alt['_idx']} (ID={alt['Clndrs']})")
        print(f"  Score: {alt['score']:.3f}")
        print(f"  Weight: {alt['Lbs-']:.1f} lbs")
        print(f"  MPG: {alt['Mpg+']:.1f} MPG")
        print(f"  Acceleration: {alt['Acc+']:.1f}s")
        print()
        # Explain minimal changes
        changes = []
        if alt['Lbs-'] != viol['Lbs-']:
            changes.append(f"weight {alt['Lbs-']:.1f} → {viol['Lbs-']:.1f} lbs")
        if alt['Mpg+'] != viol['Mpg+']:
            changes.append(f"mpg {alt['Mpg+']:.1f} → {viol['Mpg+']:.1f} MPG")
        if alt['Acc+'] != viol['Acc+']:
            changes.append(f"acceleration {alt['Acc+']:.1f}s → {viol['Acc+']:.1f}s")
        print("Minimal changes required:")
        if changes:
            for c in changes:
                print(f"  {c}")
        else:
            print("  No changes needed.")
        # Summarize improvement
        improvement = []
        if alt['Lbs-'] <= weight_target and viol['Lbs-'] > weight_target:
            improvement.append("better weight")
        if alt['Mpg+'] >= mpg_target and viol['Mpg+'] < mpg_target:
            improvement.append("better fuel efficiency")
        if alt['Acc+'] <= acc_target and viol['Acc+'] > acc_target:
            improvement.append("better acceleration")
        print()
        print(f"Improvement achieved: {', '.join(improvement)}")

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
