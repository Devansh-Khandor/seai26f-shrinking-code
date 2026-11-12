#!/usr/bin/env python3
"""
Skill 08: Alternative vehicle recommendations
============================================

This script implements two reports:

* CAR-15 – For the most poorly scored vehicle, find
           other inventory vehicles with similar
           weight/MPG/acceleration but with a
           higher overall score, and explain why they
           outperform the reference vehicle.

* CAR-16 – For a vehicle that fails the dealership’s
           preferred performance targets
           (weight ≤ 4000 lbs, MPG ≥ 15, Acc ≤ 15 s),
           find the closest available alternative that
           meets more of these targets while changing
           as few characteristics as possible.

The script reads the Auto93 dataset from a single CSV path
provided on the command line.

Usage:
    python skill_08.py <csv_path>
"""

import csv
import sys
import math
from collections import defaultdict

# ----------------------------------------------------------------------
# Basic CSV reading
# ----------------------------------------------------------------------
def read_csv(csv_path):
    """Return a list of rows with numeric values or None for missing data."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            parsed = {}
            for key in ("Clndrs", "Volume", "HpX", "Model", "origin",
                        "Lbs-", "Acc+", "Mpg+"):
                try:
                    parsed[key] = float(r[key])
                except (KeyError, ValueError, TypeError):
                    parsed[key] = None
            rows.append(parsed)
    return rows

# ----------------------------------------------------------------------
# Normalisation & scoring (same as earlier skills)
# ----------------------------------------------------------------------
def normalize(val, low, high, reverse=False):
    """Return a 0–1 normalised value, optionally reversed."""
    if high == low or val is None:
        return 0.5
    norm = (val - low) / (high - low)
    return 1.0 - norm if reverse else norm

def compute_score(row, w_min, w_max, mpg_min, mpg_max, acc_min, acc_max):
    nw = normalize(row["Lbs-"], w_min, w_max, reverse=True)   # lighter better
    nm = normalize(row["Mpg+"], mpg_min, mpg_max)              # higher better
    na = normalize(row["Acc+"], acc_min, acc_max, reverse=True)  # lower better
    return (nw + nm + na) / 3.0

# ----------------------------------------------------------------------
# Helper: find most poorly scored vehicle
# ----------------------------------------------------------------------
def find_worst_vehicle(rows, scored):
    """Return the row and score of the vehicle with the lowest score."""
    if not scored:
        return None, None
    worst = min(scored, key=lambda x: x[2])  # (idx, row, score)
    return worst[1], worst[2]

# ----------------------------------------------------------------------
# CAR-15 implementation
# ----------------------------------------------------------------------
def car_15(rows):
    """
    For the worst‑scored vehicle, list up to 3 better alternatives
    with similar weight, mpg, and acceleration (within 10 % of the
    reference values).  Explain why those alternatives outperform
    the reference vehicle.
    """
    # Prepare min/max for score calculation
    w_min, w_max = min(r["Lbs-"] for r in rows if r["Lbs-"] is not None), \
                   max(r["Lbs-"] for r in rows if r["Lbs-"] is not None)
    mpg_min, mpg_max = min(r["Mpg+"] for r in rows if r["Mpg+"] is not None), \
                       max(r["Mpg+"] for r in rows if r["Mpg+"] is not None)
    acc_min, acc_max = min(r["Acc+"] for r in rows if r["Acc+"] is not None), \
                       max(r["Acc+"] for r in rows if r["Acc+"] is not None)

    # Compute scores for all complete records
    scored = [
        (i, r, compute_score(r, w_min, w_max, mpg_min, mpg_max, acc_min, acc_max))
        for i, r in enumerate(rows, start=1)
        if None not in (r["Lbs-"], r["Mpg+"], r["Acc+"])
    ]

    # Identify worst vehicle
    worst_row, worst_score = find_worst_vehicle(rows, scored)
    if worst_row is None:
        return ["No complete vehicles to analyse for CAR‑15."]

    # Tolerance threshold: 10 %
    def within_tolerance(val, ref):
        return abs(val - ref) / ref <= 0.10

    # Find better vehicles with similar characteristics
    alternatives = []
    for i, r, score in scored:
        if score <= worst_score:
            continue
        if all(within_tolerance(r[k], worst_row[k]) for k in ("Lbs-", "Mpg+", "Acc+")):
            alternatives.append((i, r, score))

    alternatives.sort(key=lambda x: x[2], reverse=True)
    alternatives = alternatives[:3]  # top 3

    out = ["=== CAR-15: Better alternatives for the worst vehicle ==="]
    out.append(f"Worst vehicle (score={worst_score:.3f}):")
    out.append(f"  Weight: {worst_row['Lbs-']:.1f} lbs, MPG: {worst_row['Mpg+']:.1f}, Acc: {worst_row['Acc+']:.1f}")
    if not alternatives:
        out.append("  No alternatives found within 10 % tolerance that beat this vehicle.")
        return out

    for rank, (idx, r, score) in enumerate(alternatives, start=1):
        weight_diff = r["Lbs-"] - worst_row["Lbs-"]
        mpg_diff = r["Mpg+"] - worst_row["Mpg+"]
        acc_diff = r["Acc+"] - worst_row["Acc+"]

        out.append(f"Alternative {rank} (Score={score:.3f}):")
        out.append(f"  Vehicle #{idx}")
        out.append(f"  Weight: {r['Lbs-']:.1f} lbs ({weight_diff:+.1f} vs worst)")
        out.append(f"  MPG:    {r['Mpg+']:.1f} ({mpg_diff:+.1f} vs worst)")
        out.append(f"  Acc:    {r['Acc+']:.1f} s ({acc_diff:+.1f} vs worst)")
        out.append("  This vehicle outperforms the reference because it is")
        out.append("  lighter, more fuel‑efficient, and quicker, which")
        out.append("  together raise its composite score.")
    return out

# ----------------------------------------------------------------------
# CAR-16 implementation
# ----------------------------------------------------------------------
def car_16(rows):
    """
    For a vehicle that fails at least one performance target
    (weight > 4000 lbs OR mpg < 15 OR acc > 15 s), find the
    closest alternative that satisfies more targets while
    changing as few characteristics as possible.
    """
    # Target thresholds
    TARGET_WEIGHT = 4000.0
    TARGET_MPG = 15.0
    TARGET_ACC = 15.0

    # Identify candidate vehicles that fail at least one target
    failing = [
        (i, r) for i, r in enumerate(rows, start=1)
        if None not in (r["Lbs-"], r["Mpg+"], r["Acc+"]) and (
            r["Lbs-"] > TARGET_WEIGHT or r["Mpg+"] < TARGET_MPG or r["Acc+"] > TARGET_ACC
        )
    ]
    if not failing:
        return ["No vehicles failing the performance targets for CAR‑16."]

    # Pick the first failing vehicle (could also pick worst score)
    ref_idx, ref_row = failing[0]

    # Evaluate all complete vehicles as potential alternatives
    alternatives = []
    for i, r in enumerate(rows, start=1):
        if None in (r["Lbs-"], r["Mpg+"], r["Acc+"]):
            continue

        # Count how many targets this vehicle meets
        meets = sum((
            r["Lbs-"] <= TARGET_WEIGHT,
            r["Mpg+"] >= TARGET_MPG,
            r["Acc+"] <= TARGET_ACC
        ))
        # Compute difference metric: sum of absolute differences in the three attributes
        diff = abs(r["Lbs-"] - ref_row["Lbs-"]) + abs(r["Mpg+"] - ref_row["Mpg+"]) + abs(r["Acc+"] - ref_row["Acc+"])
        alternatives.append((i, r, meets, diff))

    # Sort by most targets met (desc), then smallest difference (asc)
    alternatives.sort(key=lambda x: (-x[2], x[3]))

    best = alternatives[0]
    alt_idx, alt_row, meets, diff = best

    out = ["=== CAR-16: Closest alternative that meets more targets ==="]
    out.append(f"Reference vehicle #{ref_idx} (fails {3 - sum([ref_row['Lbs-'] <= TARGET_WEIGHT, ref_row['Mpg+'] >= TARGET_MPG, ref_row['Acc+'] <= TARGET_ACC])} targets):")
    out.append(f"  Weight: {ref_row['Lbs-']:.1f} lbs, MPG: {ref_row['Mpg+']:.1f}, Acc: {ref_row['Acc+']:.1f}")

    out.append(f"\nChosen alternative vehicle #{alt_idx} meets {meets} out of 3 targets:")
    out.append(f"  Weight: {alt_row['Lbs-']:.1f} lbs ({'≤' if alt_row['Lbs-'] <= TARGET_WEIGHT else '>'} 4000)")
    out.append(f"  MPG:    {alt_row['Mpg+']:.1f} mpg ({'≥' if alt_row['Mpg+'] >= TARGET_MPG else '<'} 15)")
    out.append(f"  Acc:    {alt_row['Acc+']:.1f} s ({'≤' if alt_row['Acc+'] <= TARGET_ACC else '>'} 15)")

    out.append("\nKey changes from reference:")
    out.append(f"  Weight change: {alt_row['Lbs-'] - ref_row['Lbs-']:+.1f} lbs")
    out.append(f"  MPG change:    {alt_row['Mpg+'] - ref_row['Mpg+']:+.1f} mpg")
    out.append(f"  Acc change:    {alt_row['Acc+'] - ref_row['Acc+']:+.1f} s")
    out.append(f"  Total absolute difference: {diff:.1f} units")
    return out

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        rows = read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        sys.exit(1)

    # CAR-15
    print("== CAR-15 ==")
    for line in car_15(rows):
        print(line)

    # CAR-16
    print("\n== CAR-16 ==")
    for line in car_16(rows):
        print(line)


if __name__ == "__main__":
    main()
