#!/usr/bin/env python3
"""
Skill 07: Quick identification of balanced vehicles and inspection prioritisation
===============================================================================

This script implements two reports:

* CAR-13 – Identify the vehicle that best balances fuel efficiency,
           weight, and acceleration.  The vehicle with the highest
           composite score is reported.  The calculation uses the same
           normalisation strategy as earlier skills, but stops once the
           best vehicle is found, keeping the evaluation workload small.

* CAR-14 – When the dealership has a limited inspection budget,
           recommend the vehicles that should be inspected first.  The
           vehicles are sorted by the composite score (higher is
           better), and the top N (default 5) are listed.

Usage:
    python skill_07.py <csv_path>
"""

import csv
import sys
import math
from statistics import mean

# ----------------------------------------------------------------------
# CSV reading helper
# ----------------------------------------------------------------------
def read_csv(csv_path):
    """Read Auto93 CSV and return a list of rows with numeric values."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            parsed = {}
            for key in (
                "Clndrs",
                "Volume",
                "HpX",
                "Model",
                "origin",
                "Lbs-",
                "Acc+",
                "Mpg+",
            ):
                try:
                    parsed[key] = float(r[key])
                except (KeyError, ValueError, TypeError):
                    parsed[key] = None
            rows.append(parsed)
    return rows

# ----------------------------------------------------------------------
# Normalisation and score
# ----------------------------------------------------------------------
def normalize(val, low, high, reverse=False):
    """Return a 0–1 normalised value, optionally reversed."""
    if high == low or val is None:
        return 0.5
    norm = (val - low) / (high - low)
    return 1.0 - norm if reverse else norm

def compute_score(row, w_min, w_max, mpg_min, mpg_max, acc_min, acc_max):
    """Compute composite desirability score for a single vehicle."""
    nw = normalize(row["Lbs-"], w_min, w_max, reverse=True)   # lighter better
    nm = normalize(row["Mpg+"], mpg_min, mpg_max)              # higher better
    na = normalize(row["Acc+"], acc_min, acc_max, reverse=True)  # lower better
    return (nw + nm + na) / 3.0

# ----------------------------------------------------------------------
# CAR-13: Identify best balanced vehicle
# ----------------------------------------------------------------------
def car_13(rows):
    """Return a list of strings describing the vehicle with the best balance."""
    # Filter out incomplete records
    good_rows = [r for r in rows if None not in (r["Lbs-"], r["Mpg+"], r["Acc+"])]
    if not good_rows:
        return ["No complete vehicles to evaluate for CAR-13."]

    # Compute min/max for normalisation
    w_min, w_max = min(r["Lbs-"] for r in good_rows), max(r["Lbs-"] for r in good_rows)
    mpg_min, mpg_max = min(r["Mpg+"] for r in good_rows), max(r["Mpg+"] for r in good_rows)
    acc_min, acc_max = min(r["Acc+"] for r in good_rows), max(r["Acc+"] for r in good_rows)

    # Scan once to find best score
    best_idx = None
    best_score = -1.0
    for idx, r in enumerate(good_rows, start=1):
        score = compute_score(r, w_min, w_max, mpg_min, mpg_max, acc_min, acc_max)
        if score > best_score:
            best_score = score
            best_idx = idx
            best_row = r

    out = ["=== CAR-13: Vehicle with the best balance ==="]
    out.append(f"Vehicle #{best_idx}:")
    out.append(f"  Weight: {best_row['Lbs-']:.1f} lbs")
    out.append(f"  MPG:    {best_row['Mpg+']:.1f} mpg")
    out.append(f"  Acc:    {best_row['Acc+']:.1f} s")
    out.append(f"  Composite score: {best_score:.3f}")
    return out

# ----------------------------------------------------------------------
# CAR-14: Prioritise inspection order
# ----------------------------------------------------------------------
def car_14(rows, budget=5):
    """Return a list of strings recommending vehicles to inspect first."""
    good_rows = [r for r in rows if None not in (r["Lbs-"], r["Mpg+"], r["Acc+"])]
    if not good_rows:
        return ["No complete vehicles to evaluate for CAR-14."]

    w_min, w_max = min(r["Lbs-"] for r in good_rows), max(r["Lbs-"] for r in good_rows)
    mpg_min, mpg_max = min(r["Mpg+"] for r in good_rows), max(r["Mpg+"] for r in good_rows)
    acc_min, acc_max = min(r["Acc+"] for r in good_rows), max(r["Acc+"] for r in good_rows)

    # Compute score for every vehicle
    scored = []
    for idx, r in enumerate(good_rows, start=1):
        score = compute_score(r, w_min, w_max, mpg_min, mpg_max, acc_min, acc_max)
        scored.append((idx, r, score))

    # Sort descending by score
    scored.sort(key=lambda x: x[2], reverse=True)

    out = [f"=== CAR-14: Inspection priority (top {budget} vehicles) ==="]
    for rank, (idx, r, score) in enumerate(scored[:budget], start=1):
        out.append(f"{rank}. Vehicle #{idx}: Score={score:.3f}")
        out.append(f"   Weight: {r['Lbs-']:.1f} lbs, MPG: {r['Mpg+']:.1f}, Acc: {r['Acc+']:.1f} s")
    return out

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        rows = read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        sys.exit(1)

    print("== CAR-13 ==")
    for line in car_13(rows):
        print(line)

    print("\n== CAR-14 ==")
    for line in car_14(rows, budget=5):
        print(line)

if __name__ == "__main__":
    main()
