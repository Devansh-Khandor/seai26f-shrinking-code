#!/usr/bin/env python3
"""
Skill 09: Outlier detection and consistency checks
=================================================

This script implements two reports:

* CAR-17 – Flag vehicle records whose combinations of characteristics
            are unusually different from the rest of the inventory,
            so that staff can review them.

* CAR-18 – Identify vehicles whose reported fuel efficiency or weight
            differs substantially from what similar vehicles would
            lead us to expect, so we can check for possible recording
            mistakes.

Usage:
    python skill_09.py <csv_path>
"""

import csv
import sys
import math
from collections import defaultdict

# ----------------------------------------------------------------------
# CSV reading
# ----------------------------------------------------------------------
def read_csv(csv_path):
    """
    Read Auto93 CSV and return a list of rows with numeric values
    or None for missing data.
    """
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
# Simple statistics helpers
# ----------------------------------------------------------------------
def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0


def stddev(lst):
    n = len(lst)
    if n <= 1:
        return 0.0
    m = mean(lst)
    var = sum((x - m) ** 2 for x in lst) / (n - 1)
    return math.sqrt(var)


# ----------------------------------------------------------------------
# CAR-17: Unusual overall combinations
# ----------------------------------------------------------------------
def car_17(rows):
    """
    Detect vehicles with unusually large Mahalanobis-like z‑score sum
    across all numeric attributes.  Records with a sum of squared
    z‑scores greater than 9 (≈3 σ in a 3‑dimensional space) are
    flagged.  Three or more records are reported; if fewer exist
    we list all flagged vehicles.
    """
    # Select numeric keys to analyse
    keys = ["Lbs-", "Mpg+", "Acc+", "Clndrs", "Volume", "HpX", "Model", "origin"]
    # Build lists of complete records
    complete = [r for r in rows if all(r[k] is not None for k in keys)]

    if not complete:
        return ["No complete vehicle data available for CAR‑17."]

    # Compute means and stddevs
    stats = {}
    for k in keys:
        vals = [r[k] for r in complete]
        stats[k] = (mean(vals), stddev(vals))

    # Identify outliers
    flagged = []
    for idx, r in enumerate(rows, start=1):
        if any(r[k] is None for k in keys):
            continue
        # Sum of squared z‑scores
        z_sum = 0.0
        for k in keys:
            mu, sigma = stats[k]
            if sigma == 0:
                continue
            z = (r[k] - mu) / sigma
            z_sum += z ** 2
        if z_sum > 9:  # roughly 3‑sigma threshold in 4‑dim space
            flagged.append((idx, r, z_sum))

    out = ["=== CAR-17: Vehicles with unusual attribute combinations ==="]
    if not flagged:
        out.append("No vehicles exceeded the outlier threshold.")
        return out

    flagged.sort(key=lambda x: x[2], reverse=True)
    # Limit to at least 3 lines of results, but report all flagged
    for idx, r, z_sum in flagged[:max(3, len(flagged))]:
        out.append(
            f"Vehicle #{idx:>3}: Z²‑sum={z_sum:.2f} | "
            f"Lbs={r['Lbs-']:.1f} | MPG={r['Mpg+']:.1f} | Acc={r['Acc+']:.1f} | "
            f"Cyl={int(r['Clndrs']):>2} | Vol={int(r['Volume']):>3} | "
            f"Hp={int(r['HpX']):>3} | Model={int(r['Model']):>3}"
        )
    return out


# ----------------------------------------------------------------------
# CAR-18: Consistency checks via nearest‑neighbour prediction
# ----------------------------------------------------------------------
def car_18(rows, k=5, pct_threshold=0.20):
    """
    For each vehicle with complete weight (Lbs-) and mpg (Mpg+),
    find its k nearest neighbours using Euclidean distance on
    Clndrs, Volume, HpX, and Model.  Compute the mean weight and
    mpg of those neighbours and compare to the vehicle's own
    values.  If the absolute difference exceeds pct_threshold
    (default 20 %) for either metric, flag the vehicle for
    potential recording error.
    """
    # Prepare complete rows
    complete = [r for r in rows if r["Lbs-"] is not None and r["Mpg+"] is not None]
    if len(complete) < k + 1:
        return ["Not enough complete records for CAR‑18 comparison."]

    # Helper: compute Euclidean distance between two rows on selected keys
    def dist(r1, r2):
        d = 0.0
        for key in ("Clndrs", "Volume", "HpX", "Model"):
            d += (r1[key] - r2[key]) ** 2
        return math.sqrt(d)

    flagged = []
    for idx, r in enumerate(rows, start=1):
        if r["Lbs-"] is None or r["Mpg+"] is None:
            continue
        # Compute distances to all other complete rows
        dists = [(j, dist(r, rr)) for j, rr in enumerate(complete) if rr is not r]
        if not dists:
            continue
        # Take k nearest neighbours
        dists.sort(key=lambda x: x[1])
        neighbours = [complete[j] for j, _ in dists[:k]]

        # Predict mean weight and mpg from neighbours
        pred_w = mean([n["Lbs-"] for n in neighbours])
        pred_m = mean([n["Mpg+"] for n in neighbours])

        # Compute percent differences
        w_diff = abs(r["Lbs-"] - pred_w) / pred_w
        m_diff = abs(r["Mpg+"] - pred_m) / pred_m

        if w_diff > pct_threshold or m_diff > pct_threshold:
            flagged.append(
                (
                    idx,
                    r,
                    pred_w,
                    pred_m,
                    w_diff,
                    m_diff,
                )
            )

    out = ["=== CAR-18: Vehicles with inconsistent weight or mpg ==="]
    if not flagged:
        out.append("No vehicles exceeded the consistency threshold.")
        return out

    flagged.sort(key=lambda x: max(x[4], x[5]), reverse=True)
    # Limit output to at least 3 lines, but include all flagged
    for idx, r, pred_w, pred_m, w_diff, m_diff in flagged[:max(3, len(flagged))]:
        out.append(
            f"Vehicle #{idx:>3}: "
            f"Lbs={r['Lbs-']:.1f} (pred={pred_w:.1f}, Δ={w_diff*100:>5.1f}%) | "
            f"MPG={r['Mpg+']:.1f} (pred={pred_m:.1f}, Δ={m_diff*100:>5.1f}%)"
        )
    return out


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        rows = read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        sys.exit(1)

    # CAR-17
    print("== CAR-17 ==")
    for line in car_17(rows):
        print(line)

    # CAR-18
    print("\n== CAR-18 ==")
    for line in car_18(rows):
        print(line)


if __name__ == "__main__":
    main()
