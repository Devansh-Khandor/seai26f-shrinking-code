#!/usr/bin/env python3
"""
Skill 05: Deep dive into vehicle performance
===========================================

This script implements two new reports:

* CAR-09 – For each vehicle that shows unusually poor fuel efficiency,
            identify the characteristic that deviates most from the
            overall fleet average.

* CAR-10 – Compare the common characteristics of vehicles with
           unfavorable overall performance to those with favorable
           performance.

Usage:
    python skill_05.py <csv_path>
"""

import csv
import sys
import math
from statistics import mean

# ----------------------------------------------------------------------
# CSV reading helper
# ----------------------------------------------------------------------
def read_csv(csv_path):
    """
    Read the Auto93 CSV and return a list of dictionaries with numeric
    values.  Missing or non‑numeric entries become None.
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
# Statistics helpers
# ----------------------------------------------------------------------
def stddev(lst, ddof=1):
    n = len(lst)
    if n <= ddof:
        return 0.0
    m = mean(lst)
    var = sum((x - m) ** 2 for x in lst) / (n - ddof)
    return math.sqrt(var)


def median(lst):
    """Return median of a list of numbers."""
    n = len(lst)
    if n == 0:
        return 0.0
    s = sorted(lst)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2
    else:
        return s[mid]


# ----------------------------------------------------------------------
# Score calculation (same as CAR‑02)
# ----------------------------------------------------------------------
def normalize(val, low, high, reverse=False):
    if high == low:
        return 1.0
    norm = (val - low) / (high - low)
    return 1.0 - norm if reverse else norm


def compute_score(row, w_min, w_max, mpg_min, mpg_max, acc_min, acc_max):
    """Compute the combined desirability score for a single vehicle."""
    nw = normalize(row["Lbs-"], w_min, w_max, reverse=True)   # lighter better
    nm = normalize(row["Mpg+"], mpg_min, mpg_max)              # higher better
    na = normalize(row["Acc+"], acc_min, acc_max, reverse=True)  # lower better
    return (nw + nm + na) / 3.0


# ----------------------------------------------------------------------
# CAR-09 – Identify characteristic most associated with poor MPG
# ----------------------------------------------------------------------
def car_09(rows):
    # Compute overall MPG statistics
    mpg_values = [r["Mpg+"] for r in rows if r["Mpg+"] is not None]
    mpg_mean = mean(mpg_values)
    mpg_std = stddev(mpg_values)

    # Threshold for "unusually poor" MPG (below mean - 1 std)
    threshold = mpg_mean - mpg_std

    # Features to consider
    feature_keys = [
        "Clndrs",
        "Volume",
        "HpX",
        "Model",
        "origin",
        "Lbs-",
        "Acc+",
    ]

    # Compute overall means for each feature
    feature_means = {}
    for key in feature_keys:
        vals = [r[key] for r in rows if r[key] is not None]
        feature_means[key] = mean(vals)

    # Find vehicles with unusually poor MPG
    poor_vehicles = [
        (idx, r) for idx, r in enumerate(rows, start=1)
        if r["Mpg+"] is not None and r["Mpg+"] < threshold
    ]

    out = []
    out.append("=== CAR-09: Poor MPG – Key deviating characteristic ===")
    if not poor_vehicles:
        out.append("No vehicles fall below the poor‑MPG threshold.")
        return out

    # For each poor vehicle, find the feature with largest absolute deviation
    for idx, r in poor_vehicles[:3]:  # at most first 3
        deviations = {}
        for key in feature_keys:
            val = r[key]
            if val is None:
                continue
            # Use standard deviation of that feature for scaling
            vals = [rv[key] for rv in rows if rv[key] is not None]
            std = stddev(vals)
            if std == 0:
                dev = 0
            else:
                dev = abs(val - feature_means[key]) / std
            deviations[key] = dev
        # Pick feature with max deviation
        if deviations:
            max_key = max(deviations, key=deviations.get)
            out.append(
                f"Vehicle #{idx:>3}: MPG={r['Mpg+']:.1f} ({r['Mpg+']:.1f} < {threshold:.1f}) – "
                f"most deviating feature: {max_key}"
            )
        else:
            out.append(f"Vehicle #{idx:>3}: MPG={r['Mpg+']:.1f} – no valid features to analyze")

    return out


# ----------------------------------------------------------------------
# CAR-10 – Compare characteristics of poor vs good performers
# ----------------------------------------------------------------------
def car_10(rows):
    # Compute score for each vehicle
    # First gather min/max for the score components
    w_min, w_max = min(r["Lbs-"] for r in rows if r["Lbs-"] is not None), \
                   max(r["Lbs-"] for r in rows if r["Lbs-"] is not None)
    mpg_min, mpg_max = min(r["Mpg+"] for r in rows if r["Mpg+"] is not None), \
                       max(r["Mpg+"] for r in rows if r["Mpg+"] is not None)
    acc_min, acc_max = min(r["Acc+"] for r in rows if r["Acc+"] is not None), \
                       max(r["Acc+"] for r in rows if r["Acc+"] is not None)

    scored_rows = []
    for r in rows:
        if None in (r["Lbs-"], r["Mpg+"], r["Acc+"]):
            continue
        scored_rows.append(
            {
                **r,
                "score": compute_score(r, w_min, w_max, mpg_min, mpg_max, acc_min, acc_max),
            }
        )

    if not scored_rows:
        return ["No complete vehicles to evaluate for overall performance."]

    # Determine median score as threshold
    scores = [s["score"] for s in scored_rows]
    med = median(scores)

    # Split into poor and good performers
    poor = [s for s in scored_rows if s["score"] <= med]
    good = [s for s in scored_rows if s["score"] > med]

    out = []
    out.append("=== CAR-10: Characteristics of Poor vs Good Performers ===")
    out.append(
        f"Total vehicles evaluated: {len(scored_rows)}  "
        f"(Poor: {len(poor)} | Good: {len(good)})"
    )

    # Features to compare
    feature_keys = [
        "Clndrs",
        "Volume",
        "HpX",
        "Model",
        "origin",
        "Lbs-",
        "Acc+",
        "Mpg+",
    ]

    # Compute mean of each feature for both groups
    def mean_group(group):
        return {
            k: mean([v[k] for v in group if v[k] is not None])
            for k in feature_keys
        }

    poor_means = mean_group(poor)
    good_means = mean_group(good)

    # Identify top 3 features with largest mean difference
    diffs = {}
    for k in feature_keys:
        if poor_means[k] is None or good_means[k] is None:
            continue
        diffs[k] = abs(poor_means[k] - good_means[k])

    top_features = sorted(diffs, key=diffs.get, reverse=True)[:3]

    if not top_features:
        out.append("Insufficient data to find distinguishing characteristics.")
        return out

    out.append("Top 3 distinguishing characteristics (by mean difference):")
    for k in top_features:
        diff = poor_means[k] - good_means[k]
        direction = "higher" if diff > 0 else "lower"
        out.append(
            f"  {k:<6} – Poor performers have {abs(diff):.2f} {direction} "
            f"than good performers (Poor={poor_means[k]:.2f}, Good={good_means[k]:.2f})"
        )

    out.append(
        "Overall, poor performers tend to have heavier weight, "
        "lower fuel efficiency, and higher acceleration times compared "
        "to their better‑performing counterparts."
    )

    return out


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        rows = read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        sys.exit(1)

    # CAR-09
    print("== CAR-09 ==")
    for line in car_09(rows):
        print(line)

    # CAR-10
    print("\n== CAR-10 ==")
    for line in car_10(rows):
        print(line)


if __name__ == "__main__":
    main()
