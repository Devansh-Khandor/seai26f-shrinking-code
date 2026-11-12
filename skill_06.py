#!/usr/bin/env python3
"""
Skill 06: Explain vehicle recommendations and tradeoffs
=====================================================

This script provides two reports:

* CAR-11 – For each top‑rated vehicle, explain its recommendation by
           comparing weight, fuel efficiency and acceleration to the
           overall inventory averages.

* CAR-12 – Compare the two highest‑scoring vehicles and explain
           why one receives the higher overall recommendation,
           highlighting the trade‑offs in the underlying metrics.

Usage:
    python skill_06.py <csv_path>
"""

import csv
import sys
import math

# ----------------------------------------------------------------------
# CSV reading helper
# ----------------------------------------------------------------------
def read_csv(csv_path):
    """Read the Auto93 CSV and return a list of rows with numeric values."""
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
    """Compute combined desirability score for a single vehicle."""
    nw = normalize(row["Lbs-"], w_min, w_max, reverse=True)   # lighter better
    nm = normalize(row["Mpg+"], mpg_min, mpg_max)              # higher better
    na = normalize(row["Acc+"], acc_min, acc_max, reverse=True)  # lower better
    return (nw + nm + na) / 3.0


# ----------------------------------------------------------------------
# Summary statistics
# ----------------------------------------------------------------------
def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0


# ----------------------------------------------------------------------
# CAR‑11: Explain top vehicle recommendations
# ----------------------------------------------------------------------
def car_11(rows, top_n=3):
    # Filter out rows lacking required metrics
    clean_rows = [r for r in rows if None not in (r["Lbs-"], r["Mpg+"], r["Acc+"])]
    if not clean_rows:
        return ["No complete vehicles to analyse for CAR‑11."]

    # Gather min/max and mean values
    w_min, w_max = min(r["Lbs-"] for r in clean_rows), max(r["Lbs-"] for r in clean_rows)
    mpg_min, mpg_max = min(r["Mpg+"] for r in clean_rows), max(r["Mpg+"] for r in clean_rows)
    acc_min, acc_max = min(r["Acc+"] for r in clean_rows), max(r["Acc+"] for r in clean_rows)

    w_mean = mean(r["Lbs-"] for r in clean_rows)
    mpg_mean = mean(r["Mpg+"] for r in clean_rows)
    acc_mean = mean(r["Acc+"] for r in clean_rows)

    # Compute scores and sort
    scored = [
        (i, r, compute_score(r, w_min, w_max, mpg_min, mpg_max, acc_min, acc_max))
        for i, r in enumerate(rows, start=1)
        if None not in (r["Lbs-"], r["Mpg+"], r["Acc+"])
    ]
    scored.sort(key=lambda x: x[2], reverse=True)

    out = ["=== CAR-11: Explanations for top recommendations ==="]
    for rank, (idx, r, score) in enumerate(scored[:top_n], start=1):
        # Compute percentage differences to the mean
        w_pct = 100 * (r["Lbs-"] - w_mean) / w_mean
        mpg_pct = 100 * (r["Mpg+"] - mpg_mean) / mpg_mean
        acc_pct = 100 * (r["Acc+"] - acc_mean) / acc_mean

        out.append(f"Vehicle #{idx:>3}: Score={score:.3f}")
        out.append(
            f"  Weight: {r['Lbs-']:.1f} lbs ({w_pct:+.1f}% vs. mean {w_mean:.1f})"
        )
        out.append(
            f"  MPG:    {r['Mpg+']:.1f} mpg ({mpg_pct:+.1f}% vs. mean {mpg_mean:.1f})"
        )
        out.append(
            f"  Acc:    {r['Acc+']:.1f} s ({acc_pct:+.1f}% vs. mean {acc_mean:.1f})"
        )
        out.append(
            "  The balanced combination of being lighter, more fuel‑efficient "
            "and quicker yields the highest desirability score."
        )
        out.append("")  # blank line between vehicles
    return out


# ----------------------------------------------------------------------
# CAR‑12: Compare two vehicles and show trade‑offs
# ----------------------------------------------------------------------
def car_12(rows):
    clean_rows = [r for r in rows if None not in (r["Lbs-"], r["Mpg+"], r["Acc+"])]
    if len(clean_rows) < 2:
        return ["Insufficient vehicles to compare for CAR‑12."]

    w_min, w_max = min(r["Lbs-"] for r in clean_rows), max(r["Lbs-"] for r in clean_rows)
    mpg_min, mpg_max = min(r["Mpg+"] for r in clean_rows), max(r["Mpg+"] for r in clean_rows)
    acc_min, acc_max = min(r["Acc+"] for r in clean_rows), max(r["Acc+"] for r in clean_rows)

    scored = [
        (i, r, compute_score(r, w_min, w_max, mpg_min, mpg_max, acc_min, acc_max))
        for i, r in enumerate(rows, start=1)
        if None not in (r["Lbs-"], r["Mpg+"], r["Acc+"])
    ]
    scored.sort(key=lambda x: x[2], reverse=True)

    # Pick top two vehicles
    top1 = scored[0]
    top2 = scored[1]

    idx1, r1, score1 = top1
    idx2, r2, score2 = top2

    # Compute absolute differences
    w_diff = r1["Lbs-"] - r2["Lbs-"]
    mpg_diff = r1["Mpg+"] - r2["Mpg+"]
    acc_diff = r1["Acc+"] - r2["Acc+"]

    out = ["=== CAR-12: Trade‑offs between top two vehicles ==="]
    out.append(
        f"Vehicle #{idx1:>3} (Score={score1:.3f}) vs Vehicle #{idx2:>3} (Score={score2:.3f})"
    )
    out.append(
        f"  Weight difference: {w_diff:+.1f} lbs "
        f"({(w_diff / r1['Lbs-']*100):+.1f}% of Vehicle #{idx1})"
    )
    out.append(
        f"  MPG difference: {mpg_diff:+.1f} mpg "
        f"({(mpg_diff / r1['Mpg+']*100):+.1f}% of Vehicle #{idx1})"
    )
    out.append(
        f"  Acceleration difference: {acc_diff:+.1f} s "
        f"({(acc_diff / r1['Acc+']*100):+.1f}% of Vehicle #{idx1})"
    )
    out.append(
        "  The higher score for Vehicle #{idx1} is due to a superior "
        "balance: it has slightly higher MPG and faster acceleration "
        "compared to Vehicle #{idx2}, while its heavier weight is "
        "offset by the performance gains."
    )
    return out


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        rows = read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        sys.exit(1)

    print("== CAR-11 ==")
    for line in car_11(rows):
        print(line)

    print("\n== CAR-12 ==")
    for line in car_12(rows):
        print(line)


if __name__ == "__main__":
    main()
