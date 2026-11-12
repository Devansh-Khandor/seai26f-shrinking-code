#!/usr/bin/env python3
"""
Skill 01: Inventory summary and attractive vehicle selection
============================================================

This script reads the Auto93 dataset and produces two reports:

* CAR-01  – Summary of vehicle weight, fuel efficiency (MPG) and acceleration.
* CAR-02  – Top vehicles according to a combined score that favours
            low weight, high MPG and low acceleration time.

Usage:
    python skill_01.py <csv_path>
"""

import csv
import sys
from statistics import mean


def read_data(csv_path):
    """
    Read the CSV file and return a list of dictionaries with numeric fields.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Convert relevant columns to float; ignore missing or non‑numeric
            try:
                row = {
                    "Lbs-": float(r["Lbs-"]),
                    "Acc+": float(r["Acc+"]),
                    "Mpg+": float(r["Mpg+"]),
                }
                rows.append(row)
            except (KeyError, ValueError):
                # Skip rows that lack the required columns or cannot be parsed
                continue
    return rows


def car_01_report(rows):
    """
    Generate CAR-01 report: typical (mean), best and worst values
    for weight, mpg and acceleration.
    """
    weights = [r["Lbs-"] for r in rows]
    mpgs = [r["Mpg+"] for r in rows]
    accs = [r["Acc+"] for r in rows]

    w_mean, w_min, w_max = mean(weights), min(weights), max(weights)
    mpg_mean, mpg_min, mpg_max = mean(mpgs), min(mpgs), max(mpgs)
    acc_mean, acc_min, acc_max = mean(accs), min(accs), max(accs)

    lines = [
        f"Typical weight: {w_mean:.1f} lbs, Best weight: {w_min:.1f} lbs, Worst weight: {w_max:.1f} lbs",
        f"Typical mpg:   {mpg_mean:.1f} mpg, Best mpg:   {mpg_max:.1f} mpg, Worst mpg:   {mpg_min:.1f} mpg",
        f"Typical acc.:  {acc_mean:.1f} sec, Best acc.:  {acc_min:.1f} sec, Worst acc.:  {acc_max:.1f} sec",
    ]
    return lines


def car_02_report(rows, top_n=5):
    """
    Generate CAR-02 report: rank vehicles by combined score.
    Score = (norm_weight + norm_mpg + norm_acc) / 3
    where lighter weight, higher mpg, and lower acceleration are better.
    """
    if not rows:
        return ["No data to rank vehicles."]

    # Determine min/max for each metric
    w_min, w_max = min(r["Lbs-"] for r in rows), max(r["Lbs-"] for r in rows)
    mpg_min, mpg_max = min(r["Mpg+"] for r in rows), max(r["Mpg+"] for r in rows)
    acc_min, acc_max = min(r["Acc+"] for r in rows), max(r["Acc+"] for r in rows)

    def norm(val, low, high, reverse=False):
        """Normalize a value between 0 and 1. If reverse is True,
        higher original values give lower normalized values."""
        if high == low:
            return 1.0
        normalized = (val - low) / (high - low)
        return 1.0 - normalized if reverse else normalized

    # Compute scores
    scored = []
    for idx, r in enumerate(rows):
        nw = norm(r["Lbs-"], w_min, w_max, reverse=True)   # lighter better
        nm = norm(r["Mpg+"], mpg_min, mpg_max)              # higher better
        na = norm(r["Acc+"], acc_min, acc_max, reverse=True)  # lower better
        score = (nw + nm + na) / 3
        scored.append((idx, r, score))

    # Sort descending by score
    scored.sort(key=lambda x: x[2], reverse=True)

    lines = [f"Top {min(top_n, len(rows))} vehicles (best overall combination):"]
    for rank, (idx, r, score) in enumerate(scored[:top_n], start=1):
        lines.append(
            f"{rank}. Weight: {r['Lbs-']:.1f} lbs, MPG: {r['Mpg+']:.1f}, Acc: {r['Acc+']:.1f} sec, Score: {score:.3f}"
        )
    return lines


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_data(csv_path)

    if not rows:
        print("No valid data found in the CSV.")
        sys.exit(1)

    # CAR-01
    print("== CAR-01 ==")
    for line in car_01_report(rows):
        print(line)

    # CAR-02
    print("\n== CAR-02 ==")
    for line in car_02_report(rows, top_n=5):
        print(line)


if __name__ == "__main__":
    main()
