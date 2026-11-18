#!/usr/bin/env python3
"""
skill_01.py

Usage:
    python skill_01.py <csv path>

Prints a summary of the inventory (CAR-01) and identifies the top 10 vehicles
with the best overall combination of low weight, high fuel efficiency, and
favorable acceleration (CAR-02).  The output also contains machine‑checked
answers in the required JSON format.
"""

import sys
import csv
import json
import math
import numpy as np

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def read_csv(path):
    """
    Read the CSV and return:
      - header: list of column names
      - data: list of rows (each row is a list of values as strings)
    """
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)
    return header, data

def parse_float(value):
    """
    Convert a value to float or return None if it is missing.
    """
    if value == '?' or value == '':
        return None
    try:
        return float(value)
    except ValueError:
        return None

def collect_column_values(data, idx):
    """
    Return a list of float values for the column at idx, skipping missing entries.
    """
    vals = []
    for row in data:
        val = parse_float(row[idx])
        if val is not None:
            vals.append(val)
    return np.array(vals)

def compute_stats(vals, maximize):
    """
    Compute mean, median, min, max, best, worst.
    - If maximize is True, best=max, worst=min.
    - If maximize is False, best=min, worst=max.
    """
    mean = float(np.mean(vals))
    median = float(np.median(vals))
    minimum = float(np.min(vals))
    maximum = float(np.max(vals))
    if maximize:
        best = maximum
        worst = minimum
    else:
        best = minimum
        worst = maximum
    return {
        'mean': mean,
        'median': median,
        'min': minimum,
        'max': maximum,
        'best': best,
        'worst': worst
    }

def normalize(val, minimum, maximum):
    """
    Normalize a value to the interval [0, 1].
    If maximum == minimum, return 0.0 to avoid division by zero.
    """
    if maximum == minimum:
        return 0.0
    return (val - minimum) / (maximum - minimum)

def compute_top_rows(rows, idx_lbs, idx_acc, idx_mpg, top_n=10):
    """
    Compute the top `top_n` rows based on a combined score.
    Lower Lbs- and Acc+ are better; higher Mpg+ is better.
    Returns a list of row numbers (1-based).
    """
    # Extract only rows with all three values present
    valid = []
    for rn, row in enumerate(rows, start=1):
        lbs = parse_float(row[idx_lbs])
        acc = parse_float(row[idx_acc])
        mpg = parse_float(row[idx_mpg])
        if lbs is not None and acc is not None and mpg is not None:
            valid.append((rn, lbs, acc, mpg))

    # Compute min and max for each metric across valid rows
    lbs_vals = [t[1] for t in valid]
    acc_vals = [t[2] for t in valid]
    mpg_vals = [t[3] for t in valid]

    lbs_min, lbs_max = min(lbs_vals), max(lbs_vals)
    acc_min, acc_max = min(acc_vals), max(acc_vals)
    mpg_min, mpg_max = min(mpg_vals), max(mpg_vals)

    # Compute scores
    scored = []
    for rn, lbs, acc, mpg in valid:
        # Lower is better for lbs and acc
        norm_lbs = normalize(lbs, lbs_min, lbs_max)
        norm_acc = normalize(acc, acc_min, acc_max)
        norm_mpg = normalize(mpg, mpg_min, mpg_max)

        score = (1 - norm_lbs) + (1 - norm_acc) + norm_mpg
        scored.append((score, rn))

    # Sort by score descending
    scored.sort(reverse=True, key=lambda x: x[0])
    top_rows = [rn for _, rn in scored[:top_n]]
    return top_rows

# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def main(csv_path):
    header, data = read_csv(csv_path)

    # Identify indices of relevant columns
    idx_lbs = header.index('Lbs-')
    idx_acc = header.index('Acc+')
    idx_mpg = header.index('Mpg+')

    # Collect values
    lbs_vals = collect_column_values(data, idx_lbs)
    acc_vals = collect_column_values(data, idx_acc)
    mpg_vals = collect_column_values(data, idx_mpg)

    # Compute stats
    lbs_stats = compute_stats(lbs_vals, maximize=False)  # minimize
    acc_stats = compute_stats(acc_vals, maximize=False)  # minimize
    mpg_stats = compute_stats(mpg_vals, maximize=True)   # maximize

    # -------------------------------------------------- CAR-01 --------------------------------------------------
    print("== CAR-01 ==")
    print(f"Lbs-: mean={lbs_stats['mean']:.2f}  median={lbs_stats['median']:.2f}  min={lbs_stats['min']:.2f}  max={lbs_stats['max']:.2f}  best={lbs_stats['best']:.2f}  worst={lbs_stats['worst']:.2f}")
    print(f"Acc+: mean={acc_stats['mean']:.2f}  median={acc_stats['median']:.2f}  min={acc_stats['min']:.2f}  max={acc_stats['max']:.2f}  best={acc_stats['best']:.2f}  worst={acc_stats['worst']:.2f}")
    print(f"Mpg+: mean={mpg_stats['mean']:.2f}  median={mpg_stats['median']:.2f}  min={mpg_stats['min']:.2f}  max={mpg_stats['max']:.2f}  best={mpg_stats['best']:.2f}  worst={mpg_stats['worst']:.2f}")

    # Machine‑checked answer for CAR‑01
    answer_car01 = {
        "Lbs-": lbs_stats,
        "Acc+": acc_stats,
        "Mpg+": mpg_stats
    }
    print(f"ANSWER CAR-01 {json.dumps(answer_car01, separators=(',', ':'), sort_keys=True)}")

    # -------------------------------------------------- CAR-02 --------------------------------------------------
    top_rows = compute_top_rows(data, idx_lbs, idx_acc, idx_mpg, top_n=10)

    print("\n== CAR-02 ==")
    print(f"Top 10 rows: {top_rows}")
    print(f"Row {top_rows[0]} has the best combined score.")

    # Machine‑checked answer for CAR‑02
    answer_car02 = {"top": top_rows}
    print(f"ANSWER CAR-02 {json.dumps(answer_car02, separators=(',', ':'), sort_keys=True)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
