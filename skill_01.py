#!/usr/bin/env python3
"""
skill_01.py

Usage:
    python skill_01.py <csv_path>

The script prints:
    * CAR-01: Summary statistics for weight, fuel efficiency (mpg) and acceleration.
    * CAR-02: Top 5 vehicles with the most attractive overall combination of
      low weight, high mpg, and favorable acceleration.
"""

import sys
import csv
import math
from pathlib import Path

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def read_csv(csv_path: Path):
    """Read the CSV file and return a list of vehicle dicts with numeric fields."""
    data = []
    with csv.DictReader(open(csv_path, newline='')) as rdr:
        for row in rdr:
            try:
                vehicle = {
                    'Clndrs': int(row['Clndrs']),
                    'Volume': float(row['Volume']),
                    'HpX': float(row['HpX']),
                    'Model': int(row['Model']),
                    'origin': int(row['origin']),
                    'Lbs-': float(row['Lbs-']),
                    'Acc+': float(row['Acc+']),
                    'Mpg+': float(row['Mpg+']),
                }
                data.append(vehicle)
            except (KeyError, ValueError) as exc:
                # Skip rows that don't have the expected format
                continue
    return data

def mean(values):
    return sum(values) / len(values) if values else float('nan')

def format_number(value, decimals=2):
    """Format a number to a string with fixed decimals, stripping trailing zeros."""
    if math.isnan(value):
        return "nan"
    return f"{value:.{decimals}f}".rstrip('0').rstrip('.') if '.' in f"{value:.{decimals}f}" else f"{value:.{decimals}f}"

# --------------------------------------------------------------------------- #
# CAR-01: Summary statistics
# --------------------------------------------------------------------------- #

def print_car_01_summary(data):
    weights = [v['Lbs-'] for v in data]
    mpgs    = [v['Mpg+'] for v in data]
    accs    = [v['Acc+'] for v in data]

    mean_w = mean(weights)
    mean_m = mean(mpgs)
    mean_a = mean(accs)

    best_w = min(weights)
    best_m = max(mpgs)
    best_a = min(accs)  # lowest acceleration time considered best

    worst_w = max(weights)
    worst_m = min(mpgs)
    worst_a = max(accs)  # highest time considered worst

    print("== CAR-01 ==")
    print(f"Mean weight: {format_number(mean_w)} kg, Mean mpg: {format_number(mean_m)}, Mean acceleration: {format_number(mean_a)}")
    print(f"Best (lightest) weight: {format_number(best_w)} kg")
    print(f"Best (highest mpg): {format_number(best_m)} mpg")
    print(f"Best (fastest acceleration): {format_number(best_a)}")
    print(f"Worst (heaviest) weight: {format_number(worst_w)} kg")
    print(f"Worst (lowest mpg): {format_number(worst_m)} mpg")
    print(f"Worst (slowest acceleration): {format_number(worst_a)}")

# --------------------------------------------------------------------------- #
# CAR-02: Composite scoring
# --------------------------------------------------------------------------- #

def compute_composite_scores(data):
    """Return a list of tuples (score, vehicle_dict)."""
    mean_w = mean([v['Lbs-'] for v in data])
    mean_m = mean([v['Mpg+'] for v in data])
    mean_a = mean([v['Acc+'] for v in data])

    scores = []
    for v in data:
        weight_norm = v['Lbs-'] / mean_w
        acc_norm    = v['Acc+'] / mean_a
        mpg_norm    = v['Mpg+'] / mean_m
        score = weight_norm + acc_norm - mpg_norm
        scores.append((score, v))
    return scores

def print_car_02_best(data, top_n=5):
    scores = compute_composite_scores(data)
    scores.sort(key=lambda x: x[0])  # lower score = better

    print("\n== CAR-02 ==")
    print(f"Top {top_n} vehicles with best overall combination:")
    for i, (score, v) in enumerate(scores[:top_n], start=1):
        print(
            f"{i}. Weight: {format_number(v['Lbs-'])} kg, "
            f"MPG: {format_number(v['Mpg+'])}, "
            f"Acc: {format_number(v['Acc+'])}, "
            f"Composite Score: {format_number(score)}"
        )

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    data = read_csv(csv_path)
    if not data:
        print("No data found in CSV.", file=sys.stderr)
        sys.exit(1)

    print_car_01_summary(data)
    print_car_02_best(data)

if __name__ == "__main__":
    main()
