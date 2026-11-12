#!/usr/bin/env python3
"""
skill_04.py

Usage:
    python skill_04.py <csv_path>

The script prints:
    * CAR-07: For each new vehicle record, determine if it is among
      the most desirable inventory options.
    * CAR-08: Updated inventory summaries and top‑5 recommended vehicles
      after processing the latest records.
"""

import sys
import csv
import math
from pathlib import Path

# Import helper functions from the previous skills
from skill_01 import read_csv, compute_composite_scores, format_number, mean

# --------------------------------------------------------------------------- #
# CAR-07: Desirability assessment for each vehicle
# --------------------------------------------------------------------------- #

def print_car_07_assessment(data):
    """
    For every vehicle compute a composite score and flag it as desirable
    if it is within the top 5 (or top 10%) best scores.  Print only the
    desirable vehicles (at least three lines of output).
    """
    # Calculate composite scores
    scores = compute_composite_scores(data)

    # Sort by score (lower is better)
    scores.sort(key=lambda x: x[0])

    # Determine threshold for top vehicles (top 5 or 10% whichever is larger)
    top_n = 5
    pct_top = 0.10
    n = len(scores)
    if n == 0:
        print("\n== CAR-07 ==")
        print("  No vehicle records to evaluate.")
        return

    threshold_index = min(top_n, max(1, int(math.ceil(pct_top * n))))
    threshold_score = scores[threshold_index - 1][0]

    # Gather desirable vehicles
    desirable = [ (score, v) for score, v in scores if score <= threshold_score ]

    print("\n== CAR-07 ==")
    print("Desirable vehicles (top {}) according to composite score:".format(len(desirable)))
    # Print each desirable vehicle
    for rank, (score, v) in enumerate(desirable, start=1):
        print(f"  {rank}. Clndrs={v['Clndrs']}, Model={v['Model']}, "
              f"Weight={format_number(v['Lbs-'])} kg, "
              f"MPG={format_number(v['Mpg+'])}, Acc={format_number(v['Acc+'])}, "
              f"Composite Score={format_number(score)}")

# --------------------------------------------------------------------------- #
# CAR-08: Updated inventory summary and top‑5 recommendation
# --------------------------------------------------------------------------- #

def print_car_08_inventory_summary(data):
    """
    Print updated inventory summaries (mean, best/worst) and the top‑5
    recommended vehicles using the composite score.
    """
    if not data:
        print("\n== CAR-08 ==")
        print("  No vehicle records to summarize.")
        return

    # Summary statistics
    weights = [v['Lbs-'] for v in data]
    mpgs    = [v['Mpg+'] for v in data]
    accs    = [v['Acc+'] for v in data]

    mean_w = mean(weights)
    mean_m = mean(mpgs)
    mean_a = mean(accs)

    best_w = min(weights)
    best_m = max(mpgs)
    best_a = min(accs)

    worst_w = max(weights)
    worst_m = min(mpgs)
    worst_a = max(accs)

    print("\n== CAR-08 ==")
    print("Updated inventory summary:")
    print(f"  Mean weight: {format_number(mean_w)} kg, Mean mpg: {format_number(mean_m)}, Mean acceleration: {format_number(mean_a)}")
    print(f"  Lightest weight: {format_number(best_w)} kg")
    print(f"  Highest mpg: {format_number(best_m)} mpg")
    print(f"  Fastest acceleration: {format_number(best_a)}")
    print(f"  Heaviest weight: {format_number(worst_w)} kg")
    print(f"  Lowest mpg: {format_number(worst_m)} mpg")
    print(f"  Slowest acceleration: {format_number(worst_a)}")

    # Top‑5 vehicles based on composite score
    scores = compute_composite_scores(data)
    scores.sort(key=lambda x: x[0])
    print("\n  Top 5 vehicles (best composite score):")
    for i, (score, v) in enumerate(scores[:5], start=1):
        print(f"    {i}. Weight={format_number(v['Lbs-'])} kg, MPG={format_number(v['Mpg+'])}, "
              f"Acc={format_number(v['Acc+'])}, Composite Score={format_number(score)}")

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

    # CAR-07: Assess desirability for each vehicle
    print_car_07_assessment(data)

    # CAR-08: Updated inventory summary and top‑5 recommendation
    print_car_08_inventory_summary(data)

if __name__ == "__main__":
    main()
