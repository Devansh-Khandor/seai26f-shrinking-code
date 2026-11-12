#!/usr/bin/env python3
"""
skill_07.py

Usage:
    python skill_07.py <csv_path>

The script prints:
    * CAR-13: Identify a vehicle with a strong balance of fuel efficiency,
              weight, and acceleration while evaluating as few candidate
              vehicles as reasonably possible.
    * CAR-14: Recommend a limited number of vehicles to inspect first
              so that promising inventory options are discovered early.
"""

import sys
import math
from pathlib import Path

# Import helper functions from previous skills
from skill_01 import read_csv, compute_composite_scores, format_number, mean

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def percentile_threshold(values, percentile: float):
    """Return the threshold value at the given percentile (0-1)."""
    sorted_vals = sorted(values)
    idx = math.ceil(percentile * len(sorted_vals)) - 1
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return sorted_vals[idx]

def top_percentile_indices(values, percentile: float, reverse: bool = False):
    """Return indices of elements in the top percentile (largest if reverse)."""
    sorted_pairs = sorted(enumerate(values), key=lambda kv: kv[1], reverse=reverse)
    cutoff = math.ceil(percentile * len(values))
    return {idx for idx, val in sorted_pairs[:cutoff]}

# --------------------------------------------------------------------------- #
# CAR-13: Balanced vehicle selection
# --------------------------------------------------------------------------- #

def print_car_13_balanced_vehicle(data):
    """
    Identify a vehicle that balances low weight, high mpg, and fast acceleration.
    The algorithm searches for vehicles that are in the top (or bottom) percentile
    of each metric and then chooses the one with the best composite score.
    """
    if not data:
        print("\n== CAR-13 ==")
        print("  No vehicle records to evaluate.")
        return

    n = len(data)
    # Extract metric lists
    weights = [v['Lbs-'] for v in data]
    mpgs    = [v['Mpg+'] for v in data]
    accs    = [v['Acc+'] for v in data]

    # Start with 5% percentile and relax until we find at least one candidate
    percentile = 0.05
    candidates = set()
    while percentile <= 0.30 and not candidates:
        # Lowest weight (bottom 5%)
        weight_idx = top_percentile_indices(weights, percentile, reverse=False)
        # Highest mpg (top 5%)
        mpg_idx = top_percentile_indices(mpgs, percentile, reverse=True)
        # Fastest acceleration (bottom 5%)
        acc_idx = top_percentile_indices(accs, percentile, reverse=False)

        # Intersection of indices
        candidates = weight_idx & mpg_idx & acc_idx
        if candidates:
            break
        percentile += 0.05

    # If still empty, fallback to all vehicles
    if not candidates:
        candidates = set(range(n))

    # Compute composite scores for candidates
    scored = compute_composite_scores(data)
    # Find best (lowest) composite among candidates
    best_score, best_vehicle = min(
        ((score, v) for score, v in scored if data.index(v) in candidates),
        key=lambda sv: sv[0]
    )

    print("\n== CAR-13 ==")
    print("Vehicle with strong balance of fuel efficiency, weight, and acceleration:")
    print(f"  Clndrs={best_vehicle['Clndrs']}, Model={best_vehicle['Model']}, "
          f"Weight={format_number(best_vehicle['Lbs-'])} kg, "
          f"MPG={format_number(best_vehicle['Mpg+'])}, "
          f"Acc={format_number(best_vehicle['Acc+'])} s, "
          f"Composite Score={format_number(best_score)}")

# --------------------------------------------------------------------------- #
# CAR-14: Prioritized inspection list
# --------------------------------------------------------------------------- #

def print_car_14_inspection_recommendation(data, top_n=5):
    """
    Recommend a small set of vehicles to inspect first. The vehicles
    are sorted by the composite score (lower is better), and the top
    'top_n' vehicles are returned. This list is intended to surface
    promising inventory options early in a limited inspection budget.
    """
    if not data:
        print("\n== CAR-14 ==")
        print("  No vehicle records to inspect.")
        return

    scored = compute_composite_scores(data)
    scored.sort(key=lambda x: x[0])  # best first

    print("\n== CAR-14 ==")
    print(f"Recommended vehicles to evaluate first (top {top_n} by composite score):")
    for rank, (score, v) in enumerate(scored[:top_n], start=1):
        print(f"  {rank}. Clndrs={v['Clndrs']}, Model={v['Model']}, "
              f"Weight={format_number(v['Lbs-'])} kg, "
              f"MPG={format_number(v['Mpg+'])}, "
              f"Acc={format_number(v['Acc+'])} s, "
              f"Composite Score={format_number(score)}")

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

    # CAR-13: Balanced vehicle selection
    print_car_13_balanced_vehicle(data)

    # CAR-14: Inspection priority recommendation
    print_car_14_inspection_recommendation(data, top_n=5)

if __name__ == "__main__":
    main()
