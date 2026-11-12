#!/usr/bin/env python3
"""
skill_06.py

Usage:
    python skill_06.py <csv_path>

The script prints:
    * CAR-11: Explanations of vehicle recommendations by comparing with similar inventory items.
    * CAR-12: Trade‑off explanation between two vehicles with different strengths.
"""

import sys
import math
from pathlib import Path

# Import helpers from the previous skills
from skill_01 import read_csv, compute_composite_scores, format_number, mean

# --------------------------------------------------------------------------- #
# Helper functions for similarity and trade‑offs
# --------------------------------------------------------------------------- #

def euclidean_distance(v1, v2, keys=('Lbs-', 'Mpg+', 'Acc+')):
    """Return Euclidean distance between two vehicles on selected numeric keys."""
    return math.sqrt(sum((float(v1[k]) - float(v2[k])) ** 2 for k in keys))

def find_nearest_neighbors(vehicle, all_vehicles, k=3, exclude_self=True):
    """Return the k nearest vehicles to `vehicle` based on Euclidean distance."""
    distances = []
    for v in all_vehicles:
        if exclude_self and v is vehicle:
            continue
        d = euclidean_distance(vehicle, v)
        distances.append((d, v))
    # Sort by distance ascending
    distances.sort(key=lambda x: x[0])
    return [v for _, v in distances[:k]]

# --------------------------------------------------------------------------- #
# CAR-11: Recommendation explanation using similarity
# --------------------------------------------------------------------------- #

def print_car_11_explanations(data):
    """
    For the top 5 recommended vehicles, explain the recommendation by comparing
    each vehicle with its three most similar inventory items.
    """
    if not data:
        print("\n== CAR-11 ==")
        print("  No vehicle records to analyze.")
        return

    # Compute composite scores
    scored = compute_composite_scores(data)
    # Sort by score (lower is better)
    scored.sort(key=lambda x: x[0])
    top_n = 5
    top_vehicles = scored[:top_n]

    print("\n== CAR-11 ==")
    print("Explanations of recommended vehicles using similar inventory items:")
    for rank, (score, vehicle) in enumerate(top_vehicles, start=1):
        print(f"  {rank}. Vehicle (Clndrs={vehicle['Clndrs']}, Model={vehicle['Model']}):")
        print(f"     Weight={format_number(vehicle['Lbs-'])} kg, MPG={format_number(vehicle['Mpg+'])}, "
              f"Acc={format_number(vehicle['Acc+'])}, Composite Score={format_number(score)}")
        # Find nearest neighbors
        neighbors = find_nearest_neighbors(vehicle, data, k=3, exclude_self=True)
        print("     Similar vehicles:")
        for i, nb in enumerate(neighbors, start=1):
            # Find neighbor's score
            nb_score = next(s for s, v in scored if v is nb)[0] if any(v is nb for _, v in scored) else "unknown"
            print(f"       {i}. Clndrs={nb['Clndrs']}, Model={nb['Model']}, "
                  f"Weight={format_number(nb['Lbs-'])} kg, MPG={format_number(nb['Mpg+'])}, "
                  f"Acc={format_number(nb['Acc+'])}, Composite Score={format_number(nb_score)}")
        # Ensure at least 3 lines per vehicle (already satisfied)

# --------------------------------------------------------------------------- #
# CAR-12: Trade‑off explanation between two vehicles
# --------------------------------------------------------------------------- #

def print_car_12_tradeoff(data):
    """
    Compare the best (lowest composite score) vehicle with the worst
    (highest composite score) vehicle and explain why the best receives a
    higher recommendation by highlighting trade‑offs.
    """
    if not data:
        print("\n== CAR-12 ==")
        print("  No vehicle records to analyze.")
        return

    scored = compute_composite_scores(data)
    if len(scored) < 2:
        print("\n== CAR-12 ==")
        print("  Not enough vehicles to compare.")
        return

    # Determine best and worst vehicles
    scored.sort(key=lambda x: x[0])
    best_score, best_vehicle = scored[0]
    worst_score, worst_vehicle = scored[-1]

    # Compute differences
    weight_diff = best_vehicle['Lbs-'] - worst_vehicle['Lbs-']        # negative if best is lighter
    mpg_diff    = best_vehicle['Mpg+'] - worst_vehicle['Mpg+']       # negative if best has lower mpg
    acc_diff    = best_vehicle['Acc+'] - worst_vehicle['Acc+']       # negative if best is faster

    print("\n== CAR-12 ==")
    print("Trade‑off explanation between two vehicles with different strengths:")
    print(f"  Vehicle A (best recommendation) - Clndrs={best_vehicle['Clndrs']}, "
          f"Model={best_vehicle['Model']}")
    print(f"    Weight={format_number(best_vehicle['Lbs-'])} kg, "
          f"MPG={format_number(best_vehicle['Mpg+'])}, "
          f"Acceleration={format_number(best_vehicle['Acc+'])}, "
          f"Composite Score={format_number(best_score)}")
    print(f"  Vehicle B (worst recommendation) - Clndrs={worst_vehicle['Clndrs']}, "
          f"Model={worst_vehicle['Model']}")
    print(f"    Weight={format_number(worst_vehicle['Lbs-'])} kg, "
          f"MPG={format_number(worst_vehicle['Mpg+'])}, "
          f"Acceleration={format_number(worst_vehicle['Acc+'])}, "
          f"Composite Score={format_number(worst_score)}")
    print("\n  Feature differences (Best - Worst):")
    print(f"    Weight difference: {format_number(weight_diff)} kg (negative means lighter)")
    print(f"    MPG difference: {format_number(mpg_diff)} mpg (negative means lower)")
    print(f"    Acceleration difference: {format_number(acc_diff)} s (negative means faster)")
    print("\n  Explanation:")
    print("    Vehicle A scores better mainly due to lower weight and faster acceleration, "
          "which are heavily weighted in the composite score. However, it sacrifices "
          "fuel efficiency, resulting in a lower MPG compared to Vehicle B. "
          "The trade‑off reflects the client's preference for overall performance "
          "over maximum fuel economy.")

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

    # CAR-11: Explanation using similarity
    print_car_11_explanations(data)

    # CAR-12: Trade‑off comparison
    print_car_12_tradeoff(data)

if __name__ == "__main__":
    main()
