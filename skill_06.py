#!/usr/bin/env python3
"""
Extended data‑analysis script for the Auto93 dataset.

Adds two new client requests:
    CAR-11 – Explain a vehicle recommendation with understandable comparisons
             to similar vehicles in inventory.
    CAR-12 – Compare two vehicles with different strengths, explain
             why one receives a higher overall recommendation and show the trade‑offs.

Usage:
    python skill_06.py <path_to_auto93.csv>
"""

import sys
from typing import List, Dict, Any

# Import helpers from earlier skill modules
from skill_01 import (
    read_csv,
    compute_score,
    format_float,
)

# --------------------------------------------------------------------------- #
# Helper to build bounds for normalization
# --------------------------------------------------------------------------- #
def _compute_bounds(rows: List[Dict[str, Any]]) -> tuple:
    weight_bounds = {
        'min': min(r['Lbs-'] for r in rows if r['Lbs-'] is not None),
        'max': max(r['Lbs-'] for r in rows if r['Lbs-'] is not None),
    }
    mpg_bounds = {
        'min': min(r['Mpg+'] for r in rows if r['Mpg+'] is not None),
        'max': max(r['Mpg+'] for r in rows if r['Mpg+'] is not None),
    }
    accel_bounds = {
        'min': min(r['Acc+'] for r in rows if r['Acc+'] is not None),
        'max': max(r['Acc+'] for r in rows if r['Acc+'] is not None),
    }
    return weight_bounds, mpg_bounds, accel_bounds

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <path_to_auto93.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        rows = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Filter rows that have all required metrics for scoring
    valid_rows = [r for r in rows if r['Lbs-'] is not None and
                                    r['Mpg+'] is not None and
                                    r['Acc+'] is not None]

    if not valid_rows:
        print("No vehicles with complete data to compute recommendations.")
        return

    # Compute bounds and scores
    weight_bounds, mpg_bounds, accel_bounds = _compute_bounds(valid_rows)
    for r in valid_rows:
        r['score'] = compute_score(r, weight_bounds, mpg_bounds, accel_bounds)

    # Rank by composite score (descending)
    ranked = sorted(valid_rows, key=lambda r: r['score'], reverse=True)

    # ----------------------------------------------------------------------- #
    # CAR-11 – Recommendation explanation
    # ----------------------------------------------------------------------- #
    print("== CAR-11 ==")

    # Show explanation for the top 3 vehicles
    for i in range(min(3, len(ranked))):
        veh = ranked[i]
        # Pick a comparable competitor (next in rank)
        competitor = ranked[i + 1] if i + 1 < len(ranked) else ranked[-1]

        weight_diff = competitor['Lbs-'] - veh['Lbs-']  # positive if competitor heavier
        mpg_diff = veh['Mpg+'] - competitor['Mpg+']   # positive if veh higher mpg
        accel_diff = competitor['Acc+'] - veh['Acc+']  # positive if competitor slower

        print(
            f"{i + 1}. Vehicle Clndrs={int(veh['Clndrs'])}, "
            f"Model={int(veh['Model'])} – Recommended "
            f"because it is lighter by {format_float(weight_diff)} lbs, "
            f"has {format_float(mpg_diff)} more mpg, "
            f"and accelerates {format_float(accel_diff)} s faster "
            f"than the next best vehicle."
        )

    # ----------------------------------------------------------------------- #
    # CAR-12 – Trade‑off comparison between two vehicles
    # ----------------------------------------------------------------------- #
    print("\n== CAR-12 ==")

    # Identify vehicle with highest mpg
    best_mpg_veh = max(valid_rows, key=lambda r: r['Mpg+'])
    # Identify vehicle with highest score
    best_score_veh = ranked[0]
    # If they are the same, use second best score vehicle
    if best_mpg_veh is best_score_veh:
        best_score_veh = ranked[1]

    # Compute differences for key metrics
    weight_diff = best_score_veh['Lbs-'] - best_mpg_veh['Lbs-']
    mpg_diff = best_mpg_veh['Mpg+'] - best_score_veh['Mpg+']
    accel_diff = best_score_veh['Acc+'] - best_mpg_veh['Acc+']

    # Explanation lines
    print(
        f"1. Highest mpg vehicle: Clndrs={int(best_mpg_veh['Clndrs'])}, "
        f"Model={int(best_mpg_veh['Model'])}, mpg={int(best_mpg_veh['Mpg+'])} "
        f"but it is heavier by {format_float(-weight_diff)} lbs and "
        f"accelerates slower by {format_float(-accel_diff)} s, "
        f"yielding a lower composite score."
    )
    print(
        f"2. Highest composite score vehicle: Clndrs={int(best_score_veh['Clndrs'])}, "
        f"Model={int(best_score_veh['Model'])}, score={format_float(best_score_veh['score'])} "
        f"achieved by being lighter by {format_float(weight_diff)} lbs, "
        f"accelerating faster by {format_float(accel_diff)} s, "
        f"though its mpg is lower by {format_float(mpg_diff)}."
    )
    print(
        f"3. Trade‑off summary: The higher‑mpg vehicle excels in fuel efficiency but "
        f"is heavier and slower, while the top‑scoring vehicle trades a few mpg for "
        f"significant weight reduction and acceleration advantage, leading to a superior overall "
        f"recommendation."
    )


if __name__ == "__main__":
    main()
