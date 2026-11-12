#!/usr/bin/env python3
"""
Skill 04: Real‑time vehicle assessment and continuous summary updates.

This script implements:
  CAR-07 – Assess each newly arrived vehicle and decide if it belongs among
           the most desirable inventory options.
  CAR-08 – Re‑compute inventory summaries and the recommended shortlist
           as new vehicles are added, without needing to restart the analysis.

Usage:
    python skill_04.py <path_to_auto93.csv>
"""

import sys

# Re‑use the loading and analysis utilities from earlier skills
from skill_02 import (
    load_vehicles,
    compute_stats,
    best_combination,
    format_stats,
    format_top_vehicles,
)

# --------------------------------------------------------------------------- #
# CAR-07: Immediate assessment of every vehicle
# --------------------------------------------------------------------------- #

def assess_new_vehicles(vehicles):
    """
    For every vehicle, compute its composite score and decide
    whether it would fall into the top‑3 best options.

    Returns a list of strings ready for printing.
    """
    # Compute scores for all vehicles (descending order)
    all_scores = best_combination(vehicles, top_n=len(vehicles))

    # Map index to score for quick lookup
    scores_by_index = {idx: score for score, idx, _ in all_scores}

    # Determine threshold (score of the third best vehicle)
    if len(all_scores) >= 3:
        threshold = all_scores[2][0]  # third highest score
    else:
        threshold = all_scores[-1][0]  # if fewer than 3, use the lowest

    lines = ["Analyzing vehicle desirability..."]
    for idx, v in enumerate(vehicles):
        score = scores_by_index[idx]
        in_top3 = score >= threshold
        lines.append(
            f"Vehicle {idx + 1}: score={score:.4f}, "
            f"in top 3: {'Yes' if in_top3 else 'No'}"
        )
    return lines

# --------------------------------------------------------------------------- #
# CAR-08: Continuous inventory summary and recommended list
# --------------------------------------------------------------------------- #

def update_inventory_summary_and_recommendations(vehicles):
    """
    Re‑compute inventory statistics and top‑3 vehicle recommendations.
    Returns a list of strings ready for printing.
    """
    lines = []

    # Summary statistics
    lines.append("Updated Inventory Summary:")
    stats = compute_stats(vehicles)
    lines.extend(format_stats(stats))

    # Recommended vehicles
    lines.append("\nUpdated Recommended Vehicles:")
    top_vehicles = best_combination(vehicles, top_n=3)
    lines.extend(format_top_vehicles(top_vehicles))

    return lines

# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #

def main(csv_path):
    vehicles = load_vehicles(csv_path)

    # CAR-07
    print("== CAR-07 ==")
    for line in assess_new_vehicles(vehicles):
        print(line)

    # CAR-08
    print("\n== CAR-08 ==")
    for line in update_inventory_summary_and_recommendations(vehicles):
        print(line)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_04.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])
