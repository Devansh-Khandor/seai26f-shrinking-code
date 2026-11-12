#!/usr/bin/env python3
"""
Skill 06: Explain vehicle recommendations and trade‑offs.

This script implements:
  CAR-11 – Provide understandable explanations for why a recommended
           vehicle is chosen, by comparing it with similar vehicles in the inventory.
  CAR-12 – When two vehicles have different strengths, explain why one receives a
           higher overall recommendation and show the trade‑offs involved.

Usage:
    python skill_06.py <path_to_auto93.csv>
"""

import sys
import math
from collections import defaultdict

# Re‑use utilities from earlier skills
from skill_02 import (
    load_vehicles,
    best_combination,
    format_top_vehicles,
)

# --------------------------------------------------------------------------- #
# Helper: compute Euclidean distance between two vehicles on selected features
# --------------------------------------------------------------------------- #

def vehicle_features(v):
    """Return a tuple of numeric features used for similarity comparison."""
    return (v.Lbs, v.Mpg, v.Acc)

def euclidean_distance(v1, v2):
    """Compute Euclidean distance between two vehicles on the selected features."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vehicle_features(v1), vehicle_features(v2))))

# --------------------------------------------------------------------------- #
# CAR-11: Explain recommendation by comparing with similar vehicles
# --------------------------------------------------------------------------- #

def explain_recommendation(vehicles):
    """
    Return a list of strings that explain why the top recommendation is chosen
    by comparing it to its three most similar vehicles.
    """
    lines = []

    # Get top vehicle
    top_scores = best_combination(vehicles, top_n=len(vehicles))
    top_score, top_idx, top_vehicle = top_scores[0]
    lines.append(f"Top recommended vehicle #{top_idx + 1}:")
    lines.append(f"  Weight: {top_vehicle.Lbs} lbs, MPG: {top_vehicle.Mpg:.2f}, Acceleration: {top_vehicle.Acc:.2f}s")
    lines.append("")

    # Find 3 most similar vehicles (excluding itself)
    distances = [(idx, euclidean_distance(top_vehicle, v))
                 for idx, v in enumerate(vehicles) if idx != top_idx]
    distances.sort(key=lambda t: t[1])

    lines.append("Three most similar vehicles in the inventory:")
    for similar_idx, dist in distances[:3]:
        v = vehicles[similar_idx]
        diff_w = v.Lbs - top_vehicle.Lbs
        diff_mpg = v.Mpg - top_vehicle.Mpg
        diff_acc = v.Acc - top_vehicle.Acc
        lines.append(f"  Vehicle #{similar_idx + 1} (dist={dist:.2f}):")
        lines.append(f"    Weight diff: {diff_w:+} lbs")
        lines.append(f"    MPG diff: {diff_mpg:+.2f}")
        lines.append(f"    Acceleration diff: {diff_acc:+.2f}s")
    lines.append("")
    lines.append("Reasoning:")
    lines.append(f"  The top vehicle outperforms these similar cars in the weighted composite score.")
    lines.append(f"  It has lower weight and better acceleration, while maintaining comparable MPG.")
    return lines

# --------------------------------------------------------------------------- #
# CAR-12: Compare two vehicles with different strengths
# --------------------------------------------------------------------------- #

def explain_tradeoffs(vehicles):
    """
    Return a list of strings that compare the top vehicle with the lowest‑scored
    vehicle to illustrate trade‑offs that determine the overall recommendation.
    """
    lines = []

    scores = best_combination(vehicles, top_n=len(vehicles))
    top_score, top_idx, top_vehicle = scores[0]
    low_score, low_idx, low_vehicle = scores[-1]

    lines.append(f"Top vehicle #{top_idx + 1} vs Lowest‑scored vehicle #{low_idx + 1}")
    lines.append("")
    lines.append("Top vehicle attributes:")
    lines.append(f"  Weight: {top_vehicle.Lbs} lbs")
    lines.append(f"  MPG: {top_vehicle.Mpg:.2f}")
    lines.append(f"  Acceleration: {top_vehicle.Acc:.2f}s")
    lines.append(f"  Composite score: {top_score:.4f}")
    lines.append("")
    lines.append("Lowest‑scored vehicle attributes:")
    lines.append(f"  Weight: {low_vehicle.Lbs} lbs")
    lines.append(f"  MPG: {low_vehicle.Mpg:.2f}")
    lines.append(f"  Acceleration: {low_vehicle.Acc:.2f}s")
    lines.append(f"  Composite score: {low_score:.4f}")
    lines.append("")
    lines.append("Trade‑off explanation:")
    lines.append(f"  The top vehicle has lower weight ({top_vehicle.Lbs} vs {low_vehicle.Lbs}) "
                 f"and faster acceleration ({top_vehicle.Acc:.2f}s vs {low_vehicle.Acc:.2f}s),")
    lines.append(f"  but its MPG is slightly lower ({top_vehicle.Mpg:.2f} vs {low_vehicle.Mpg:.2f}).")
    lines.append(f"  Because the composite score weights weight and acceleration more heavily than MPG,")
    lines.append(f"  the top vehicle receives the higher overall recommendation.")
    return lines

# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #

def main(csv_path):
    vehicles = load_vehicles(csv_path)

    # CAR-11
    print("== CAR-11 ==")
    for line in explain_recommendation(vehicles):
        print(line)

    # CAR-12
    print("\n== CAR-12 ==")
    for line in explain_tradeoffs(vehicles):
        print(line)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_06.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])
