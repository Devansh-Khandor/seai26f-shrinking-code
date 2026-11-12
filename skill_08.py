#!/usr/bin/env python3
"""
Skill 08: Advanced alternative vehicle selection.

This script implements:
  CAR-15 – For a poorly rated vehicle, show alternative vehicles in the historical
           inventory with similar characteristics but better recorded performance,
           and explain which differences could account for the improvement.
  CAR-16 – For a vehicle that does not meet preferred performance targets,
           identify the closest available alternative that meets more of those
           targets while changing as few vehicle characteristics as possible.

Usage:
    python skill_08.py <path_to_auto93.csv>
"""

import sys
import math
import numpy as np

# Re‑use utilities from earlier skills
from skill_02 import load_vehicles, Vehicle
from skill_01 import compute_z_scores

# --------------------------------------------------------------------------- #
# Helper: compute composite score for a vehicle given z‑scores lists
# --------------------------------------------------------------------------- #
def composite_score(v, weight_z, mpg_z, acc_z, idx, known_idx):
    """
    Compute the composite score used in best_combination.
    Parameters
    ----------
    v : Vehicle
    weight_z, mpg_z, acc_z : list[float]
        Z‑scores for weight, mpg, and acceleration over the known‑mpg set.
    idx : int
        Index of v in the original vehicle list.
    known_idx : list[int]
        List of indices for vehicles with known mpg.

    Returns
    -------
    float
    """
    # Find the position j of idx in known_idx
    j = known_idx.index(idx)
    return -weight_z[j] + mpg_z[j] - acc_z[j]

# --------------------------------------------------------------------------- #
# Helper: Euclidean distance between two vehicles on selected numeric features
# --------------------------------------------------------------------------- #
def vehicle_distance(v1, v2):
    """Euclidean distance between two vehicles on (Lbs, Mpg, Acc)."""
    return math.sqrt(
        (v1.Lbs - v2.Lbs) ** 2 + (v1.Mpg - v2.Mpg) ** 2 + (v1.Acc - v2.Acc) ** 2
    )

# --------------------------------------------------------------------------- #
# CAR-15 implementation
# --------------------------------------------------------------------------- #
def car_15(vehicles):
    """
    Find the poorly rated (lowest composite score) vehicle and list the
    three best alternatives with similar characteristics and higher
    recorded performance. Return list of output lines.
    """
    lines = []

    # Find vehicles with known mpg
    known_idx = [i for i, v in enumerate(vehicles) if v.Mpg is not None]
    if not known_idx:
        return ["No vehicle with known MPG available for comparison."]

    # Compute z-scores for weight, mpg, acc
    weights = [vehicles[i].Lbs for i in known_idx]
    mpgs    = [vehicles[i].Mpg for i in known_idx]
    accs    = [vehicles[i].Acc for i in known_idx]

    weight_z = compute_z_scores(weights)
    mpg_z    = compute_z_scores(mpgs)
    acc_z    = compute_z_scores(accs)

    # Compute composite scores
    scores = {i: composite_score(vehicles[i], weight_z, mpg_z, acc_z, i, known_idx) for i in known_idx}

    # Identify worst vehicle
    worst_idx = min(scores, key=scores.get)
    worst_vehicle = vehicles[worst_idx]
    worst_score  = scores[worst_idx]

    lines.append(f"Worst-rated vehicle is Vehicle #{worst_idx + 1} (score={worst_score:.4f})")
    lines.append(f"  Weight: {worst_vehicle.Lbs} lbs, MPG: {worst_vehicle.Mpg:.2f}, Acc: {worst_vehicle.Acc:.2f}s")

    # Find alternatives: higher score & similar characteristics
    candidates = []
    for idx, v in enumerate(vehicles):
        if v.Mpg is None or idx == worst_idx:
            continue
        score = scores[idx]
        if score <= worst_score:
            continue
        dist = vehicle_distance(worst_vehicle, v)
        candidates.append((dist, idx, v, score))

    # Sort by increasing distance, then by higher score
    candidates.sort(key=lambda t: (t[0], -t[3]))

    if not candidates:
        lines.append("No better alternatives found.")
        return lines

    lines.append("Top 3 better alternatives with similar characteristics:")
    for i, (dist, idx, v, score) in enumerate(candidates[:3], 1):
        weight_diff = v.Lbs - worst_vehicle.Lbs
        mpg_diff    = v.Mpg - worst_vehicle.Mpg
        acc_diff    = v.Acc - worst_vehicle.Acc
        lines.append(f"{i}. Vehicle #{idx + 1} (score={score:.4f}, dist={dist:.2f})")
        lines.append(f"   Weight diff: {weight_diff:+d} lbs")
        lines.append(f"   MPG diff:    {mpg_diff:+.2f}")
        lines.append(f"   Acc diff:    {acc_diff:+.2f}s")
        lines.append(f"   Reason: This vehicle is lighter by {abs(weight_diff)} lbs,")
        lines.append(f"           gains {mpg_diff:.2f} MPG, and accelerates {abs(acc_diff):.2f}s faster.")
        lines.append("")

    return lines

# --------------------------------------------------------------------------- #
# CAR-16 implementation
# --------------------------------------------------------------------------- #
def car_16(vehicles):
    """
    For a vehicle that does not meet preferred performance targets,
    identify the closest available alternative that meets more of those
    targets while changing as few characteristics as possible.
    Return a list of output lines.
    """
    lines = []

    # Find vehicles with known mpg
    known_idx = [i for i, v in enumerate(vehicles) if v.Mpg is not None]
    if not known_idx:
        return ["No vehicle with known MPG available for comparison."]

    # Compute z-scores for weight, mpg, acc
    weights = [vehicles[i].Lbs for i in known_idx]
    mpgs    = [vehicles[i].Mpg for i in known_idx]
    accs    = [vehicles[i].Acc for i in known_idx]

    weight_z = compute_z_scores(weights)
    mpg_z    = compute_z_scores(mpgs)
    acc_z    = compute_z_scores(accs)

    # Composite scores
    scores = {i: composite_score(vehicles[i], weight_z, mpg_z, acc_z, i, known_idx) for i in known_idx}

    # Identify worst vehicle (fails targets)
    worst_idx = min(scores, key=scores.get)
    worst_vehicle = vehicles[worst_idx]
    worst_score  = scores[worst_idx]

    # Determine preferred targets from best vehicle
    best_idx = max(scores, key=scores.get)
    best_vehicle = vehicles[best_idx]
    targets = {
        "weight": best_vehicle.Lbs,
        "mpg": best_vehicle.Mpg,
        "acc": best_vehicle.Acc,
    }

    # Function to count satisfied targets
    def count_targets(v):
        return int(v.Lbs <= targets["weight"]) + int(v.Mpg >= targets["mpg"]) + int(v.Acc <= targets["acc"])

    # Find candidates with higher score than worst and at least one target satisfied
    candidates = []
    for idx, v in enumerate(vehicles):
        if v.Mpg is None or idx == worst_idx:
            continue
        if scores[idx] <= worst_score:
            continue
        targets_met = count_targets(v)
        if targets_met == 0:
            continue
        dist = vehicle_distance(worst_vehicle, v)
        candidates.append((targets_met, -dist, idx, v, scores[idx]))

    if not candidates:
        lines.append(f"Vehicle #{worst_idx + 1} does not meet preferred targets and no better alternative was found.")
        return lines

    # Choose candidate with most targets met; tie‑break by closest distance
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best_met, neg_dist, alt_idx, alt_vehicle, alt_score = candidates[0]

    lines.append(f"Vehicle #{worst_idx + 1} does not meet preferred targets.")
    lines.append(f"Best alternative that meets {best_met} target(s) is Vehicle #{alt_idx + 1} (score={alt_score:.4f}).")
    lines.append("Differences compared to the worst vehicle:")
    lines.append(f"  Weight: {alt_vehicle.Lbs} lbs vs {worst_vehicle.Lbs} lbs")
    lines.append(f"  MPG:    {alt_vehicle.Mpg:.2f} vs {worst_vehicle.Mpg:.2f}")
    lines.append(f"  Acc:    {alt_vehicle.Acc:.2f}s vs {worst_vehicle.Acc:.2f}s")
    lines.append("Explanation:")
    lines.append("  The alternative achieves a lower weight and faster acceleration,")
    lines.append("  while maintaining a higher MPG, thus satisfying more of the preferred targets.")
    return lines

# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #
def main(csv_path):
    vehicles = load_vehicles(csv_path)

    print("== CAR-15 ==")
    for line in car_15(vehicles):
        print(line)

    print("\n== CAR-16 ==")
    for line in car_16(vehicles):
        print(line)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_08.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])
