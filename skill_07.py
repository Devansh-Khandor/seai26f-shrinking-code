#!/usr/bin/env python3
"""
Skill 07: Balancing performance and inspection strategy for the auto93 dataset.

This script implements:
  CAR-13 – Identify a vehicle with a strong balance of fuel efficiency,
            weight, and acceleration while evaluating as few candidate
            vehicles as reasonably possible.
  CAR-14 – Recommend which vehicles should be evaluated first under
            a limited inspection budget, so that promising inventory
            options are discovered early.

Usage:
    python skill_07.py <path_to_auto93.csv>
"""

import sys
import numpy as np

# Re‑use the loading and analysis utilities from earlier skills
from skill_02 import load_vehicles
from skill_01 import compute_z_scores

# --------------------------------------------------------------------------- #
# CAR-13: Balanced performance finder
# --------------------------------------------------------------------------- #

def find_balanced_vehicle(vehicles):
    """
    Return the vehicle that offers the best balance of fuel efficiency,
    weight, and acceleration, while evaluating only a small subset of
    candidates.

    The approach:
      * Compute z‑scores for weight, mpg, and acceleration over all
        vehicles with known mpg.
      * Determine the 50th percentile (median) for each metric.
      * Filter to vehicles that are in the lower 50% for weight and
        acceleration, and in the upper 50% for mpg.
      * If the filtered set is empty, fall back to all vehicles with known mpg.
      * Select the vehicle with the highest composite score:
          score = -weight_z + mpg_z - acc_z
    """
    # Identify vehicles with known mpg
    known_idx = [i for i, v in enumerate(vehicles) if v.Mpg is not None]
    if not known_idx:
        return None, None  # No data to analyze

    weights = [vehicles[i].Lbs for i in known_idx]
    mpgs = [vehicles[i].Mpg for i in known_idx]
    accs = [vehicles[i].Acc for i in known_idx]

    # Compute z‑scores
    weight_z = compute_z_scores(weights)
    mpg_z = compute_z_scores(mpgs)
    acc_z = compute_z_scores(accs)

    # Percentiles
    w_med = np.percentile(weights, 50)
    mpg_med = np.percentile(mpgs, 50)
    acc_med = np.percentile(accs, 50)

    # Candidate subset
    subset = [
        (j, i)
        for j, i in enumerate(known_idx)
        if (weights[j] <= w_med and mpgs[j] >= mpg_med and accs[j] <= acc_med)
    ]
    if not subset:
        subset = [(j, i) for j, i in enumerate(known_idx)]  # all known

    # Compute composite scores for subset
    best_score = -np.inf
    best_idx = None
    for j, i in subset:
        score = -weight_z[j] + mpg_z[j] - acc_z[j]
        if score > best_score:
            best_score = score
            best_idx = i

    best_vehicle = vehicles[best_idx]
    return best_vehicle, best_score

# --------------------------------------------------------------------------- #
# CAR-14: Inspection priority recommendation
# --------------------------------------------------------------------------- #

def recommend_inspection_order(vehicles, top_n=10):
    """
    Recommend the order in which vehicles should be inspected so that the
    most promising options (highest composite score) are evaluated first.

    Parameters
    ----------
    vehicles : list[Vehicle]
        List of vehicle objects.
    top_n : int
        Number of top vehicles to recommend (default 10).

    Returns
    -------
    list[tuple[int, Vehicle, float]]
        Each tuple contains the original (0‑based) index, the vehicle,
        and its composite score.
    """
    # Only consider vehicles with known mpg
    known_idx = [i for i, v in enumerate(vehicles) if v.Mpg is not None]
    if not known_idx:
        return []

    weights = [vehicles[i].Lbs for i in known_idx]
    mpgs = [vehicles[i].Mpg for i in known_idx]
    accs = [vehicles[i].Acc for i in known_idx]

    # Compute z‑scores once
    weight_z = compute_z_scores(weights)
    mpg_z = compute_z_scores(mpgs)
    acc_z = compute_z_scores(accs)

    # Compute composite scores
    scores = []
    for j, i in enumerate(known_idx):
        score = -weight_z[j] + mpg_z[j] - acc_z[j]
        scores.append((score, i, vehicles[i]))

    # Sort descending
    scores.sort(key=lambda t: t[0], reverse=True)
    return scores[:top_n]

# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #

def main(csv_path):
    vehicles = load_vehicles(csv_path)

    # CAR-13
    print("== CAR-13 ==")
    best_vehicle, best_score = find_balanced_vehicle(vehicles)
    if best_vehicle is None:
        print("No vehicle with known MPG to analyze.")
    else:
        idx = vehicles.index(best_vehicle)
        print(f"Vehicle #{idx + 1} provides the best balanced performance:")
        print(f"  Weight: {best_vehicle.Lbs} lbs")
        print(f"  MPG:    {best_vehicle.Mpg:.2f}")
        print(f"  Acc:    {best_vehicle.Acc:.2f} s")
        print(f"  Composite score: {best_score:.4f}")

    # CAR-14
    print("\n== CAR-14 ==")
    order = recommend_inspection_order(vehicles, top_n=10)
    if not order:
        print("Insufficient data to recommend inspection order.")
    else:
        print("Recommended inspection order (top vehicles by composite score):")
        for rank, (score, idx, v) in enumerate(order, start=1):
            print(f"{rank}. Vehicle #{idx + 1}:")
            print(f"   Score: {score:.4f}")
            print(f"   Weight: {v.Lbs} lbs")
            print(f"   MPG:    {v.Mpg:.2f}")
            print(f"   Acc:    {v.Acc:.2f} s")
            print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_07.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])
