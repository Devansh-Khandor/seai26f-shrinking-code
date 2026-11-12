#!/usr/bin/env python3
"""
Skill 06 – Recommendation Explanations & Trade‑off Analysis

Usage:
    python skill_06.py <path-to-auto93.csv>
"""

import sys
import numpy as np

# Reuse helpers from earlier skills
from skill_01 import load_csv, compute_scores, compute_summary, print_summary

def nan_safe_norm(a: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Compute Euclidean norm while safely handling NaNs.
    NaNs are treated as zeros but a large penalty is added to distance
    if a value is missing in either compared row.
    """
    # Replace NaN with 0 for the norm calculation
    a_filled = np.nan_to_num(a, nan=0.0)
    norm = np.linalg.norm(a_filled, axis=axis)

    # If any NaN present in a row, inflate distance slightly
    mask_nan = np.isnan(a).any(axis=1)
    norm[mask_nan] += 1e5  # large penalty
    return norm

def nearest_neighbors(data: np.ndarray, idx: int, n: int = 3):
    """
    Find indices of the n nearest neighbours of the row at idx.
    Uses all numeric columns; NaNs are penalised.
    """
    diffs = data - data[idx]
    dists = nan_safe_norm(diffs, axis=1)
    dists[idx] = np.inf  # exclude the row itself
    nearest = np.argsort(dists)[:n]
    return nearest

def explain_recommendation(data, header):
    """
    CAR‑11: For each recommended vehicle, explain its recommendation by
    comparing it to its nearest neighbours in the inventory.
    """
    print("\n== CAR-11 ==")
    scores = compute_scores(data, header)
    top_indices = np.argsort(scores)[::-1][:5]  # top 5 vehicles

    # Limit to three vehicles for brevity but still provide detail
    for rank, idx in enumerate(top_indices[:3], start=1):
        vehicle = data[idx]
        score = scores[idx]
        weight = vehicle[header.index("Lbs-")]
        mpg = vehicle[header.index("Mpg+")]
        accel = vehicle[header.index("Acc+")]
        # Find neighbours
        neigh_idx = nearest_neighbors(data, idx, n=3)
        neigh_data = data[neigh_idx]
        neigh_mean = np.nanmean(neigh_data, axis=0)

        print(f"Rank {rank}: Vehicle #{idx + 1}")
        print(f"  Score: {score:.2f}")
        print(f"  Features: Weight={weight:.1f} lbs, MPG={mpg:.1f}, Accel={accel:.1f}")
        print(
            f"  Compared to its 3 nearest neighbours – "
            f"Avg Weight={neigh_mean[header.index('Lbs-')]:.1f} lbs, "
            f"Avg MPG={neigh_mean[header.index('Mpg+')]:.1f}, "
            f"Avg Accel={neigh_mean[header.index('Acc+')]:.1f}"
        )
        print()

def explain_tradeoffs(data, header):
    """
    CAR‑12: Compare two vehicles with different strengths,
    explain why one receives a higher overall recommendation
    and show the trade‑offs involved.
    """
    print("\n== CAR-12 ==")
    scores = compute_scores(data, header)

    # Vehicle with highest score
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    # Vehicle with highest MPG (best fuel‑efficiency)
    mpg_idx = header.index("Mpg+")
    highest_mpg_idx = np.nanargmax(data[:, mpg_idx])
    highest_mpg_score = scores[highest_mpg_idx]

    # Feature indices
    weight_idx = header.index("Lbs-")
    accel_idx = header.index("Acc+")

    # Extract features
    best = data[best_idx]
    mpg_best = data[highest_mpg_idx]

    print(f"Vehicle A (Best Overall): Vehicle #{best_idx + 1}")
    print(f"  Score={best_score:.2f}, Weight={best[weight_idx]:.1f} lbs, "
          f"MPG={best[mpg_idx]:.1f}, Accel={best[accel_idx]:.1f}")
    print(f"Vehicle B (Best Fuel‑Efficiency): Vehicle #{highest_mpg_idx + 1}")
    print(f"  Score={highest_mpg_score:.2f}, Weight={mpg_best[weight_idx]:.1f} lbs, "
          f"MPG={mpg_best[mpg_idx]:.1f}, Accel={mpg_best[accel_idx]:.1f}")

    # Trade‑offs
    weight_diff = best[weight_idx] - mpg_best[weight_idx]
    accel_diff = best[accel_idx] - mpg_best[accel_idx]
    mpg_diff = best[mpg_idx] - mpg_best[mpg_idx]

    print("\nTrade‑off Summary:")
    if weight_diff > 0:
        print(f"  Vehicle A is heavier by {weight_diff:.1f} lbs, "
              f"which contributes to its higher score.")
    else:
        print(f"  Vehicle A is lighter by {abs(weight_diff):.1f} lbs.")
    if accel_diff > 0:
        print(f"  Vehicle A accelerates faster by {accel_diff:.1f} units.")
    else:
        print(f"  Vehicle B accelerates faster by {abs(accel_diff):.1f} units.")
    if mpg_diff < 0:
        print(f"  Vehicle B has {abs(mpg_diff):.1f} mpg better fuel efficiency.")
    else:
        print(f"  Vehicle A has {mpg_diff:.1f} mpg better fuel efficiency.")
    print()

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv-path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        data, header = load_csv(csv_path)
    except Exception as exc:
        print(f"Error loading CSV: {exc}")
        sys.exit(1)

    explain_recommendation(data, header)
    explain_tradeoffs(data, header)

if __name__ == "__main__":
    main()
