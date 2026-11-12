#!/usr/bin/env python3
"""
Skill 08 – Alternative Vehicle Recommendations

Usage:
    python skill_08.py <path-to-auto93.csv>
"""

import sys
import numpy as np

# Import helper functions from earlier skills
from skill_01 import load_csv, compute_scores, compute_summary, print_summary

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def nan_safe_norm(a: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Compute Euclidean norm while safely handling NaNs.
    NaNs are treated as zeros but a large penalty is added to distance
    if a value is missing in either compared row.
    """
    a_filled = np.nan_to_num(a, nan=0.0)
    norm = np.linalg.norm(a_filled, axis=axis)
    mask_nan = np.isnan(a).any(axis=1)
    norm[mask_nan] += 1e5  # large penalty
    return norm

def nearest_neighbors(data: np.ndarray, idx: int, n: int = 3) -> np.ndarray:
    """
    Return indices of the `n` nearest neighbors of the row at `idx`.
    """
    diffs = data - data[idx]
    dists = nan_safe_norm(diffs, axis=1)
    dists[idx] = np.inf
    return np.argsort(dists)[:n]

# ----------------------------------------------------------------------
# CAR-15: Alternatives for a poorly rated vehicle
# ----------------------------------------------------------------------
def car_15(data: np.ndarray, header: list[str]) -> None:
    print("\n== CAR-15 ==")

    scores = compute_scores(data, header)
    poor_idx = np.argmin(scores)
    poor_score = scores[poor_idx]

    # Find nearest neighbors (ignore poor vehicle itself)
    neigh_idx = nearest_neighbors(data, poor_idx, n=10)  # take more to filter
    # Filter neighbors that have a higher score
    good_neigh = [i for i in neigh_idx if scores[i] > poor_score]
    if not good_neigh:
        print("  No better vehicles found in the inventory.")
        return

    # Pick top 3 closest better neighbors
    selected = good_neigh[:3]

    poor_vehicle = data[poor_idx]
    poor_features = dict(zip(header, poor_vehicle))
    print(f"Poorly rated vehicle #{poor_idx + 1} (Score={poor_score:.2f}):")
    for k in ("Lbs-", "Mpg+", "Acc+"):
        print(f"  {k} = {poor_features[k]:.2f}")

    for rank, idx in enumerate(selected, 1):
        neigh = data[idx]
        neigh_features = dict(zip(header, neigh))
        print(f"\n  Alternative {rank}: Vehicle #{idx + 1} (Score={scores[idx]:.2f})")
        print(f"    Lbs-  : {neigh_features['Lbs-']:.2f}")
        print(f"    Mpg+  : {neigh_features['Mpg+']:.2f}")
        print(f"    Acc+  : {neigh_features['Acc+']:.2f}")

        # Explain differences
        weight_diff = neigh_features['Lbs-'] - poor_features['Lbs-']
        mpg_diff = neigh_features['Mpg+'] - poor_features['Mpg+']
        accel_diff = neigh_features['Acc+'] - poor_features['Acc+']
        print(f"    Differences → Weight: {weight_diff:+.2f} lbs, "
              f"MPG: {mpg_diff:+.2f}, Accel: {accel_diff:+.2f}")

# ----------------------------------------------------------------------
# CAR-16: Closest vehicle meeting more performance targets
# ----------------------------------------------------------------------
def car_16(data: np.ndarray, header: list[str]) -> None:
    print("\n== CAR-16 ==")

    # Define simple performance targets
    mpg_target = 12.0          # mpg
    weight_target = 4400.0     # lbs
    accel_target = 14.0        # accel units

    # Find a vehicle that does NOT meet all three targets
    # We'll pick the one with the lowest score for illustration
    scores = compute_scores(data, header)
    poor_idx = np.argmin(scores)
    poor_vehicle = data[poor_idx]

    def meets(v):
        return (
            (v[header.index("Mpg+")] >= mpg_target) and
            (v[header.index("Lbs-")] <= weight_target) and
            (v[header.index("Acc+")] <= accel_target)
        )

    # Count how many targets each vehicle meets
    target_counts = np.array([(
        int(data[i, header.index("Mpg+")] >= mpg_target) +
        int(data[i, header.index("Lbs-")] <= weight_target) +
        int(data[i, header.index("Acc+")] <= accel_target)
    ) for i in range(len(data))])

    poor_count = target_counts[poor_idx]

    # Candidates that meet more targets than the poor vehicle
    candidates = np.where(target_counts > poor_count)[0]
    if candidates.size == 0:
        print("  No alternative vehicle meets more targets than the poor vehicle.")
        return

    # Choose candidate closest to the poor vehicle (Euclidean distance)
    distances = nan_safe_norm(data[candidates] - poor_vehicle, axis=1)
    best_candidate_idx = candidates[np.argmin(distances)]

    alt_vehicle = data[best_candidate_idx]
    alt_score = scores[best_candidate_idx]

    print(f"Poor vehicle #{poor_idx + 1} (Score={scores[poor_idx]:.2f}) does not meet all targets.")
    print(f"  MPG = {data[poor_idx, header.index('Mpg+')]:.2f}, "
          f"Weight = {data[poor_idx, header.index('Lbs-')]:.2f}, "
          f"Accel = {data[poor_idx, header.index('Acc+')]:.2f}")

    print(f"\nClosest alternative meeting more targets: Vehicle #{best_candidate_idx + 1} (Score={alt_score:.2f})")
    print(f"  MPG = {alt_vehicle[header.index('Mpg+')]:.2f}, "
          f"Weight = {alt_vehicle[header.index('Lbs-')]:.2f}, "
          f"Accel = {alt_vehicle[header.index('Acc+')]:.2f}")

    # Explain which targets improved and the magnitude of change
    improvement = []
    for name, idx, comp, val in (
        ("MPG", header.index("Mpg+"), lambda x: x >= mpg_target, data[poor_idx, header.index("Mpg+")] ),
        ("Weight", header.index("Lbs-"), lambda x: x <= weight_target, data[poor_idx, header.index("Lbs-")] ),
        ("Accel", header.index("Acc+"), lambda x: x <= accel_target, data[poor_idx, header.index("Acc+")] ),
    ):
        if comp(alt_vehicle[idx]) and not comp(val):
            improvement.append(name)

    print("\nImproved targets:", ", ".join(improvement) if improvement else "None")
    print(f"Change in characteristics compared to poor vehicle:")
    weight_diff = alt_vehicle[header.index("Lbs-")] - data[poor_idx, header.index("Lbs-")]
    mpg_diff = alt_vehicle[header.index("Mpg+")] - data[poor_idx, header.index("Mpg+")]
    accel_diff = alt_vehicle[header.index("Acc+")] - data[poor_idx, header.index("Acc+")]
    print(f"  Weight change: {weight_diff:+.2f} lbs")
    print(f"  MPG change: {mpg_diff:+.2f}")
    print(f"  Accel change: {accel_diff:+.2f}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv-path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        data, header = load_csv(csv_path)
    except Exception as exc:
        print(f"Error loading CSV: {exc}")
        sys.exit(1)

    car_15(data, header)
    car_16(data, header)

if __name__ == "__main__":
    main()
