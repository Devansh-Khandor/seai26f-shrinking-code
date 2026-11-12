#!/usr/bin/env python3
"""
Skill 07: Efficient Vehicle Selection (CAR-13 & CAR-14)

Usage:
    python skill_07.py <path_to_auto93.csv>
"""

import sys
import numpy as np


# --------------------------------------------------------------------------- #
# Data loading helper
# --------------------------------------------------------------------------- #

def load_csv(path: str) -> np.ndarray:
    """Load the CSV into a structured NumPy array with numeric columns as float."""
    try:
        data = np.genfromtxt(
            path,
            delimiter=',',
            names=True,
            dtype=None,
            encoding='utf-8'
        )
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Convert numeric columns to float (missing -> NaN)
    numeric_cols = [
        'Clndrs', 'Volume', 'HpX', 'Model', 'origin',
        'Lbs-', 'Acc+', 'Mpg+'
    ]
    for col in numeric_cols:
        if col in data.dtype.names:
            try:
                data[col] = data[col].astype(float)
            except Exception:
                pass
    return data


# --------------------------------------------------------------------------- #
# Composite score calculation
# --------------------------------------------------------------------------- #

def composite_score(data: np.ndarray) -> np.ndarray:
    """
    Lower weight, higher mpg, lower acceleration => higher score.
    NaNs are ignored in the min/max calculation.
    """
    weight = data['Lbs-'].astype(float)
    mpg    = data['Mpg+'].astype(float)
    acc    = data['Acc+'].astype(float)

    def norm_minmax(arr: np.ndarray) -> np.ndarray:
        mn = np.nanmin(arr)
        mx = np.nanmax(arr)
        return (arr - mn) / (mx - mn)

    weight_score = 1.0 - norm_minmax(weight)
    mpg_score    = norm_minmax(mpg)
    acc_score    = 1.0 - norm_minmax(acc)

    return weight_score + mpg_score + acc_score


# --------------------------------------------------------------------------- #
# CAR-13: Strong balanced vehicle
# --------------------------------------------------------------------------- #

def car_13(data: np.ndarray):
    print("== CAR-13 ==")

    # Mask for rows with complete data
    complete_mask = (
        ~np.isnan(data['Lbs-']) &
        ~np.isnan(data['Mpg+']) &
        ~np.isnan(data['Acc+'])
    )

    if not complete_mask.any():
        print("No vehicle has complete data for weight, mpg, and acceleration.")
        return

    scores = composite_score(data)[complete_mask]
    idxs   = np.where(complete_mask)[0]
    top_idx = idxs[np.argmax(scores)]

    vehicle = data[top_idx]
    weight  = vehicle['Lbs-']
    mpg     = vehicle['Mpg+']
    acc     = vehicle['Acc+']
    comp    = scores[np.argmax(scores)]

    print(f"Vehicle index: {top_idx}")
    print(f"Composite score: {comp:.3f}")
    print(f"Weight: {weight:.1f} lbs, MPG: {mpg:.1f}, Acceleration: {acc:.1f} s")
    print()


# --------------------------------------------------------------------------- #
# CAR-14: Prioritize vehicles for limited inspection
# --------------------------------------------------------------------------- #

def car_14(data: np.ndarray, top_n: int = 5):
    print("== CAR-14 ==")

    # Mask for rows with complete data
    complete_mask = (
        ~np.isnan(data['Lbs-']) &
        ~np.isnan(data['Mpg+']) &
        ~np.isnan(data['Acc+'])
    )

    if not complete_mask.any():
        print("No vehicle has complete data for weight, mpg, and acceleration.")
        return

    scores = composite_score(data)[complete_mask]
    idxs   = np.where(complete_mask)[0]

    # Sort descending by composite score
    sorted_order = np.argsort(-scores)
    top_indices = idxs[sorted_order][:top_n]
    top_scores = scores[sorted_order][:top_n]

    print(f"Top {top_n} vehicles to inspect first (by composite score):")
    print("{:>4} {:>8} {:>6} {:>6} {:>6}".format("Rank", "Index", "Score", "Weight", "MPG"))
    for rank, idx, score in zip(range(1, len(top_indices) + 1), top_indices, top_scores):
        weight = data['Lbs-'][idx]
        mpg    = data['Mpg+'][idx]
        print(f"{rank:>4} {idx:>8} {score:>6.3f} {weight:>6.1f} {mpg:>6.1f}")
    print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    data = load_csv(csv_path)
    car_13(data)
    car_14(data, top_n=5)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <path_to_auto93.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    main(csv_path)
