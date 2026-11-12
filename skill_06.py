#!/usr/bin/env python3
"""
Skill 06: Recommendation Justification (CAR-11 & CAR-12)

Usage:
    python skill_06.py <path_to_auto93.csv>
"""

import sys
import numpy as np

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def load_csv(path: str) -> np.ndarray:
    """
    Load the CSV into a structured NumPy array with all numeric columns
    as float (NaNs for missing values).
    """
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

    # Convert numeric columns to float
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


def _norm_minmax(arr: np.ndarray) -> np.ndarray:
    """Return min‑max normalised array handling NaNs."""
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    return (arr - mn) / (mx - mn)


def composite_score(data: np.ndarray) -> np.ndarray:
    """Compute the composite score (lower weight, higher mpg, lower acceleration)."""
    weight = data['Lbs-'].astype(float)
    mpg    = data['Mpg+'].astype(float)
    acc    = data['Acc+'].astype(float)

    weight_score = 1.0 - _norm_minmax(weight)
    mpg_score    = _norm_minmax(mpg)
    acc_score    = 1.0 - _norm_minmax(acc)

    return weight_score + mpg_score + acc_score


def component_scores(data: np.ndarray):
    """Return individual component scores."""
    weight = data['Lbs-'].astype(float)
    mpg    = data['Mpg+'].astype(float)
    acc    = data['Acc+'].astype(float)

    weight_score = 1.0 - _norm_minmax(weight)
    mpg_score    = _norm_minmax(mpg)
    acc_score    = 1.0 - _norm_minmax(acc)

    return weight_score, mpg_score, acc_score


def nearest_neighbors(data: np.ndarray, idx: int, k: int = 3):
    """
    Find the indices of the k nearest vehicles to vehicle `idx` using
    Euclidean distance on normalised features.
    """
    # Features to consider
    feats = np.column_stack((
        _norm_minmax(data['Lbs-']),
        _norm_minmax(data['Mpg+']),
        _norm_minmax(data['Acc+']),
        _norm_minmax(data['HpX']),
        _norm_minmax(data['Volume'])
    ))

    vec = feats[idx]
    dists = np.linalg.norm(feats - vec, axis=1)
    dists[idx] = np.inf  # exclude self
    nearest = np.argpartition(dists, k)[:k]
    return nearest


# --------------------------------------------------------------------------- #
# CAR-11: Recommendation Justification
# --------------------------------------------------------------------------- #

def car_11(data: np.ndarray):
    """Explain recommendation of the top vehicle using comparisons to similar vehicles."""
    print("== CAR-11 ==")

    comp_scores = composite_score(data)
    top_idx = np.argmax(comp_scores)

    weight = data['Lbs-'][top_idx]
    mpg    = data['Mpg+'][top_idx]
    acc    = data['Acc+'][top_idx]

    neighbors = nearest_neighbors(data, top_idx, k=3)
    neigh_weight = data['Lbs-'][neighbors]
    neigh_mpg    = data['Mpg+'][neighbors]
    neigh_acc    = data['Acc+'][neighbors]

    mean_neigh_w = np.nanmean(neigh_weight)
    mean_neigh_m = np.nanmean(neigh_mpg)
    mean_neigh_a = np.nanmean(neigh_acc)

    delta_w = weight - mean_neigh_w
    delta_m = mpg    - mean_neigh_m
    delta_a = acc    - mean_neigh_a

    # Three explanatory lines
    print(f"Vehicle {top_idx} has weight {weight:.1f} lbs, "
          f"{abs(delta_w):.1f} lbs {'lighter' if delta_w < 0 else 'heavier'} "
          f"than the average of its three nearest neighbors ({mean_neigh_w:.1f} lbs).")
    print(f"Vehicle {top_idx} achieves {mpg:.1f} mpg, "
          f"{abs(delta_m):.1f} mpg {'higher' if delta_m > 0 else 'lower'} "
          f"than its nearest neighbors ({mean_neigh_m:.1f} mpg).")
    print(f"Vehicle {top_idx} accelerates in {acc:.1f} s, "
          f"{abs(delta_a):.1f} s {'faster' if delta_a < 0 else 'slower'} "
          f"than its nearest neighbors ({mean_neigh_a:.1f} s).")
    print()


# --------------------------------------------------------------------------- #
# CAR-12: Trade‑off Explanation between two vehicles
# --------------------------------------------------------------------------- #

def car_12(data: np.ndarray):
    """Explain why the top vehicle gets a higher overall recommendation than the second."""
    print("== CAR-12 ==")

    comp_scores = composite_score(data)
    sorted_idx = np.argsort(-comp_scores)

    top_idx = sorted_idx[0]
    second_idx = sorted_idx[1]

    weight_score, mpg_score, acc_score = component_scores(data)

    top_w, top_m, top_a = weight_score[top_idx], mpg_score[top_idx], acc_score[top_idx]
    sec_w, sec_m, sec_a = weight_score[second_idx], mpg_score[second_idx], acc_score[second_idx]

    top_comp = top_w + top_m + top_a
    sec_comp = sec_w + sec_m + sec_a
    diff_comp = top_comp - sec_comp

    # Line 1: composite scores
    print(f"Vehicle {top_idx} has composite score {top_comp:.3f} vs Vehicle {second_idx} "
          f"{sec_comp:.3f} (difference {diff_comp:.3f}).")

    # Line 2: component strengths
    print(f"Vehicle {top_idx} excels in weight ({top_w:.2f} vs {sec_w:.2f}) "
          f"and mpg ({top_m:.2f} vs {sec_m:.2f}), while "
          f"Vehicle {second_idx} has better acceleration ({sec_a:.2f} vs {top_a:.2f}).")

    # Line 3: trade‑off summary
    print(f"Overall, Vehicle {top_idx} is lighter and more fuel efficient, "
          f"but slightly slower in acceleration compared to Vehicle {second_idx}.")
    print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    data = load_csv(csv_path)
    car_11(data)
    car_12(data)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <path_to_auto93.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    main(csv_path)
