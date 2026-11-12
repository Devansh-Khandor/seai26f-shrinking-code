#!/usr/bin/env python3
"""
Skill 08: Targeted Vehicle Alternatives (CAR-15 & CAR-16)

Usage:
    python skill_08.py <path_to_auto93.csv>
"""

import sys
import numpy as np


# --------------------------------------------------------------------------- #
# Data loading helpers
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
                # Leave as-is (e.g., if already string)
                pass
    return data


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def _norm_minmax(arr: np.ndarray) -> np.ndarray:
    """Return min‑max normalised array, ignoring NaNs."""
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    return (arr - mn) / (mx - mn)


def composite_score(data: np.ndarray) -> np.ndarray:
    """Compute composite score (lower weight, higher mpg, lower acceleration)."""
    weight = data['Lbs-'].astype(float)
    mpg    = data['Mpg+'].astype(float)
    acc    = data['Acc+'].astype(float)

    weight_score = 1.0 - _norm_minmax(weight)
    mpg_score    = _norm_minmax(mpg)
    acc_score    = 1.0 - _norm_minmax(acc)

    return weight_score + mpg_score + acc_score


def nearest_neighbors(
    data: np.ndarray,
    idx: int,
    k: int = 10
) -> np.ndarray:
    """
    Return indices of the k nearest vehicles to vehicle `idx` using
    Euclidean distance on normalised features.
    """
    # Features to consider – use all available numeric columns
    feat_names = ['Clndrs', 'Volume', 'HpX', 'Model', 'origin',
                  'Lbs-', 'Acc+', 'Mpg+']
    # Build feature matrix, normalised per column
    feats = []
    for name in feat_names:
        col = data[name].astype(float)
        feats.append(_norm_minmax(col))
    feats = np.column_stack(feats)

    vec = feats[idx]
    dists = np.linalg.norm(feats - vec, axis=1)
    dists[idx] = np.inf  # exclude self
    nearest = np.argpartition(dists, k)[:k]
    return nearest


# --------------------------------------------------------------------------- #
# CAR-15: Alternatives to poorly rated vehicle
# --------------------------------------------------------------------------- #

def car_15(data: np.ndarray):
    print("== CAR-15 ==")

    comp_scores = composite_score(data)
    # Identify the vehicle with the lowest composite score
    bad_idx = np.nanargmin(comp_scores)
    bad_score = comp_scores[bad_idx]

    # Find nearest neighbors with higher score
    neigh_indices = nearest_neighbors(data, bad_idx, k=30)
    # Filter neighbors that have higher composite score
    better_mask = comp_scores[neigh_indices] > bad_score
    better_indices = neigh_indices[better_mask]

    # If not enough, relax criteria
    if len(better_indices) < 3:
        # Include neighbors regardless of score
        better_indices = neigh_indices[:3]

    # Show top 3 alternatives
    for alt_idx in better_indices[:3]:
        alt_score = comp_scores[alt_idx]
        row = data[alt_idx]
        # Compute differences
        dw = row['Lbs-'] - data[bad_idx]['Lbs-']
        dm = row['Mpg+'] - data[bad_idx]['Mpg+']
        da = row['Acc+'] - data[bad_idx]['Acc+']

        print(f"Alternative Vehicle {alt_idx}: Score={alt_score:.3f}")
        print(f"    Weight: {row['Lbs-']:.1f} lbs (Δ={dw:+.1f})")
        print(f"    MPG:    {row['Mpg+']:.1f} (Δ={dm:+.1f})")
        print(f"    Accel:  {row['Acc+']:.1f} s (Δ={da:+.1f})")
        print()

    if len(better_indices) == 0:
        print("No suitable alternatives found.\n")


# --------------------------------------------------------------------------- #
# CAR-16: Find closest better vehicle with minimal changes
# --------------------------------------------------------------------------- #

def car_16(data: np.ndarray):
    print("== CAR-16 ==")

    # Define preferred performance thresholds
    weight_thr = 4000.0   # lbs
    mpg_thr    = 20.0     # mpg
    acc_thr    = 15.0     # s

    # Compute how many thresholds each vehicle meets
    meets_weight = data['Lbs-'] <= weight_thr
    meets_mpg    = data['Mpg+'] >= mpg_thr
    meets_acc    = data['Acc+'] <= acc_thr

    # Count number of met thresholds
    met_counts = meets_weight.astype(int) + meets_mpg.astype(int) + meets_acc.astype(int)

    # Identify a vehicle that does NOT meet all three thresholds and has the lowest composite score
    comp_scores = composite_score(data)
    # Find vehicles with met_counts < 3
    bad_mask = met_counts < 3
    if not np.any(bad_mask):
        print("All vehicles meet all preferred targets. No need for replacement.\n")
        return

    # Choose the one with lowest composite score among those
    bad_indices = np.where(bad_mask)[0]
    bad_scores = comp_scores[bad_indices]
    bad_idx = bad_indices[np.argmin(bad_scores)]
    bad_met = met_counts[bad_idx]
    bad_score = comp_scores[bad_idx]
    bad_row = data[bad_idx]

    # Find nearest neighbors
    neigh_indices = nearest_neighbors(data, bad_idx, k=30)
    # Evaluate neighbors
    best_candidate = None
    best_met = -1
    best_dist = np.inf

    for n_idx in neigh_indices:
        n_met = met_counts[n_idx]
        if n_met <= bad_met:
            continue  # need better
        # distance
        n_dist = np.linalg.norm(
            _norm_minmax(data['Lbs-'])[bad_idx] - _norm_minmax(data['Lbs-'])[n_idx],
            _norm_minmax(data['Mpg+'])[bad_idx] - _norm_minmax(data['Mpg+'])[n_idx],
            _norm_minmax(data['Acc+'])[bad_idx] - _norm_minmax(data['Acc+'])[n_idx]
        )
        # choose candidate with highest met, then closest distance
        if n_met > best_met or (n_met == best_met and n_dist < best_dist):
            best_candidate = n_idx
            best_met = n_met
            best_dist = n_dist

    if best_candidate is None:
        # fallback: pick nearest neighbor regardless of thresholds
        best_candidate = neigh_indices[0]
        best_met = met_counts[best_candidate]

    alt_idx = best_candidate
    alt_row = data[alt_idx]
    alt_score = comp_scores[alt_idx]

    # Differences
    dw = alt_row['Lbs-'] - bad_row['Lbs-']
    dm = alt_row['Mpg+'] - bad_row['Mpg+']
    da = alt_row['Acc+'] - bad_row['Acc+']

    # Show result
    print(f"Selected vehicle for replacement: Index={alt_idx}")
    print(f"    Composite Score: {alt_score:.3f}")
    print(f"    Weight: {alt_row['Lbs-']:.1f} lbs (Δ={dw:+.1f})")
    print(f"    MPG:    {alt_row['Mpg+']:.1f} (Δ={dm:+.1f})")
    print(f"    Accel:  {alt_row['Acc+']:.1f} s (Δ={da:+.1f})")
    print(f"    Meets {best_met}/3 preferred targets vs {bad_met}/3 of original.\n")

    # Explain minimal changes
    changes = []
    if abs(dw) < 100:  # weight change <100 lbs considered minimal
        changes.append("weight")
    if abs(dm) < 2:  # mpg change <2 mpg considered minimal
        changes.append("mpg")
    if abs(da) < 1:  # acceleration change <1 s considered minimal
        changes.append("acceleration")
    if changes:
        changes_str = ", ".join(changes)
        print(f"Minimal change in {changes_str} needed to meet more targets.")
    else:
        print("Multiple characteristics changed to achieve better targets.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    data = load_csv(csv_path)
    car_15(data)
    car_16(data)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <path_to_auto93.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    main(csv_path)
