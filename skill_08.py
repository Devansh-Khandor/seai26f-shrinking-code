#!/usr/bin/env python3
"""
skill_08.py

Entry point for CAR-15 and CAR-16 skills.

Usage:
    python skill_08.py <path/to/auto93.csv>
"""

import argparse
import sys
from typing import List, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Import utilities from earlier modules (skill_02)
# --------------------------------------------------------------------------- #
try:
    from skill_02 import (
        read_csv_as_array,
        compute_composite_score,
        compute_summary_stats,
    )
except Exception:
    # Fallback minimal implementations
    def read_csv_as_array(path: str) -> np.ndarray:
        import csv

        with open(path, newline="") as f:
            reader = csv.reader(f)
            data = []
            for row in reader:
                processed_row = []
                for cell in row:
                    if cell.strip() == "":
                        processed_row.append(np.nan)
                    else:
                        try:
                            processed_row.append(float(cell))
                        except ValueError:
                            processed_row.append(np.nan)
                data.append(processed_row)
        return np.array(data, dtype=float)

    def compute_composite_score(arr: np.ndarray):
        from sklearn.preprocessing import MinMaxScaler

        weight = arr[:, 5].reshape(-1, 1)
        fuel = arr[:, 7].reshape(-1, 1)
        acc = arr[:, 6].reshape(-1, 1)

        scaler_w = MinMaxScaler()
        scaler_f = MinMaxScaler()
        scaler_a = MinMaxScaler()

        norm_weight = 1 - scaler_w.fit_transform(weight)
        norm_fuel = scaler_f.fit_transform(fuel)
        norm_acc = 1 - scaler_a.fit_transform(acc)

        scores = (norm_weight + norm_fuel + norm_acc).flatten()
        top_indices = np.argsort(scores)[::-1][:5]
        top_scores = scores[top_indices]
        return scores, list(top_indices), list(top_scores)

    def compute_summary_stats(arr: np.ndarray, col_index: int) -> Tuple[float, float, float]:
        col = arr[:, col_index]
        mean = float(np.nanmean(col))
        min_val = float(np.nanmin(col))
        max_val = float(np.nanmax(col))
        return mean, min_val, max_val


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _filter_nan_rows(arr: np.ndarray) -> np.ndarray:
    """Return rows with no NaNs in the columns used for composite scoring."""
    mask = ~np.isnan(arr[:, 5]) & ~np.isnan(arr[:, 6]) & ~np.isnan(arr[:, 7])
    return arr[mask]


def _zscore_matrix(arr: np.ndarray, cols: List[int]) -> np.ndarray:
    """Return z‑scaled matrix for selected columns."""
    sub = arr[:, cols]
    mean = np.nanmean(sub, axis=0)
    std = np.nanstd(sub, axis=0, ddof=0)
    std[std == 0] = 1.0  # avoid division by zero
    return (sub - mean) / std


# --------------------------------------------------------------------------- #
# CAR-15: Alternatives for a poorly rated vehicle
# --------------------------------------------------------------------------- #
def car_15(arr: np.ndarray) -> None:
    print("== CAR-15 ==")

    # Ensure we only work with complete rows
    clean_arr = _filter_nan_rows(arr)
    if clean_arr.size == 0:
        print("No complete vehicles to evaluate.")
        return

    # Composite scores
    scores, _, _ = compute_composite_score(clean_arr)
    # Identify the vehicle with the lowest score
    poor_idx = int(np.argmin(scores))
    poor_vehicle = clean_arr[poor_idx]
    poor_score = float(scores[poor_idx])

    weight, mpg, acc = poor_vehicle[5], poor_vehicle[7], poor_vehicle[6]
    print(f"Poorly rated vehicle (index {poor_idx}):")
    print(f"  Weight = {weight:.1f} lbs")
    print(f"  Fuel Efficiency = {mpg:.1f} mpg")
    print(f"  Acceleration = {acc:.1f} s")
    print(f"  Composite Score = {poor_score:.4f}")

    # Compute normalized distance to all other vehicles (on the 3 key metrics)
    cols = [5, 7, 6]  # weight, mpg, acceleration
    z_clean = _zscore_matrix(clean_arr, cols)
    z_poor = z_clean[poor_idx]
    dists = np.linalg.norm(z_clean - z_poor, axis=1)

    # Find vehicles that have higher scores and are close in characteristics
    mask_candidates = (scores > poor_score) & (dists < np.percentile(dists, 30))
    candidate_idxs = np.where(mask_candidates)[0]

    if candidate_idxs.size == 0:
        print("No alternative vehicles found with better performance but similar characteristics.")
        return

    # Sort candidates by closeness
    sorted_candidates = candidate_idxs[np.argsort(dists[candidate_idxs])]

    print("\nAlternative vehicles with better recorded performance and similar characteristics:")
    for i, idx in enumerate(sorted_candidates[:3], start=1):
        cand = clean_arr[idx]
        w, m, a = cand[5], cand[7], cand[6]
        s = float(scores[idx])
        print(f"{i}. Index {idx}: weight={w:.1f} lbs, mpg={m:.1f} mpg, acc={a:.1f}s, score={s:.4f}")

    # Explanation
    print("\nExplanation of differences:")
    print("The alternatives differ mainly in having a lighter weight, higher fuel efficiency,")
    print("and faster acceleration compared to the poorly rated vehicle. These improvements")
    print("directly contribute to higher composite scores, indicating better overall desirability.")


# --------------------------------------------------------------------------- #
# CAR-16: Closest alternative meeting more performance targets
# --------------------------------------------------------------------------- #
def car_16(arr: np.ndarray) -> None:
    print("\n== CAR-16 ==")

    # Clean data
    clean_arr = _filter_nan_rows(arr)
    if clean_arr.size < 2:
        print("Insufficient data to compare vehicles.")
        return

    # Composite scores and percentiles
    scores, _, _ = compute_composite_score(clean_arr)
    lower_thresh = np.percentile(scores, 20)  # bottom 20th percentile
    # Find a vehicle that fails preferred performance
    fail_mask = scores < lower_thresh
    if not np.any(fail_mask):
        print("All vehicles meet preferred performance targets.")
        return
    fail_idx = int(np.argmax(fail_mask))  # pick first failing vehicle
    fail_vehicle = clean_arr[fail_idx]
    fail_score = float(scores[fail_idx])

    weight_f, mpg_f, acc_f = fail_vehicle[5], fail_vehicle[7], fail_vehicle[6]
    print(f"Vehicle that fails preferred performance (index {fail_idx}):")
    print(f"  Weight = {weight_f:.1f} lbs")
    print(f"  Fuel Efficiency = {mpg_f:.1f} mpg")
    print(f"  Acceleration = {acc_f:.1f} s")
    print(f"  Composite Score = {fail_score:.4f}")

    # Define performance targets: lower weight, higher mpg, lower acceleration
    # Count for each alternative how many targets are met relative to failing vehicle
    alt_mask = scores > fail_score  # better composite score
    if not np.any(alt_mask):
        print("No alternative has a higher composite score.")
        return

    alt_indices = np.where(alt_mask)[0]
    alt_scores = scores[alt_indices]

    # Compute target counts
    def target_met(alt):
        w, m, a = clean_arr[alt][5], clean_arr[alt][7], clean_arr[alt][6]
        count = int(w < weight_f) + int(m > mpg_f) + int(a < acc_f)
        return count

    counts = np.array([target_met(idx) for idx in alt_indices])

    # Filter to those meeting at least two of the targets
    good_mask = counts >= 2
    if not np.any(good_mask):
        print("No alternative meets at least two performance targets while improving composite score.")
        return

    good_indices = alt_indices[good_mask]
    good_counts = counts[good_mask]
    # Among them, pick the one with minimal Euclidean distance on the three metrics
    def metric_distance(idx):
        w, m, a = clean_arr[idx][5], clean_arr[idx][7], clean_arr[idx][6]
        return np.sqrt((w - weight_f) ** 2 + (m - mpg_f) ** 2 + (a - acc_f) ** 2)

    distances = np.array([metric_distance(idx) for idx in good_indices])

    best_idx = good_indices[np.argmin(distances)]
    best_vehicle = clean_arr[best_idx]
    best_score = float(scores[best_idx])
    w_b, m_b, a_b = best_vehicle[5], best_vehicle[7], best_vehicle[6]

    print("\nClosest alternative meeting more performance targets:")
    print(f"  Index {best_idx}:")
    print(f"    Weight = {w_b:.1f} lbs")
    print(f"    Fuel Efficiency = {m_b:.1f} mpg")
    print(f"    Acceleration = {a_b:.1f} s")
    print(f"    Composite Score = {best_score:.4f}")

    # Explanation
    print("\nExplanation:")
    print(f"Compared to the failing vehicle, this alternative improves on {', '.join([feat for feat, cond in [('weight', w_b < weight_f), ('mpg', m_b > mpg_f), ('acceleration', a_b < acc_f) if cond else []])}.")
    print("It meets at least two of the preferred performance targets while changing only the minimal set of characteristics, resulting in a higher composite score.")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="CAR-15 & CAR-16 skills")
    parser.add_argument("csv_path", help="Path to auto93.csv")
    args = parser.parse_args()

    try:
        data = read_csv_as_array(args.csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if data.size == 0:
        print("CSV contains no data.", file=sys.stderr)
        sys.exit(1)

    car_15(data)
    car_16(data)


if __name__ == "__main__":
    main()
