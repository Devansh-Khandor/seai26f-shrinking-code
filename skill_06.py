#!/usr/bin/env python3
"""
skill_06.py

Entry point for CAR-11 and CAR-12 skills.

Usage:
    python skill_06.py <path/to/auto93.csv>
"""

import argparse
import sys
from typing import Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Reuse utilities from earlier modules
# --------------------------------------------------------------------------- #
try:
    from skill_02 import read_csv_as_array, compute_summary_stats, compute_composite_score
except Exception:
    # Fallback minimal implementations if skill_02 is missing
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

    def compute_summary_stats(arr: np.ndarray, col_index: int) -> Tuple[float, float, float]:
        col = arr[:, col_index]
        mean = float(np.nanmean(col))
        min_val = float(np.nanmin(col))
        max_val = float(np.nanmax(col))
        return mean, min_val, max_val

    def compute_composite_score(arr: np.ndarray) -> Tuple[np.ndarray, list, list]:
        # Basic implementation (ignores NaNs)
        weight = arr[:, 5].reshape(-1, 1)
        fuel = arr[:, 7].reshape(-1, 1)
        acc = arr[:, 6].reshape(-1, 1)

        from sklearn.preprocessing import MinMaxScaler

        scaler_w = MinMaxScaler()
        scaler_f = MinMaxScaler()
        scaler_a = MinMaxScaler()

        norm_weight = 1 - scaler_w.fit_transform(weight)  # lower is better
        norm_fuel = scaler_f.fit_transform(fuel)          # higher is better
        norm_acc = 1 - scaler_a.fit_transform(acc)        # lower is better

        scores = (norm_weight + norm_fuel + norm_acc).flatten()
        top_indices = np.argsort(scores)[::-1][:5]
        top_scores = scores[top_indices]
        return scores, list(top_indices), list(top_scores)

# --------------------------------------------------------------------------- #
# Helper functions for CAR-11
# --------------------------------------------------------------------------- #
def explain_recommendation(arr: np.ndarray) -> None:
    """
    Prints an explanation for the top‑scoring vehicle using comparisons to
    the overall inventory.
    """
    if arr.shape[0] == 0:
        print("No vehicles in inventory.")
        return

    scores, top_indices, top_scores = compute_composite_score(arr)
    top_idx = top_indices[0]
    top_score = top_scores[0]

    # Vehicle features
    weight = arr[top_idx, 5]
    mpg = arr[top_idx, 7]
    acc = arr[top_idx, 6]

    # Summary statistics for comparison
    w_mean, w_min, w_max = compute_summary_stats(arr, 5)
    mpg_mean, mpg_min, mpg_max = compute_summary_stats(arr, 7)
    acc_mean, acc_min, acc_max = compute_summary_stats(arr, 6)

    # Percentage deviations from mean
    w_pct = (weight - w_mean) / w_mean * 100
    mpg_pct = (mpg - mpg_mean) / mpg_mean * 100
    acc_pct = (acc - acc_mean) / acc_mean * 100

    print(f"Vehicle at index {top_idx} has a composite score of {top_score:.4f}.")
    print(f"  Weight: {weight:.1f} lbs ({w_pct:+.1f}% relative to inventory mean).")
    print(f"  Fuel efficiency: {mpg:.1f} mpg ({mpg_pct:+.1f}% relative to inventory mean).")
    print(f"  Acceleration: {acc:.1f} s ({acc_pct:+.1f}% relative to inventory mean).")
    print(f"\nThis vehicle scores high because it combines a relatively low weight,"
          " high fuel efficiency, and quick acceleration compared to most other vehicles in the shop.")


# --------------------------------------------------------------------------- #
# Helper functions for CAR-12
# --------------------------------------------------------------------------- #
def compare_top_two(arr: np.ndarray) -> None:
    """
    Prints a side‑by‑side comparison of the best two vehicles, highlighting
    trade‑offs that explain the higher overall recommendation of the top one.
    """
    if arr.shape[0] < 2:
        print("Not enough vehicles to compare.")
        return

    scores, top_indices, top_scores = compute_composite_score(arr)
    idx1, idx2 = top_indices[0], top_indices[1]
    score1, score2 = top_scores[0], top_scores[1]

    v1 = arr[idx1]
    v2 = arr[idx2]

    # Feature names for readability
    features = ["Weight", "Fuel efficiency", "Acceleration"]

    def pct_change(val, mean):
        return (val - mean) / mean * 100

    w_mean, _, _ = compute_summary_stats(arr, 5)
    mpg_mean, _, _ = compute_summary_stats(arr, 7)
    acc_mean, _, _ = compute_summary_stats(arr, 6)

    print(f"Top two vehicles:")
    print(f"  Vehicle A (index {idx1}): score={score1:.4f}")
    print(f"  Vehicle B (index {idx2}): score={score2:.4f}\n")

    print("Feature comparison:")
    print(f"{'Feature':>20} | {'Vehicle A':>15} | {'Vehicle B':>15} | {'Trade‑off'}")
    print("-" * 70)
    for feat_name, col, mean_val in zip(features, [5, 7, 6], [w_mean, mpg_mean, acc_mean]):
        val_a = v1[col]
        val_b = v2[col]
        pct_a = pct_change(val_a, mean_val)
        pct_b = pct_change(val_b, mean_val)
        # Determine which vehicle is better for this metric
        if feat_name == "Weight" or feat_name == "Acceleration":
            # lower is better
            better = idx1 if val_a < val_b else idx2
        else:  # Fuel efficiency
            better = idx1 if val_a > val_b else idx2
        tradeoff = (f"Vehicle {'A' if better == idx1 else 'B'} better"
                    f" ({abs(val_a-val_b):.1f} units)")
        print(f"{feat_name:>20} | {val_a:>15.1f} ({pct_a:+.1f}%) |"
              f" {val_b:>15.1f} ({pct_b:+.1f}%) | {tradeoff}")

    print("\nInterpretation:")
    print(f"Vehicle A outperforms Vehicle B in the following metrics: "
          f"({', '.join([feat for feat, col, _ in zip(features, [5, 7, 6], [w_mean, mpg_mean, acc_mean])])})")
    print(f"These gains outweigh any disadvantages in other areas, leading to a higher composite score.")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="CAR-11 & CAR-12 skills")
    parser.add_argument("csv_path", help="Path to auto93.csv")
    args = parser.parse_args()

    # Load data
    try:
        data = read_csv_as_array(args.csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if data.size == 0:
        print("CSV contains no data.", file=sys.stderr)
        sys.exit(1)

    # CAR-11
    print("== CAR-11 ==")
    explain_recommendation(data)

    # CAR-12
    print("\n== CAR-12 ==")
    compare_top_two(data)


if __name__ == "__main__":
    main()
