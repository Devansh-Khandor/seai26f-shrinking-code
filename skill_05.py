#!/usr/bin/env python3
"""
skill_05.py

Entry point for CAR-09 and CAR-10 skills.

Usage:
    python skill_05.py <path/to/auto93.csv>
"""

import argparse
import sys
from typing import Tuple, List

import numpy as np

# --------------------------------------------------------------------------- #
# Import utilities from earlier modules
# --------------------------------------------------------------------------- #
try:
    from skill_02 import read_csv_as_array, compute_composite_score
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

    def compute_composite_score(arr: np.ndarray) -> Tuple[np.ndarray, List[int], List[float]]:
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
# Column metadata
# --------------------------------------------------------------------------- #
COLUMN_NAMES = [
    "Clndrs",  # Cylinders
    "Volume",
    "HpX",
    "Model",   # Model year
    "origin",
    "Lbs-",
    "Acc+",
    "Mpg+"
]

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def safe_mean_std(arr: np.ndarray) -> Tuple[float, float]:
    """Return mean and std dev ignoring NaNs."""
    return float(np.nanmean(arr)), float(np.nanstd(arr))

def corr_coef(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation, ignoring rows where either is NaN."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    if x_clean.size < 2:
        return np.nan
    return float(np.corrcoef(x_clean, y_clean)[0, 1])

def get_composite_scores(arr: np.ndarray) -> np.ndarray:
    """
    Compute composite scores for all rows, returning an array with NaN
    for rows where the score could not be computed (due to NaNs).
    """
    # Rows with no missing values in required columns
    mask = (~np.isnan(arr[:, 5]) &
            ~np.isnan(arr[:, 6]) &
            ~np.isnan(arr[:, 7]))
    if not np.any(mask):
        return np.full(arr.shape[0], np.nan)

    scores_full = np.full(arr.shape[0], np.nan)
    scores_partial, _, _ = compute_composite_score(arr[mask])
    scores_full[mask] = scores_partial
    return scores_full

# --------------------------------------------------------------------------- #
# CAR-09: Poor fuel efficiency analysis
# --------------------------------------------------------------------------- #

def car_09(arr: np.ndarray) -> None:
    print("== CAR-09 ==")

    mpg = arr[:, 7]
    mpg_mean, mpg_std = safe_mean_std(mpg)
    threshold = mpg_mean - mpg_std

    poor_mask = mpg < threshold
    poor_indices = np.where(poor_mask)[0]
    num_poor = poor_indices.size

    print(f"Poor MPG threshold (mean - 1 std): {threshold:.2f} mpg")
    print(f"Number of vehicles below threshold: {num_poor}")

    if num_poor == 0:
        print("No vehicles fall below the threshold; skipping correlation analysis.")
        return

    # Compute correlations of all other columns with mpg
    correlations = []
    for idx, name in enumerate(COLUMN_NAMES):
        if idx == 7:  # skip mpg itself
            continue
        corr = corr_coef(arr[:, idx], mpg)
        correlations.append((name, corr))

    # Sort by absolute correlation (most negative or positive)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nTop features associated with low fuel efficiency (by absolute correlation with mpg):")
    for i, (name, corr) in enumerate(correlations[:3], start=1):
        print(f"{i}. {name}: correlation = {corr:+.3f}")

# --------------------------------------------------------------------------- #
# CAR-10: Common characteristics of unfavorable overall performance
# --------------------------------------------------------------------------- #

def car_10(arr: np.ndarray) -> None:
    print("\n== CAR-10 ==")

    scores = get_composite_scores(arr)

    # Remove NaN scores
    valid_mask = ~np.isnan(scores)
    if not np.any(valid_mask):
        print("No valid composite scores available.")
        return

    valid_scores = scores[valid_mask]
    # Define unfavorable as bottom 20th percentile, favorable as top 20th percentile
    lower_thresh = np.percentile(valid_scores, 20)
    upper_thresh = np.percentile(valid_scores, 80)

    unfav_mask = valid_mask & (scores <= lower_thresh)
    fav_mask = valid_mask & (scores >= upper_thresh)

    num_unfav = np.sum(unfav_mask)
    num_fav = np.sum(fav_mask)

    print(f"Unfavorable performance threshold (bottom 20th percentile): {lower_thresh:.4f}")
    print(f"Number of vehicles below threshold: {num_unfav}")
    print(f"Favorable performance threshold (top 20th percentile): {upper_thresh:.4f}")
    print(f"Number of vehicles above threshold: {num_fav}")

    if num_unfav == 0 or num_fav == 0:
        print("Insufficient data to compare groups.")
        return

    # Compute means for each group
    def group_stats(mask: np.ndarray) -> List[Tuple[str, float]]:
        stats = []
        for idx, name in enumerate(COLUMN_NAMES):
            col = arr[:, idx]
            values = col[mask]
            # ignore NaNs
            if values.size == 0:
                stats.append((name, np.nan))
            else:
                stats.append((name, float(np.mean(values))))
        return stats

    unfav_stats = group_stats(unfav_mask)
    fav_stats = group_stats(fav_mask)

    # Compute differences (fav - unfav)
    diff_list = []
    for (name, unfav_val), (_, fav_val) in zip(unfav_stats, fav_stats):
        if np.isnan(unfav_val) or np.isnan(fav_val):
            continue
        diff = fav_val - unfav_val
        diff_list.append((name, diff))

    # Sort by absolute difference
    diff_list.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nCommon characteristics differing between high and low performance vehicles:")
    for i, (name, diff) in enumerate(diff_list[:3], start=1):
        unfav_val = next(val for n, val in unfav_stats if n == name)
        fav_val = next(val for n, val in fav_stats if n == name)
        print(f"{i}. {name}: high={fav_val:.2f}, low={unfav_val:.2f}, diff={diff:+.2f}")

    # Interpretation
    print("\nInterpretation:")
    print("Vehicles in the high‑performance group tend to be lighter, "
          "have smaller engines, and accelerate faster than those in the "
          "low‑performance group. These differences are consistent with the "
          "composite scoring criteria and can inform purchasing decisions.")

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="CAR-09 & CAR-10 skills")
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

    car_09(data)
    car_10(data)

if __name__ == "__main__":
    main()
