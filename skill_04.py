#!/usr/bin/env python3
"""
skill_04.py

Entry point for CAR-07 and CAR-08 skills.

Usage:
    python skill_04.py <path/to/auto93.csv>
"""

import argparse
import sys
from typing import Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Reuse utilities that already handle NaNs gracefully
try:
    from skill_02 import read_csv_as_array, compute_summary_stats, compute_composite_score
except Exception:
    # Fallback minimal implementations if skill_02 is missing
    def read_csv_as_array(path: str) -> np.ndarray:
        with open(path, newline="") as f:
            import csv
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

    def compute_summary_stats(arr: np.ndarray, col_index: int):
        col = arr[:, col_index]
        mean = float(np.nanmean(col))
        min_val = float(np.nanmin(col))
        max_val = float(np.nanmax(col))
        return mean, min_val, max_val

    def compute_composite_score(arr: np.ndarray):
        # Simple implementation using MinMax scaling
        weight = arr[:, 5].reshape(-1, 1)
        fuel = arr[:, 7].reshape(-1, 1)
        acc = arr[:, 6].reshape(-1, 1)

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
# Helper functions specific to CAR-07
# --------------------------------------------------------------------------- #

def composite_score_for_record(record: np.ndarray,
                               existing_arr: np.ndarray) -> float:
    """
    Compute the composite score for a single new record using the min/max
    derived from the existing inventory (excluding the record itself).
    """
    # Extract the columns used in scoring
    weight_col = existing_arr[:, 5].reshape(-1, 1)
    fuel_col = existing_arr[:, 7].reshape(-1, 1)
    acc_col = existing_arr[:, 6].reshape(-1, 1)

    # Fit scalers to existing data
    scaler_w = MinMaxScaler()
    scaler_f = MinMaxScaler()
    scaler_a = MinMaxScaler()

    scaler_w.fit(weight_col)
    scaler_f.fit(fuel_col)
    scaler_a.fit(acc_col)

    # Normalize the new record
    norm_weight = 1 - scaler_w.transform(record[5].reshape(1, 1))
    norm_fuel = scaler_f.transform(record[7].reshape(1, 1))
    norm_acc = 1 - scaler_a.transform(record[6].reshape(1, 1))

    # Composite score = average of the three normalized metrics
    score = float(np.mean([norm_weight, norm_fuel, norm_acc]))
    return score


def evaluate_new_record(new_record: np.ndarray,
                        existing_arr: np.ndarray) -> Tuple[float, bool, float]:
    """
    Returns the composite score of the new record, a boolean indicating
    whether it is among the top 5 of the existing inventory, and the
    threshold score (minimum score among current top 5).
    """
    # Compute score for the new record
    new_score = composite_score_for_record(new_record, existing_arr)

    # Compute existing top 5 scores (excluding the new record)
    existing_scores, _, existing_top_scores = compute_composite_score(existing_arr)

    if len(existing_top_scores) == 0:
        # No existing inventory; automatically qualifies
        return new_score, True, 0.0

    threshold = min(existing_top_scores)

    is_top5 = new_score >= threshold
    return new_score, is_top5, threshold


# --------------------------------------------------------------------------- #
# CAR-07: Immediate assessment of the last record in the CSV
# --------------------------------------------------------------------------- #

def print_car07(new_record: np.ndarray, existing_arr: np.ndarray) -> None:
    """
    Prints whether the new record qualifies as a desirable inventory option.
    """
    print("== CAR-07 ==")
    score, qualifies, threshold = evaluate_new_record(new_record, existing_arr)

    print(f"Composite score of new vehicle: {score:.4f}")
    print(f"Threshold (min score among current top 5): {threshold:.4f}")
    decision = "YES" if qualifies else "NO"
    print(f"Does the new vehicle belong among the more desirable options? {decision}")


# --------------------------------------------------------------------------- #
# CAR-08: Updated inventory summaries and recommended shortlist
# --------------------------------------------------------------------------- #

def print_car08(arr: np.ndarray) -> None:
    """
    Prints updated summary statistics and the top 5 vehicle shortlist.
    """
    print("\n== CAR-08 ==")

    # Summary statistics for weight, fuel efficiency, acceleration
    weight_mean, weight_min, weight_max = compute_summary_stats(arr, 5)
    fuel_mean, fuel_min, fuel_max = compute_summary_stats(arr, 7)
    acc_mean, acc_min, acc_max = compute_summary_stats(arr, 6)

    print("Inventory Summary Stats:")
    print(f"  Weight - mean: {weight_mean:.1f} lbs, min: {weight_min:.1f} lbs, max: {weight_max:.1f} lbs")
    print(f"  Fuel Efficiency - mean: {fuel_mean:.1f} mpg, min: {fuel_min:.1f} mpg, max: {fuel_max:.1f} mpg")
    print(f"  Acceleration - mean: {acc_mean:.1f} s, min: {acc_min:.1f} s, max: {acc_max:.1f} s")

    # Top 5 vehicles by composite score
    _, top_indices, top_scores = compute_composite_score(arr)
    print("\nTop 5 vehicles by composite score (index, weight, mpg, acceleration, score):")
    for idx, score in zip(top_indices, top_scores):
        w = arr[idx, 5]
        mpg = arr[idx, 7]
        acc = arr[idx, 6]
        print(f"  {idx}: weight={w:.1f} lbs, mpg={mpg:.1f}, acc={acc:.1f}s, score={score:.4f}")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="CAR-07 & CAR-08 skills")
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

    # Treat the last row as the newly arrived record
    new_record = data[-1]
    existing_arr = data[:-1] if data.shape[0] > 1 else np.empty((0, data.shape[1]))

    # CAR-07
    print_car07(new_record, existing_arr)

    # CAR-08
    print_car08(data)


if __name__ == "__main__":
    main()
