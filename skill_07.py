#!/usr/bin/env python3
"""
skill_07.py

Entry point for CAR-13 and CAR-14 skills.

Usage:
    python skill_07.py <path/to/auto93.csv>
"""

import argparse
import sys

import numpy as np

# Reuse utilities that already handle NaNs gracefully
try:
    from skill_02 import read_csv_as_array, compute_composite_score, compute_summary_stats
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

    def compute_composite_score(arr: np.ndarray) -> tuple[np.ndarray, list[int], list[float]]:
        from sklearn.preprocessing import MinMaxScaler
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

    def compute_summary_stats(arr: np.ndarray, col_index: int):
        col = arr[:, col_index]
        mean = float(np.nanmean(col))
        min_val = float(np.nanmin(col))
        max_val = float(np.nanmax(col))
        return mean, min_val, max_val


# --------------------------------------------------------------------------- #
# CAR-13: Identify a vehicle with a strong balance of fuel efficiency, weight, and acceleration
# --------------------------------------------------------------------------- #

def car_13(arr: np.ndarray) -> None:
    """Find the single vehicle that best balances fuel efficiency, weight, and acceleration."""
    print("== CAR-13 ==")

    # Compute composite scores for all vehicles
    scores, _, _ = compute_composite_score(arr)

    # Identify the best-scoring vehicle
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    # Extract key metrics
    weight = float(arr[best_idx, 5])
    mpg = float(arr[best_idx, 7])
    acc = float(arr[best_idx, 6])

    print(f"Vehicle with the best overall balance:")
    print(f"  Index: {best_idx}")
    print(f"  Weight: {weight:.1f} lbs")
    print(f"  Fuel Efficiency: {mpg:.1f} mpg")
    print(f"  Acceleration: {acc:.1f} s")
    print(f"  Composite Score: {best_score:.4f}")

    # Provide a short explanatory sentence
    print("\nThis vehicle achieves a high composite score because it combines a relatively low weight, "
          "high fuel efficiency, and quick acceleration compared to the rest of the inventory.")


# --------------------------------------------------------------------------- #
# CAR-14: Recommend vehicles to inspect first with a limited budget
# --------------------------------------------------------------------------- #

def car_14(arr: np.ndarray) -> None:
    """Suggest the most promising vehicles to inspect first when resources are limited."""
    print("\n== CAR-14 ==")

    # Compute composite scores and sort descending
    scores, top_indices, top_scores = compute_composite_score(arr)
    sorted_idx = np.argsort(scores)[::-1]  # descending order
    sorted_scores = scores[sorted_idx]

    # We recommend inspecting the top 5 vehicles first.
    # This is a pragmatic balance between thoroughness and efficiency.
    recommend_count = min(5, len(sorted_idx))
    print(f"Recommend inspecting the top {recommend_count} vehicles first to maximize discovery of high‑value inventory.")

    print("\nTop candidates (index, weight, mpg, acceleration, composite score):")
    for i in range(recommend_count):
        idx = sorted_idx[i]
        weight = float(arr[idx, 5])
        mpg = float(arr[idx, 7])
        acc = float(arr[idx, 6])
        score = float(sorted_scores[i])
        print(f"{i+1}. Index {idx}: weight={weight:.1f} lbs, mpg={mpg:.1f} mpg, acc={acc:.1f} s, score={score:.4f}")

    print("\nRationale:")
    print("These vehicles lead the inventory in the composite metric, indicating a superior combination of "
          "light weight, efficient fuel usage, and quick acceleration. Inspecting them first increases the "
          "likelihood of acquiring high‑quality, sellable cars early in the inspection cycle.")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="CAR-13 & CAR-14 skills")
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

    # Execute CAR-13
    car_13(data)

    # Execute CAR-14
    car_14(data)


if __name__ == "__main__":
    main()
