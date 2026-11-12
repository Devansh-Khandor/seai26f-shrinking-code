#!/usr/bin/env python3
"""
skill_01.py

Entry point for CAR-01 and CAR-02 skill requests.

Usage:
    python skill_01.py <path/to/auto93.csv>
"""

import argparse
import csv
import sys
from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def read_csv_as_array(path: str) -> np.ndarray:
    """
    Reads a CSV file containing only numeric values into a NumPy array.
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    # Convert to float; assume all columns are numeric
    return np.array(data, dtype=float)


def compute_summary_stats(arr: np.ndarray, col_index: int) -> Tuple[float, float, float]:
    """
    Computes mean, min, and max for a specified column.
    """
    col = arr[:, col_index]
    mean = float(np.mean(col))
    min_val = float(np.min(col))
    max_val = float(np.max(col))
    return mean, min_val, max_val


def compute_composite_score(arr: np.ndarray) -> Tuple[np.ndarray, List[int], List[float]]:
    """
    Computes a composite score for each vehicle based on
    - Weight (column 5): lower is better
    - Fuel Efficiency (column 7): higher is better
    - Acceleration (column 6): lower is better
    Returns:
        scores: array of composite scores (higher is better)
        top_indices: list of indices of top 5 vehicles (descending order)
        top_scores: list of scores for those vehicles
    """
    weight = arr[:, 5].reshape(-1, 1)
    fuel = arr[:, 7].reshape(-1, 1)
    acc = arr[:, 6].reshape(-1, 1)

    # Normalize each feature to [0, 1]
    scaler_w = MinMaxScaler()
    scaler_f = MinMaxScaler()
    scaler_a = MinMaxScaler()

    norm_weight = 1 - scaler_w.fit_transform(weight)  # lower is better
    norm_fuel = scaler_f.fit_transform(fuel)          # higher is better
    norm_acc = 1 - scaler_a.fit_transform(acc)        # lower is better

    # Composite score: equal weights
    scores = (norm_weight + norm_fuel + norm_acc).flatten()

    # Get top 5 indices (descending)
    top_indices = np.argsort(scores)[::-1][:5]
    top_scores = scores[top_indices]
    return scores, list(top_indices), list(top_scores)


def main():
    parser = argparse.ArgumentParser(description="CAR-01 & CAR-02 skills")
    parser.add_argument("csv_path", help="Path to auto93.csv")
    args = parser.parse_args()

    # Load data
    try:
        data = read_csv_as_array(args.csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # CAR-01
    print("== CAR-01 ==")
    # Weight
    weight_mean, weight_min, weight_max = compute_summary_stats(data, 5)
    print(f"Weight - mean: {weight_mean:.2f} lbs, min: {weight_min:.2f} lbs, max: {weight_max:.2f} lbs")
    # Fuel Efficiency
    fuel_mean, fuel_min, fuel_max = compute_summary_stats(data, 7)
    print(f"Fuel Efficiency - mean: {fuel_mean:.2f} mpg, min: {fuel_min:.2f} mpg, max: {fuel_max:.2f} mpg")
    # Acceleration
    acc_mean, acc_min, acc_max = compute_summary_stats(data, 6)
    print(f"Acceleration - mean: {acc_mean:.2f} s, min: {acc_min:.2f} s, max: {acc_max:.2f} s")

    # CAR-02
    print("\n== CAR-02 ==")
    scores, top_indices, top_scores = compute_composite_score(data)
    print("Top 5 vehicles by composite score (index, weight, mpg, acceleration, score):")
    for idx, score in zip(top_indices, top_scores):
        w = data[idx, 5]
        mpg = data[idx, 7]
        acc = data[idx, 6]
        print(f"{idx}: weight={w:.1f} lbs, mpg={mpg:.1f}, acc={acc:.1f}s, score={score:.4f}")


if __name__ == "__main__":
    main()
