#!/usr/bin/env python3
"""
Skill 01 – Inventory Summary and Best-Combination Vehicle Identification

Usage:
    python skill_01.py <path-to-auto93.csv>
"""

import csv
import sys
import numpy as np


def load_csv(path: str) -> tuple[np.ndarray, list[str]]:
    """Load the CSV file and return a NumPy array of numeric values
    and a list of column headers. Empty strings are interpreted as NaN."""
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for r in reader:
            # Convert all values to float; allow empty fields → NaN
            row = []
            for v in r:
                if v == '':
                    row.append(np.nan)
                else:
                    row.append(float(v))
            rows.append(row)
        data = np.array(rows, dtype=float)
    return data, header


def compute_summary(data: np.ndarray, header: list[str], col_name: str) -> dict:
    """Return summary statistics for a specified column."""
    idx = header.index(col_name)
    col = data[:, idx]
    return {
        "mean": np.nanmean(col),
        "median": np.nanmedian(col),
        "std": np.nanstd(col, ddof=1),
        "min": np.nanmin(col),
        "max": np.nanmax(col),
    }


def print_summary(col_name: str, stats: dict):
    """Print summary statistics in a readable format."""
    print(f"{col_name} Summary:")
    print(f"  Mean   : {stats['mean']:.2f}")
    print(f"  Median : {stats['median']:.2f}")
    print(f"  Std Dev: {stats['std']:.2f}")
    print(f"  Min    : {stats['min']:.2f}")
    print(f"  Max    : {stats['max']:.2f}")


def compute_scores(data: np.ndarray, header: list[str]) -> np.ndarray:
    """Compute a composite score for each vehicle based on weight, mpg, and acceleration."""
    weight_idx = header.index("Lbs-")
    mpg_idx = header.index("Mpg+")
    accel_idx = header.index("Acc+")

    weight = data[:, weight_idx]
    mpg = data[:, mpg_idx]
    accel = data[:, accel_idx]

    # Z‑scores
    z_weight = (weight - weight.mean()) / weight.std(ddof=1)
    z_mpg = (mpg - mpg.mean()) / mpg.std(ddof=1)
    z_accel = (accel - accel.mean()) / accel.std(ddof=1)

    # Lower weight and lower acceleration are better → use negative z‑score
    score = -z_weight + z_mpg - z_accel
    return score


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv-path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        data, header = load_csv(csv_path)
    except Exception as exc:
        print(f"Error loading CSV: {exc}")
        sys.exit(1)

    # CAR-01: Inventory summary
    print("== CAR-01 ==")
    for col in ("Lbs-", "Mpg+", "Acc+"):
        stats = compute_summary(data, header, col)
        print_summary(col, stats)

    # CAR-02: Best combination vehicles
    print("\n== CAR-02 ==")
    scores = compute_scores(data, header)

    # Get indices of top 5 vehicles
    top_indices = np.argsort(scores)[::-1][:5]

    for rank, idx in enumerate(top_indices, 1):
        row = data[idx]
        weight, mpg, accel = row[header.index("Lbs-")], row[header.index("Mpg+")], row[header.index("Acc+")]
        print(f"Rank {rank}: Vehicle #{idx + 1} → Weight={weight:.2f} Lbs, "
              f"MPG={mpg:.2f}, Accel={accel:.2f}, Score={scores[idx]:.2f}")


if __name__ == "__main__":
    main()
