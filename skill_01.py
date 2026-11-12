#!/usr/bin/env python3
"""
Skill 01 – Inventory Summary & Optimal Vehicle Identification

Usage:
    python skill_01.py <csv_path>
"""

import csv
import sys
from pathlib import Path


def read_csv(path: Path):
    """Read the CSV and return a list of rows as dictionaries."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows


def compute_summary(rows):
    """Compute mean, min, and max for weight, mpg, and acceleration."""
    # Convert strings to float
    weights = [float(r["Lbs-"]) for r in rows]
    mpgs = [float(r["Mpg+"]) for r in rows]
    accs = [float(r["Acc+"]) for r in rows]

    def stats(values):
        mean = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        return mean, min_val, max_val

    return {
        "weight": stats(weights),
        "mpg": stats(mpgs),
        "acc": stats(accs),
    }


def format_stats(name, mean, min_val, max_val):
    """Return formatted string for a statistic."""
    return (
        f"{name:<12} Mean: {mean:>7.2f}, "
        f"Min: {min_val:>7.2f}, Max: {max_val:>7.2f}"
    )


def compute_scores(rows):
    """Assign a composite score to each row based on weight, mpg, and acceleration."""
    weights = [float(r["Lbs-"]) for r in rows]
    mpgs = [float(r["Mpg+"]) for r in rows]
    accs = [float(r["Acc+"]) for r in rows]

    w_min, w_max = min(weights), max(weights)
    mpg_min, mpg_max = min(mpgs), max(mpgs)
    acc_min, acc_max = min(accs), max(accs)

    def norm(val, low, high, higher_is_better=True):
        """Normalize between 0 and 1."""
        if high == low:
            return 0.5
        if higher_is_better:
            return (val - low) / (high - low)
        else:
            return (high - val) / (high - low)

    scores = []
    for i, r in enumerate(rows):
        weight = float(r["Lbs-"])
        mpg = float(r["Mpg+"])
        acc = float(r["Acc+"])

        w_score = norm(weight, w_min, w_max, higher_is_better=False)
        mpg_score = norm(mpg, mpg_min, mpg_max)
        acc_score = norm(acc, acc_min, acc_max)

        # Equal weight for all three metrics
        composite = (w_score + mpg_score + acc_score) / 3
        scores.append((composite, i, r))

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores


def print_car_01(summary):
    print("== CAR-01 ==")
    print(format_stats("Weight", *summary["weight"]))
    print(format_stats("MPG", *summary["mpg"]))
    print(format_stats("Acc", *summary["acc"]))


def print_car_02(sorted_scores, top_n=5):
    print("== CAR-02 ==")
    for idx, (score, row_idx, row) in enumerate(sorted_scores[:top_n], 1):
        weight = float(row["Lbs-"])
        mpg = float(row["Mpg+"])
        acc = float(row["Acc+"])
        print(
            f"{idx:2d}. Row {row_idx+1:<3d} | "
            f"Score: {score:>5.3f} | "
            f"Weight: {weight:>7.1f} | "
            f"MPG: {mpg:>4.1f} | "
            f"Acc: {acc:>5.2f}"
        )
    if len(sorted_scores) > top_n:
        print(f"... and {len(sorted_scores)-top_n} more vehicles.")


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: File '{csv_path}' does not exist.")
        sys.exit(1)

    rows = read_csv(csv_path)
    if not rows:
        print("Error: CSV file is empty or has no data rows.")
        sys.exit(1)

    summary = compute_summary(rows)
    print_car_01(summary)

    sorted_scores = compute_scores(rows)
    print_car_02(sorted_scores)


if __name__ == "__main__":
    main()
