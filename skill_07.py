#!/usr/bin/env python3
"""
Skill 07 – Inventory Selection & Inspection Prioritization

Usage:
    python skill_07.py <path-to-auto93.csv>
"""

import sys
import numpy as np
from skill_01 import load_csv, compute_scores, compute_summary, print_summary


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv-path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        data, header = load_csv(csv_path)
    except Exception as exc:
        print(f"Error loading CSV: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # CAR-13: Identify vehicle with balanced fuel efficiency, weight and acceleration
    # ------------------------------------------------------------------
    print("\n== CAR-13 ==")

    scores = compute_scores(data, header)
    # Get top 3 vehicles by composite score (higher is better)
    top3_idx = np.argsort(scores)[::-1][:3]

    print("Top 3 balanced vehicles by composite score (higher is better):")
    for rank, idx in enumerate(top3_idx, 1):
        weight = data[idx, header.index("Lbs-")]
        mpg = data[idx, header.index("Mpg+")]
        accel = data[idx, header.index("Acc+")]
        score = scores[idx]
        print(
            f"  Rank {rank}: Vehicle #{idx + 1} → "
            f"Weight={weight:.1f} lbs, MPG={mpg:.1f}, Accel={accel:.1f}, Score={score:.2f}"
        )

    best_idx = top3_idx[0]
    print("\nRecommendation:")
    print(f"  Vehicle #{best_idx + 1} has the strongest balance of fuel efficiency, weight, and acceleration.")

    # ------------------------------------------------------------------
    # CAR-14: Prioritize vehicles for inspection under a limited budget
    # ------------------------------------------------------------------
    print("\n== CAR-14 ==")

    # Assume the inspection budget allows evaluating 5 vehicles
    budget = 5
    top_budget_idx = np.argsort(scores)[::-1][:budget]

    print(f"With a limited inspection budget of {budget} vehicles, we recommend evaluating first:")
    for rank, idx in enumerate(top_budget_idx, 1):
        weight = data[idx, header.index("Lbs-")]
        mpg = data[idx, header.index("Mpg+")]
        accel = data[idx, header.index("Acc+")]
        score = scores[idx]
        print(
            f"  Rank {rank}: Vehicle #{idx + 1} → "
            f"Weight={weight:.1f} lbs, MPG={mpg:.1f}, Accel={accel:.1f}, Score={score:.2f}"
        )

    # Optional: provide quick summaries of key columns
    print("\nKey column summaries:")
    for col in ("Lbs-", "Mpg+", "Acc+"):
        stats = compute_summary(data, header, col)
        print_summary(col, stats)


if __name__ == "__main__":
    main()
