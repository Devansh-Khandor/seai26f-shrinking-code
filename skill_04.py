#!/usr/bin/env python3
"""
Skill 04 – Real‑Time Assessment & Dynamic Inventory Summary

Usage:
    python skill_04.py <path-to-auto93.csv>
"""

import sys
import numpy as np

# Reuse helpers from previous skills
from skill_01 import (
    load_csv,
    compute_summary,
    print_summary,
    compute_scores,
)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv-path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        data, header = load_csv(csv_path)
    except Exception as exc:
        print(f"Error loading CSV: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # CAR-07: Immediate desirability assessment for each vehicle
    # ------------------------------------------------------------------
    print("\n== CAR-07 ==")

    # Compute composite score for all vehicles
    scores = compute_scores(data, header)
    # Determine top‑5 desirable indices
    top5_idx = np.argsort(scores)[::-1][:5]

    # Report desirability for first few vehicles (at least 3 lines)
    for i, score in enumerate(scores):
        status = "Desirable" if i in top5_idx else "Not desirable"
        print(
            f"Vehicle #{i + 1}: Score={score:.2f} → {status}"
        )
        if i >= 2:  # ensure at least 3 result lines
            break

    # ------------------------------------------------------------------
    # CAR-08: Updated inventory summaries and recommended shortlist
    # ------------------------------------------------------------------
    print("\n== CAR-08 ==")

    # 1. Summary statistics for key columns
    for col in ("Lbs-", "Mpg+", "Acc+"):
        stats = compute_summary(data, header, col)
        print_summary(col, stats)

    # 2. Recommended vehicle shortlist (top 5 by composite score)
    print("\nRecommended vehicles (top 5):")
    for rank, idx in enumerate(top5_idx, 1):
        weight = data[idx, header.index("Lbs-")]
        mpg = data[idx, header.index("Mpg+")]
        accel = data[idx, header.index("Acc+")]
        print(
            f"Rank {rank}: Vehicle #{idx + 1} → "
            f"Weight={weight:.2f} Lbs, MPG={mpg:.2f}, Accel={accel:.2f}, "
            f"Score={scores[idx]:.2f}"
        )


if __name__ == "__main__":
    main()
