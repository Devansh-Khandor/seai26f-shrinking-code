#!/usr/bin/env python3
"""
Skill 04: Real‑time vehicle assessment and live inventory update.

Usage:
    python skill_04.py <path-to-auto93.csv>
"""

import sys
import csv
import statistics
from pathlib import Path

# Reuse functions from the previous skills
from skill_01 import load_data, compute_stats, compute_balanced_scores, format_value


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <path-to-auto93.csv>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    if not Path(csv_path).is_file():
        print(f"Error: file '{csv_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Load data once; in a real streaming scenario the file would be read line‑by‑line,
    # but for this skill we treat the whole file as the current inventory.
    rows = load_data(csv_path)
    if not rows:
        print("Error: no data loaded.", file=sys.stderr)
        sys.exit(1)

    # Compute balanced scores for all rows
    scored_rows = compute_balanced_scores(rows)

    # Top 3 vehicles (best scores)
    top_n = min(3, len(scored_rows))
    top3 = scored_rows[:top_n]
    top3_set = set(top3)  # set for quick lookup

    # -----------------------------
    # CAR-07: immediate assessment
    # -----------------------------
    print("== CAR-07 ==")
    for r in scored_rows:
        model = r.get("Model", "Unknown")
        score_str = format_value(r["score"], "{:.3f}")
        belongs = "Yes" if r in top3_set else "No"
        print(
            f"Model: {model}, Score: {score_str} -> {belongs}"
        )

    # -----------------------------
    # CAR-08: up‑to‑date inventory summary
    # -----------------------------
    print("\n== CAR-08 ==")

    # Inventory statistics
    stats = compute_stats(rows)
    print(
        f"Weight: mean={format_value(stats['weight']['mean'])} lbs, "
        f"min={format_value(stats['weight']['min'])} lbs, "
        f"max={format_value(stats['weight']['max'])} lbs"
    )
    print(
        f"Fuel efficiency (mpg): mean={format_value(stats['mpg']['mean'])}, "
        f"min={format_value(stats['mpg']['min'])}, "
        f"max={format_value(stats['mpg']['max'])}"
    )
    print(
        f"Acceleration time (s): mean={format_value(stats['acc']['mean'])}, "
        f"min={format_value(stats['acc']['min'])}, "
        f"max={format_value(stats['acc']['max'])}"
    )

    # Recommended shortlist
    print(f"\nTop {top_n} vehicles with balanced performance:")
    for idx, r in enumerate(top3, start=1):
        print(
            f"{idx}. Model: {r['Model']}, "
            f"Weight: {format_value(r['Lbs-'])} lbs, "
            f"MPG: {format_value(r['Mpg+'])}, "
            f"Acc: {format_value(r['Acc+'])} s, "
            f"Score: {format_value(r['score'], '{:.3f}')}"
        )


if __name__ == "__main__":
    main()
