#!/usr/bin/env python3
"""
Skill 06: Recommendation justification and trade‑off analysis

Usage:
    python skill_06.py <path to auto93.csv>
"""

import json
import sys
from skill_01 import read_csv
from skill_04 import GOAL_COLUMNS, compute_means_stds, compute_score

def main(csv_path):
    rows = read_csv(csv_path)
    n_rows = len(rows)

    # Compute overall mean & std for goal columns
    means, stds = compute_means_stds(rows, list(range(n_rows)))

    # Compute scores for all rows that have all goal values
    scored = []
    for idx, row in enumerate(rows, start=1):
        # Must have all goal columns
        if any(row.get(col) is None for col in GOAL_COLUMNS):
            continue
        score = compute_score(row, means, stds)
        if score is None:
            continue
        scored.append((score, idx, row))

    if len(scored) == 0:
        print("== CAR-11 ==")
        print("No complete records available.")
        print("ANSWER CAR-11 {}")
        print("\n== CAR-12 ==")
        print("No complete records available.")
        print("ANSWER CAR-12 {}")
        return

    # Sort by score descending
    scored.sort(key=lambda x: -x[0])

    # CAR-11: recommendation & neighbors
    best_score, best_idx, best_row = scored[0]
    recommended_row_no = best_idx

    # Find nearest neighbors (excluding the recommended itself)
    def dist(row1, row2):
        d = 0.0
        for col in GOAL_COLUMNS:
            d += abs(row1[col] - row2[col])
        return d

    neighbors = []
    for score, idx, row in scored[1:]:
        neighbors.append((dist(best_row, row), idx, row))
    neighbors.sort(key=lambda x: x[0])

    neighbor_entries = []
    for _, idx, row in neighbors[:3]:
        entry = {
            "row": idx,
            "values": {col: row[col] for col in GOAL_COLUMNS}
        }
        neighbor_entries.append(entry)

    print("== CAR-11 ==")
    print(f"Recommended row: {recommended_row_no}")
    for i, nb in enumerate(neighbor_entries, start=1):
        print(f"Neighbor {i}: row {nb['row']} values: {nb['values']}")
    print("ANSWER CAR-11 " + json.dumps({
        "recommended": recommended_row_no,
        "neighbors": neighbor_entries
    }))

    # CAR-12: compare top two vehicles
    if len(scored) < 2:
        print("\n== CAR-12 ==")
        print("Not enough complete records for comparison.")
        print("ANSWER CAR-12 {}")
        return

    a_score, a_idx, a_row = scored[0]
    b_score, b_idx, b_row = scored[1]

    differences = []
    tradeoffs = []
    for col in GOAL_COLUMNS:
        val_a = a_row[col]
        val_b = b_row[col]
        differences.append({"feature": col, "a": val_a, "b": val_b})
        # Determine which side is better for this goal
        if col.endswith("-"):   # minimize
            if val_b < val_a:
                tradeoffs.append(col)
        else:                  # maximize
            if val_b > val_a:
                tradeoffs.append(col)

    print("\n== CAR-12 ==")
    print(f"Vehicle A: row {a_idx}")
    print(f"Vehicle B: row {b_idx}")
    print(f"Preferred: {'a' if a_score >= b_score else 'b'}")
    print(f"Scores: a={a_score:.3f} b={b_score:.3f}")
    print("Differences:")
    for d in differences:
        print(f"  {d['feature']}: a={d['a']} b={d['b']}")
    print("Tradeoffs where non‑preferred is better:")
    for col in tradeoffs:
        print(f"  {col}")

    print("ANSWER CAR-12 " + json.dumps({
        "a": a_idx,
        "b": b_idx,
        "preferred": "a" if a_score >= b_score else "b",
        "scores": {"a": a_score, "b": b_score},
        "differences": differences,
        "tradeoffs": tradeoffs
    }))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
