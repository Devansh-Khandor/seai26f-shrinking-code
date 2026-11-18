#!/usr/bin/env python3
"""
Skill 07 – Targeted vehicle evaluation and inspection order.

Usage:
    python skill_07.py <csv_path>
"""

import csv
import json
import sys
from typing import Dict, List, Tuple

# Reuse helpers from earlier skills
from skill_04 import parse_float, normalize
from skill_05 import compute_overall_scores

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
GOAL_COLS = ["Lbs-", "Acc+", "Mpg+"]
GOAL_DIRECTIONS = {"Lbs-": False, "Acc+": True, "Mpg+": True}

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(csv_path: str) -> None:
    # Read CSV
    data: List[Dict[str, float | None]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed_row = {k: parse_float(v) for k, v in row.items()}
            data.append(parsed_row)

    # ------------------------------------------------------------------ #
    # Compute overall scores for all rows
    # ------------------------------------------------------------------ #
    overall_scores = compute_overall_scores(data, GOAL_COLS, GOAL_DIRECTIONS)
    if not overall_scores:
        # No rows with complete goals
        print("== CAR-13 ==")
        print("No complete rows available.")
        print("ANSWER CAR-13 {}")
        print("\n== CAR-14 ==")
        print("No complete rows available.")
        print("ANSWER CAR-14 {}")
        return

    # Map row number to score for quick lookup
    score_map = {row_num: score for row_num, score in overall_scores}

    # ------------------------------------------------------------------ #
    # CAR-13: Identify balanced vehicle with minimal evaluations
    # ------------------------------------------------------------------ #
    # Determine global best score (maximum)
    best_row_global = max(score_map, key=score_map.get)
    global_best_score = score_map[best_row_global]

    evaluated: List[int] = []
    best_row = None
    best_score = float("-inf")

    # Sequentially evaluate rows until we hit 90% of global best
    for idx, row in enumerate(data, start=1):
        if any(row[col] is None for col in GOAL_COLS):
            continue  # cannot evaluate without all goals
        evaluated.append(idx)

        cur_score = score_map[idx]
        if cur_score > best_score:
            best_score = cur_score
            best_row = idx

        if cur_score >= 0.9 * global_best_score:
            break

    print("== CAR-13 ==")
    print("Evaluated rows:", evaluated)
    print("Best vehicle row:", best_row)
    ans_car_13 = json.dumps(
        {"evaluated": evaluated, "best": best_row},
        separators=(",", ":"),
    )
    print(f"ANSWER CAR-13 {ans_car_13}")

    # ------------------------------------------------------------------ #
    # CAR-14: Recommended inspection order under budget
    # ------------------------------------------------------------------ #
    # Budget: at most min(30, number_of_rows // 5)
    budget = min(30, len(data) // 5)

    # Order rows by descending overall score (tie by row number)
    sorted_overall = sorted(overall_scores, key=lambda x: (-x[1], x[0]))
    order_list = [row_num for row_num, _ in sorted_overall[:budget]]

    print("\n== CAR-14 ==")
    print("Recommended order:", order_list)
    ans_car_14 = json.dumps(
        {"order": order_list},
        separators=(",", ":"),
    )
    print(f"ANSWER CAR-14 {ans_car_14}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
