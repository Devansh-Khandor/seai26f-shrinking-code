#!/usr/bin/env python3
"""
Small data‑analysis skill for the auto93 dataset (CAR‑13 and CAR‑14).

Usage:
    python skill_07.py <path_to_csv>
"""

import sys
import json
import numpy as np

# Shared utilities
from skill_01 import load_data, GOAL_COLUMNS, compute_overall_top10


# --------------------------------------------------------------------
# CAR-13: Find a balanced vehicle with few evaluations
# --------------------------------------------------------------------
def evaluate_balanced_vehicle(data, budget=10):
    """
    Evaluate at most ``budget`` rows (or fewer if not enough rows with full data)
    and return the row number with the best balance of weight, acceleration, and
    fuel‑efficiency.  The balance score is simply:
        -Lbs- + Acc+ + Mpg+
    Missing goal values cause the row to be skipped.
    """
    evaluated = []
    best_row = None
    best_score = -float("inf")

    for row in data:
        if len(evaluated) >= budget:
            break
        vals = row["values"]
        # Skip rows with any missing goal value
        if any(vals[col] is None for col in GOAL_COLUMNS):
            continue
        # Compute simple linear score
        score = -vals["Lbs-"] + vals["Acc+"] + vals["Mpg+"]
        evaluated.append(row["row"])
        if score > best_score:
            best_score = score
            best_row = row["row"]

    return evaluated, best_row


# --------------------------------------------------------------------
# CAR-14: Recommend evaluation order with limited budget
# --------------------------------------------------------------------
def recommend_evaluation_order(data, budget=None):
    """
    Recommend the order in which vehicles should be inspected.
    Uses the overall score from skill_01.compute_overall_top10, sorts descending,
    and returns the top ``budget`` rows.
    """
    # Compute overall scores for all rows with full data
    all_scores = compute_overall_top10(data)

    if not all_scores:
        return []

    # Sort by score descending
    all_scores.sort(key=lambda x: x[1], reverse=True)
    # Default budget: min(30, total_rows // 5)
    if budget is None:
        budget = min(30, len(data) // 5)
    order = [row_num for row_num, _ in all_scores[:budget]]
    return order


# --------------------------------------------------------------------
# Main logic
# --------------------------------------------------------------------
def main(csv_path):
    data = load_data(csv_path)

    # ----- CAR-13 --------------------------------------------------
    print("== CAR-13 ==")
    evaluated_rows, best_row = evaluate_balanced_vehicle(data, budget=10)

    # Some simple output
    print(f"Evaluated rows (up to 10): {evaluated_rows}")
    print(f"Best balanced vehicle: row {best_row}")

    # Machine‑checked answer
    answer_json = json.dumps({"evaluated": evaluated_rows, "best": best_row}, separators=(",", ":"))
    print(f"ANSWER CAR-13 {answer_json}")

    # ----- CAR-14 --------------------------------------------------
    print("\n== CAR-14 ==")
    order = recommend_evaluation_order(data)

    print(f"Recommended evaluation order (first {len(order)} rows): {order[:5]} ...")
    print(f"Total rows to evaluate: {len(order)}")

    # Machine‑checked answer
    answer_json = json.dumps({"order": order}, separators=(",", ":"))
    print(f"ANSWER CAR-14 {answer_json}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
