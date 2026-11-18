#!/usr/bin/env python3
"""
Entry point for CAR-13 and CAR-14 skills.

CAR-13: Identify a vehicle with a strong balance of fuel efficiency (Mpg+),
weight (Lbs-), and acceleration (Acc+).  Evaluate as few candidate vehicles
as possible (at most 30).  Evaluation means looking at the goal values of
the row.

CAR-14: Recommend an order of up to 30 vehicles to evaluate first so that
promising inventory options are discovered early.  The order is chosen
using only the goal values of vehicles already placed earlier in the
order – i.e. the order itself is decided beforehand and does not depend
on the results of the evaluation step.

The program reads the CSV file whose path is the sole command‑line
argument.  It prints, for each request, a section headed by
`== <ID> ==` and a machine‑checked answer line:

    ANSWER <ID> <JSON>

The JSON objects are described in the documentation above.
"""

import sys
import json
from typing import List, Dict, Tuple

# The ezr library is available in the same directory.
from ezr import csv  # only need csv to parse the file


# ----------------------------------------------------------------------
# Utility functions reused from previous skills
# ----------------------------------------------------------------------
def is_goal(col_name: str) -> bool:
    """Return True if column is a goal (ends with + or -)."""
    return col_name[-1] in "+-"


def goal_indices(header: List[str]) -> List[int]:
    """Return list of indices for goal columns."""
    return [i for i, name in enumerate(header) if is_goal(name)]


def compute_norm_limits(rows: List[List], goal_idxs: List[int]) -> Tuple[List[float], List[float]]:
    """Return (mins, maxs) for each goal column using only rows that have a value."""
    mins = [float("inf")] * len(goal_idxs)
    maxs = [float("-inf")] * len(goal_idxs)
    for row in rows:
        for j, gi in enumerate(goal_idxs):
            val = row[gi]
            if val == "?":
                continue
            f = float(val)
            if f < mins[j]:
                mins[j] = f
            if f > maxs[j]:
                maxs[j] = f
    # If a column has no values, set min=max=0
    for j in range(len(goal_idxs)):
        if mins[j] == float("inf"):
            mins[j] = maxs[j] = 0.0
    return mins, maxs


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Return normalized value in [0,1]. Handle zero range."""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


def compute_score(row: List, goal_idxs: List[int],
                  mins: List[float], maxs: List[float]) -> float:
    """Compute overall recommendation score for a row.  Higher is better."""
    score = 0.0
    for j, gi in enumerate(goal_idxs):
        val = row[gi]
        if val == "?":
            continue  # skip missing goals
        f = float(val)
        norm = normalize(f, mins[j], maxs[j])
        if row[gi][-1] == "+":
            score += norm
        else:  # "-"
            score += 1 - norm
    return score


# ----------------------------------------------------------------------
# CAR-13 implementation
# ----------------------------------------------------------------------
def car_13(header: List[str], data_rows: List[List]) -> None:
    goal_idxs = goal_indices(header)
    n_to_eval = min(30, len(data_rows))

    # Evaluate first n_to_eval rows (row numbers 1..n_to_eval)
    evaluated_rows = list(range(1, n_to_eval + 1))

    # Compute norms based on evaluated rows only
    eval_subset = [data_rows[i - 1] for i in evaluated_rows]
    mins, maxs = compute_norm_limits(eval_subset, goal_idxs)

    # Compute scores for each evaluated row
    scores = {}
    for rn in evaluated_rows:
        row = data_rows[rn - 1]
        scores[rn] = compute_score(row, goal_idxs, mins, maxs)

    # Best row: highest score
    best_row = max(scores.items(), key=lambda kv: kv[1])[0]

    # Print section
    print("== CAR-13 ==")
    print(f"Evaluated rows: {evaluated_rows}")
    print(f"Best row: {best_row}")

    # Answer line
    answer = {
        "evaluated": evaluated_rows,
        "best": best_row
    }
    print("ANSWER CAR-13", json.dumps(answer, separators=(',', ':')))


# ----------------------------------------------------------------------
# CAR-14 implementation
# ----------------------------------------------------------------------
def car_14(header: List[str], data_rows: List[List]) -> None:
    goal_idxs = goal_indices(header)
    total_rows = len(data_rows)
    order_size = min(30, total_rows)

    # Compute global norms across all rows
    mins, maxs = compute_norm_limits(data_rows, goal_idxs)

    # Compute scores for every row (skip missing goals)
    scores = []
    for rn, row in enumerate(data_rows, start=1):
        # If a row has any missing goal, we simply compute with available ones
        s = compute_score(row, goal_idxs, mins, maxs)
        scores.append((s, rn))

    # Sort by descending score, higher is better
    scores.sort(key=lambda kv: kv[0], reverse=True)
    order = [rn for _, rn in scores[:order_size]]

    # Print section
    print("== CAR-14 ==")
    print(f"Order to evaluate: {order}")

    # Answer line
    answer = {"order": order}
    print("ANSWER CAR-14", json.dumps(answer, separators=(',', ':')))


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    rows = [row for row in csv(path)]
    if not rows:
        print("No data found.")
        sys.exit(1)

    header = rows[0]
    data_rows = rows[1:]

    car_13(header, data_rows)
    print()
    car_14(header, data_rows)


if __name__ == "__main__":
    main()
