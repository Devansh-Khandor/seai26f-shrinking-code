#!/usr/bin/env python3
"""
Entry point for CAR-17 and CAR-18 skills.

CAR-17: Flag vehicle records whose combinations of characteristics are
unusually different from the rest of our inventory so that staff can
review them.

CAR-18: Identify vehicles whose reported fuel efficiency (Mpg+) or weight
(Lbs-) differs substantially from what similar vehicles would lead us to
expect, so we can check for possible recording mistakes.

The script prints a section headed by `== <ID> ==` for each request and
a machine‑checked answer line `ANSWER <ID> <JSON>`.  The JSON objects
follow the format described in the problem statement.
"""

import json
import math
from typing import List, Tuple

# The ezr library is available in the same directory.
from ezr import csv

# Reuse helper functions defined in skill_08 for input normalization
# and distance calculation.  They are not exported but available in
# the module namespace.
from skill_08 import (
    compute_input_norm_limits,
    distance_inputs,
    input_indices,
)

# ----------------------------------------------------------------------
#  Utility helpers ---------------------------------------------------
# ----------------------------------------------------------------------
def is_goal(col_name: str) -> bool:
    return col_name[-1] in "+-"


def goal_indices(header: List[str]) -> List[int]:
    return [i for i, name in enumerate(header) if is_goal(name)]


def goal_names(header: List[str]) -> List[str]:
    return [header[i] for i in goal_indices(header)]


def parse_value(val: str) -> float | None:
    """Convert a numeric string to float; return None for '?'."""
    return None if val == "?" else float(val)


# ----------------------------------------------------------------------
#  CAR-17: Unusual input combinations -------------------------------
# ----------------------------------------------------------------------
def car_17(header: List[str], data_rows: List[List[str]]) -> None:
    """
    For each row compute the average distance to its k nearest neighbors
    in the space of non‑goal input columns.  Flag the rows with the
    highest such distances (up to 40).
    """
    inp_idxs = input_indices(header)  # indices of non‑goal inputs
    if not inp_idxs:
        print("== CAR-17 ==")
        print("No input columns to compare.")
        print("ANSWER CAR-17 {\"flagged\": [], \"scores\": {}}")
        return

    mins, maxs = compute_input_norm_limits(data_rows, inp_idxs)

    # Pre‑compute pairwise distances
    n_rows = len(data_rows)
    distances = [[0.0] * n_rows for _ in range(n_rows)]
    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            d = distance_inputs(data_rows[i], data_rows[j], mins, maxs, inp_idxs)
            distances[i][j] = distances[j][i] = d

    # For each row, average distance to its 3 nearest neighbors
    k = 3
    avg_dist = []
    for i in range(n_rows):
        neigh_dist = sorted(distances[i][j] for j in range(n_rows) if j != i)[:k]
        avg = sum(neigh_dist) / len(neigh_dist) if neigh_dist else 0.0
        avg_dist.append(avg)

    # Determine top 40 rows
    sorted_rows = sorted(
        enumerate(avg_dist, start=1), key=lambda t: t[1], reverse=True
    )
    flagged_rows = [rn for rn, _ in sorted_rows[:40] if rn <= n_rows]

    # Build JSON
    scores_dict = {str(rn): round(avg_dist[rn - 1], 6) for rn in flagged_rows}

    # Print section
    print("== CAR-17 ==")
    for rn in flagged_rows:
        print(f"Row {rn}: unusualness score {scores_dict[str(rn)]}")
    # Answer line
    answer = {"flagged": flagged_rows, "scores": scores_dict}
    print("ANSWER CAR-17", json.dumps(answer, separators=(',', ':')))


# ----------------------------------------------------------------------
#  CAR-18: Unexpected goal values -----------------------------------
# ----------------------------------------------------------------------
def car_18(header: List[str], data_rows: List[List[str]]) -> None:
    """
    For each row, look at the k nearest neighbors in input space and
    compute the average value of each goal column (Mpg+, Lbs-).
    If the absolute difference between the actual value and the
    neighbor average exceeds mean + 2*sd of the differences, flag it.
    """
    goal_idxs = goal_indices(header)
    if not goal_idxs:
        print("== CAR-18 ==")
        print("No goal columns to analyze.")
        print("ANSWER CAR-18 []")
        return

    inp_idxs = input_indices(header)
    mins, maxs = compute_input_norm_limits(data_rows, inp_idxs)

    n_rows = len(data_rows)
    # Pre‑compute pairwise distances (same as in CAR-17)
    distances = [[0.0] * n_rows for _ in range(n_rows)]
    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            d = distance_inputs(data_rows[i], data_rows[j], mins, maxs, inp_idxs)
            distances[i][j] = distances[j][i] = d

    k = 3
    # For each row compute diffs per goal
    diff_stats: dict[int, dict[str, List[float]]] = {}
    for i in range(n_rows):
        # Find k nearest neighbors
        neigh = sorted(
            [(distances[i][j], j) for j in range(n_rows) if j != i]
        )[:k]
        # For each goal compute average among neighbors
        diffs = {}
        for gi in goal_idxs:
            actual_val = parse_value(data_rows[i][gi])
            if actual_val is None:
                continue
            neigh_vals = [
                parse_value(data_rows[j][gi]) for _, j in neigh
                if parse_value(data_rows[j][gi]) is not None
            ]
            if not neigh_vals:
                continue
            avg_neigh = sum(neigh_vals) / len(neigh_vals)
            diffs[header[gi]] = abs(actual_val - avg_neigh)
        diff_stats[i] = diffs

    # Compute mean and std per goal
    per_goal_diffs: dict[str, List[float]] = {g: [] for g in goal_names(header)}
    for diffs in diff_stats.values():
        for g, d in diffs.items():
            per_goal_diffs[g].append(d)

    mean_sd: dict[str, Tuple[float, float]] = {}
    for g, lst in per_goal_diffs.items():
        if lst:
            m = sum(lst) / len(lst)
            s = math.sqrt(sum((x - m) ** 2 for x in lst) / len(lst))
            mean_sd[g] = (m, s)
        else:
            mean_sd[g] = (0.0, 0.0)

    # Flag rows where diff > mean + 2*sd
    flagged_entries = []
    for i, diffs in diff_stats.items():
        for g, d in diffs.items():
            m, s = mean_sd[g]
            if d > m + 2 * s:
                flagged_entries.append(
                    {
                        "row": i + 1,
                        "column": g,
                        "actual": round(parse_value(data_rows[i][goal_indices(header)[goal_names(header).index(g)]]), 6),
                        "expected": round(
                            sum(
                                parse_value(data_rows[j][goal_indices(header)[goal_names(header).index(g)])
                                for _, j in sorted(
                                    [(distances[i][j], j) for j in range(n_rows) if j != i
                                    ],
                                    key=lambda t: t[0],
                                )[:k]
                            )
                            / len(
                                [1 for _, j in sorted(
                                    [(distances[i][j], j) for j in range(n_rows) if j != i
                                    ],
                                    key=lambda t: t[0],
                                )[:k]
                                    if parse_value(data_rows[j][goal_indices(header)[goal_names(header).index(g)]) is not None]
                            ),
                            6,
                        ),
                    }
                )
    # Limit to first 40 entries
    flagged_entries = flagged_entries[:40]

    # Print section
    print("== CAR-18 ==")
    for e in flagged_entries:
        print(f"Row {e['row']} - {e['column']} actual {e['actual']} expected {e['expected']}")
    # Answer line
    answer = flagged_entries
    print("ANSWER CAR-18", json.dumps(answer, separators=(',', ':')))


# ----------------------------------------------------------------------
#  Main entry point ---------------------------------------------------
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    rows = [row for row in csv(path)]
    if not rows:
        print("No data found.")
        sys.exit(1)

    header = rows[0]
    data_rows = rows[1:]

    car_17(header, data_rows)
    print()
    car_18(header, data_rows)


if __name__ == "__main__":
    main()
