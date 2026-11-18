#!/usr/bin/env python3
"""
Entry point for CAR-07 and CAR-08 skills.

CAR-07: As new vehicle records arrive, assess each one immediately and tell me whether it belongs among our more desirable inventory options.
CAR-08: Keep our inventory summaries and recommended‑vehicle shortlist up to date as additional vehicle records are received, without requiring staff to restart the entire analysis manually.
"""

import sys
import json
from collections import defaultdict
from math import sqrt

# The ezr library is available in the same directory.
from ezr import csv

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def is_goal(col_name: str) -> bool:
    return col_name[-1] in "+-"

def goal_indices(header):
    """Return list of indices for goal columns."""
    return [i for i, name in enumerate(header) if is_goal(name)]

def goal_names(header):
    return [header[i] for i in goal_indices(header)]

# Rank rows according to combined ranking of the goal columns.
# Returns list of row numbers (1‑based) sorted by rank_sum (best first).
def rank_rows(rows, goal_idxs):
    # Build mapping from goal to (value, row_number)
    goal_vals = defaultdict(list)  # goal_idx -> list of (value, row_number)
    row_has_all = set()  # rows that have all goals present
    for rn, r in enumerate(rows, start=1):
        # Check presence
        if any(r[i] == "?" for i in goal_idxs):
            continue
        row_has_all.add(rn)
        for i in goal_idxs:
            goal_vals[i].append((float(r[i]), rn))

    # Assign ranks for each goal
    goal_rank = defaultdict(dict)  # goal_idx -> {row_number: rank}
    for i in goal_idxs:
        vals = goal_vals[i]
        # Determine ordering: maximize for '+', minimize for '-'
        reverse = header[i][-1] == "+"
        # Sort accordingly
        sorted_vals = sorted(vals, key=lambda x: x[0], reverse=reverse)
        # Assign rank positions (1‑based)
        for rank, (_, rn) in enumerate(sorted_vals, start=1):
            goal_rank[i][rn] = rank

    # Compute rank_sum for rows with all goals
    rank_sums = {}
    for rn in row_has_all:
        rank_sums[rn] = sum(goal_rank[i][rn] for i in goal_idxs)

    # Sort rows by rank_sum ascending
    sorted_rows = sorted(rank_sums.items(), key=lambda x: x[1])
    return [rn for rn, _ in sorted_rows]

# Compute stats for each goal column
def goal_stats(rows, goal_idxs):
    stats = {}
    for i in goal_idxs:
        values = [float(r[i]) for r in rows if r[i] != "?"]
        n = len(values)
        if n == 0:
            mean = min_val = max_val = float("nan")
        else:
            mean = round(sum(values) / n, 6)
            min_val = round(min(values), 6)
            max_val = round(max(values), 6)
        stats[header[i]] = {"n": n, "mean": mean, "min": min_val, "max": max_val}
    return stats

# ----------------------------------------------------------------------
# Main logic for CAR-07 and CAR-08
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    # Load CSV
    rows_all = [row for row in csv(path)]
    if not rows_all:
        print("No data found.")
        sys.exit(1)

    global header
    header = rows_all[0]
    data_rows = rows_all[1:]          # list of rows (list of ATOMs)
    total_rows = len(data_rows)

    # Identify goal columns
    goal_idxs = goal_indices(header)

    # Split into history + arrivals
    n_arrivals = round(0.2 * total_rows)
    arrivals = data_rows[-n_arrivals:]
    history = data_rows[:total_rows - n_arrivals]

    # CAR-07: Assess each arrival
    print("== CAR-07 ==")
    seen_rows = history[:]          # copy of history
    assessments = []
    for rn, arrival in enumerate(arrivals, start=1):
        # row number in original data (1-based)
        row_num = (total_rows - n_arrivals) + rn
        seen_rows.append(arrival)  # include this arrival in seen set
        # Compute ranking of all seen rows
        top5 = rank_rows(seen_rows, goal_idxs)[:5]
        desirable = row_num in top5
        assessments.append({"row": row_num, "desirable": desirable})
        print(f"Row {row_num}: desirable {desirable}")
    # Answer line
    print("ANSWER CAR-07", json.dumps({"assessments": assessments}, separators=(',', ':')))

    # CAR-08: Checkpoints and shortlist
    print("\n== CAR-08 ==")
    checkpoints = []
    for m in [0, 40, 80]:
        # Determine seen rows up to this checkpoint
        seen = history + arrivals[:m]
        after = m  # arrivals processed so far
        # Stats per goal
        summary = goal_stats(seen, goal_idxs)
        # Shortlist: top 5 rows
        top5_rows = rank_rows(seen, goal_idxs)[:5]
        checkpoints.append({"after": after, "summary": summary, "shortlist": top5_rows})
        # Print checkpoint info
        print(f"Checkpoint after {after} arrivals:")
        for g in goal_names(header):
            s = summary[g]
            print(f"  {g}: n={s['n']} mean={s['mean']} min={s['min']} max={s['max']}")
        print(f"  Shortlist: {top5_rows}")
    # Answer line
    print("ANSWER CAR-08", json.dumps({"checkpoints": checkpoints}, separators=(',', ':')))

if __name__ == "__main__":
    main()
