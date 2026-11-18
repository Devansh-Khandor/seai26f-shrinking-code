#!/usr/bin/env python3
"""
skill_04.py: Inventory assessment and live summary for auto93 dataset.
Usage: python skill_04.py <csv_path>
"""

import json
import sys
from typing import List, Dict, Any

# Reuse helpers from the original skill_01
from skill_01 import read_csv, build_table, normalize  # type: ignore

# ------------------------------------------------------------------
# Helper: combined normalized score for a row
def _row_score(tbl, row):
    """Return the combined score of a row, based on all goal columns."""
    goal_cols = [(col.at, col) for col in tbl.cols.values() if col.goal != 0]
    # Precompute min/max for each goal column
    stats = {}
    for at, col in goal_cols:
        vals = [r[at] for r in tbl.rows if r[at] != "?"]
        if vals:
            mn, mx = min(vals), max(vals)
            stats[at] = {"min": mn, "max": mx, "goal": col.goal}
        else:
            stats[at] = {"min": 0.0, "max": 0.0, "goal": col.goal}
    score = 0.0
    for at, col in goal_cols:
        val = row[at]
        if val == "?":
            continue  # skip missing goal for this component
        mn = stats[at]["min"]
        mx = stats[at]["max"]
        goal = stats[at]["goal"]
        if mx == mn:
            comp = 1.0
        else:
            if goal == 1:  # maximize
                comp = (val - mn) / (mx - mn)
            else:          # minimize
                comp = (mx - val) / (mx - mn)
        score += comp
    return score

# ------------------------------------------------------------------
# CAR-07: Assess arrivals
def _car_07(tbl, rows, arrivals, arrival_start):
    """Return list of assessments for each arrival."""
    assessments = []
    history_rows = rows[:arrival_start]  # before arrivals
    for i, row in enumerate(arrivals, start=1):
        # current inventory includes history + earlier arrivals + this one
        current_rows = history_rows + arrivals[:i]
        # compute mean score of current inventory
        scores = [_row_score(tbl, r) for r in current_rows]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        # score for this arrival
        arr_score = _row_score(tbl, row)
        desirable = arr_score > mean_score
        assessments.append({"row": arrival_start + i, "desirable": desirable})
    return assessments

# ------------------------------------------------------------------
# CAR-08: Live summary checkpoints
def _summary_for_rows(tbl, rows_subset):
    """Return summary dict for goal columns given a list of rows."""
    goal_cols = [(col.at, col) for col in tbl.cols.values() if col.goal != 0]
    summary = {}
    for at, col in goal_cols:
        vals = [r[at] for r in rows_subset if r[at] != "?"]
        n = len(vals)
        if n:
            mn = min(vals)
            mx = max(vals)
            mean = sum(vals) / n
        else:
            mn = mx = mean = 0.0
        summary[col.txt] = {"n": n, "mean": mean, "min": mn, "max": mx}
    return summary

def _shortlist_for_rows(tbl, rows_subset, count=5):
    """Return list of row numbers of top `count` rows by score."""
    scored = [(_row_score(tbl, r), idx+1) for idx, r in enumerate(rows_subset)]
    # sort by score descending, then row number ascending
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [row_no for _, row_no in scored[:count]]

def _car_08(tbl, rows, arrivals, arrival_start, checkpoints=[0, 40, 80]):
    """Return checkpoints with summaries and shortlist."""
    checkpoints_out = []
    for m in checkpoints:
        # include first m arrivals
        subset = rows[:arrival_start + m]
        summary = _summary_for_rows(tbl, subset)
        shortlist = _shortlist_for_rows(tbl, subset, 5)
        checkpoints_out.append({
            "after": m,
            "summary": summary,
            "shortlist": shortlist
        })
    return checkpoints_out

# ------------------------------------------------------------------
# Main driver
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    header, rows = read_csv(path)
    tbl = build_table(header, rows)

    total_rows = len(rows)
    arrivals_count = round(0.2 * total_rows)
    if arrivals_count == 0:
        arrivals_count = 1
    arrival_start = total_rows - arrivals_count
    arrivals = rows[arrival_start:]

    # CAR-07
    assessments = _car_07(tbl, rows, arrivals, arrival_start)
    print("== CAR-07 ==")
    for a in assessments:
        print(f"Row {a['row']}: desirable={a['desirable']}")
    print(f"ANSWER CAR-07 {json.dumps({'assessments': assessments})}")

    # CAR-08
    checkpoints = _car_08(tbl, rows, arrivals, arrival_start, checkpoints=[0, 40, 80])
    print("\n== CAR-08 ==")
    for cp in checkpoints:
        print(f"Checkpoint after {cp['after']} arrivals:")
        for col, stats in cp['summary'].items():
            print(f"  {col}: n={stats['n']} mean={stats['mean']} min={stats['min']} max={stats['max']}")
        print(f"  Shortlist: {cp['shortlist']}")
    print(f"ANSWER CAR-08 {json.dumps({'checkpoints': checkpoints})}")

if __name__ == "__main__":
    main()
