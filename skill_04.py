#!/usr/bin/env python3
"""
Skill 04: Real‑time assessment and incremental inventory summaries for the auto93 dataset.

Requirements
------------
CAR-07  - For every new vehicle record (arrival) evaluate whether it is
          among the 10 best vehicles seen so far based on normalized goal
          metrics.  The evaluation must use only the history and previously
          processed arrivals (no look‑ahead).

CAR-08  - Keep the summary statistics for every goal column and a 5‑vehicle
          shortlist up to date after 0, 40 and 80 arrivals.  The summary
          is based on the history plus the processed arrivals for that
          checkpoint.  The shortlist contains the row numbers of the
          5 best vehicles according to the same normalized metric used
          in CAR‑07.

The script reads the CSV file supplied as the single command line
argument and produces the required output and machine‑checked JSON
answers.
"""

import csv
import json
import sys
import math
from collections import defaultdict

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def parse_value(val: str):
    """Convert a string from the CSV to a float or None."""
    if val == '?' or val.strip() == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None


def is_goal(col_name: str) -> bool:
    """Return True for columns that end with '+' or '-' and do not end with 'X'."""
    return (col_name.endswith('+') or col_name.endswith('-')) and not col_name.endswith('X')


def is_maximize(col_name: str) -> bool:
    return col_name.endswith('+')


# ----------------------------------------------------------------------
# CSV I/O
# ----------------------------------------------------------------------
def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows, reader.fieldnames


# ----------------------------------------------------------------------
# Statistics utilities
# ----------------------------------------------------------------------
def compute_minmax(rows, goal_cols):
    mins = {}
    maxs = {}
    for col in goal_cols:
        vals = [parse_value(r[col]) for r in rows if parse_value(r[col]) is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
    return mins, maxs


def compute_score(row, minsmax, goal_cols):
    mins, maxs = minsmax
    score = 0.0
    for col in goal_cols:
        v = parse_value(row[col])
        if v is None:
            return None  # cannot score if any goal is missing
        mn = mins[col]
        mx = maxs[col]
        if mx == mn:
            norm = 1.0
        else:
            if is_maximize(col):
                norm = (v - mn) / (mx - mn)
            else:
                norm = (mx - v) / (mx - mn)
        score += norm
    return score


def compute_summary(rows, goal_cols):
    """Return a dict {col: {'n': int, 'mean': float, 'min': float, 'max': float}}"""
    summary = {}
    for col in goal_cols:
        vals = [parse_value(r[col]) for r in rows if parse_value(r[col]) is not None]
        n = len(vals)
        if n == 0:
            mean = min_val = max_val = 0.0
        else:
            mean = sum(vals) / n
            min_val = min(vals)
            max_val = max(vals)
        summary[col] = {"n": n, "mean": mean, "min": min_val, "max": max_val}
    return summary


# ----------------------------------------------------------------------
# Core logic
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    rows, fieldnames = read_csv(path)

    goal_cols = [c for c in fieldnames if is_goal(c)]
    total_rows = len(rows)
    # Determine arrival and history split
    arrivals_count = round(0.2 * total_rows)
    history_count = total_rows - arrivals_count
    history_rows = rows[:history_count]
    arrival_rows = rows[history_count:]

    # ---------- CAR-07 ----------
    print("== CAR-07 ==")

    assessments = []
    seen_rows = history_rows[:]  # rows processed so far
    # Map from row index (1‑based) to row data for quick lookup
    row_index_map = {i + 1: r for i, r in enumerate(rows)}

    for idx, arrival in enumerate(arrival_rows):
        arrival_index = history_count + idx + 1  # 1‑based row number

        # Compute min/max from data seen so far (history + earlier arrivals)
        mins, maxs = compute_minmax(seen_rows, goal_cols)
        current_minmax = (mins, maxs)

        # Score for the current arrival
        score_cur = compute_score(arrival, current_minmax, goal_cols)

        # If the arrival has missing goal values, mark as not desirable
        if score_cur is None:
            desirable = False
        else:
            # Scores of all previously seen rows
            scores_prev = []
            for r in seen_rows:
                s = compute_score(r, current_minmax, goal_cols)
                if s is not None:
                    rownum = row_index_map[(rows.index(r) + 1)]
                    scores_prev.append((rownum, s))
            # Add the current arrival
            scores_all = scores_prev + [(arrival_index, score_cur)]
            # Sort descending score, then ascending row number
            sorted_scores = sorted(scores_all, key=lambda x: (-x[1], x[0]))
            # Determine rank of the current arrival
            rank = next(i for i, (rnum, _) in enumerate(sorted_scores, start=1) if rnum == arrival_index)
            desirable = rank <= 10

        assessments.append({"row": arrival_index, "desirable": desirable})
        print(f"Row {arrival_index}: desirable={desirable}")

        # Update seen_rows for next arrival
        seen_rows.append(arrival)

    # JSON answer
    print(f"ANSWER CAR-07 {json.dumps({'assessments': assessments}, ensure_ascii=False)}")

    # ---------- CAR-08 ----------
    print("\n== CAR-08 ==")

    checkpoints = []
    for m in [0, 40, 80]:
        m = min(m, arrivals_count)  # cannot exceed available arrivals
        checkpoint_rows = history_rows + arrival_rows[:m]
        summary = compute_summary(checkpoint_rows, goal_cols)

        # Compute shortlist
        mins, maxs = compute_minmax(checkpoint_rows, goal_cols)
        scores = []
        for r in checkpoint_rows:
            s = compute_score(r, (mins, maxs), goal_cols)
            if s is not None:
                rnum = row_index_map[(rows.index(r) + 1)]
                scores.append((rnum, s))
        scores_sorted = sorted(scores, key=lambda x: (-x[1], x[0]))
        shortlist = [rnum for rnum, _ in scores_sorted[:5]]

        checkpoints.append({"after": m, "summary": summary, "shortlist": shortlist})

        # Human‑readable output
        print(f"Checkpoint after {m} arrivals:")
        for col in sorted(summary.keys()):
            stats = summary[col]
            print(f"{col}: n={stats['n']} mean={stats['mean']:.6f} "
                  f"min={stats['min']:.6f} max={stats['max']:.6f}")
        print(f"Top 5 rows: {shortlist}")

    # JSON answer
    print(f"ANSWER CAR-08 {json.dumps({'checkpoints': checkpoints}, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
