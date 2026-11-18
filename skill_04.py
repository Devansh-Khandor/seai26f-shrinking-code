#!/usr/bin/env python3
"""
Skill 04: Real‑time performance monitoring and incremental summaries.

This module implements the following requests:

* APACHE-07 –  Detect whether each newly arrived configuration is
                unusually slow compared to all previously evaluated rows.
                An alert is issued when the arrival's Performance- value
                lies more than two standard deviations above the
                previous mean.

* APACHE-08 –  Maintain incremental performance summaries and a
                shortlist of the five best (lowest) configurations
                seen so far, after 0, 19 and 38 arrivals.  The
                summaries contain the count, mean, min and max for
                every goal column.

The input is a CSV file whose first line is a header.  The only
command‑line argument is the path to that file.
"""

import json
import math
import sys
from typing import List, Dict, Any

import ezr

# ----------------------------------------------------------------------
# Helpers ---------------------------------------------------------------
# ----------------------------------------------------------------------
def compute_stats(rows: List[ezr.ROW], goal_col: ezr.COL) -> Dict[str, Any]:
    """
    Return stats for a goal column over the given rows.
    Only non‑missing values are considered.
    """
    vals = [row[goal_col.at] for row in rows if row[goal_col.at] != "?"]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "min": None, "max": None}
    mean = sum(vals) / n
    mn = min(vals)
    mx = max(vals)
    return {"n": n, "mean": mean, "min": mn, "max": mx}


def best_rows(rows: List[ezr.ROW], goal_col: ezr.COL, count: int = 5) -> List[int]:
    """
    Return the row numbers (1‑based) of the `count` best rows seen so far.
    Best means smallest goal value.  Rows with missing goal are ignored.
    Ties are broken by earlier appearance.
    """
    # build list of (goal, index)
    cand = [(row[goal_col.at], i) for i, row in enumerate(rows, start=1)
            if row[goal_col.at] != "?"]
    # sort by goal value ascending, then index ascending
    cand.sort(key=lambda x: (x[0], x[1]))
    return [idx for _, idx in cand[:count]]


def arrival_indices(nrows: int, arrivals: int) -> List[int]:
    """
    Convert zero‑based slice indices into 1‑based row numbers.
    """
    return [i + 1 for i in range(nrows - arrivals, nrows)]


# ----------------------------------------------------------------------
# Core logic -------------------------------------------------------------
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    rows: List[ezr.ROW] = list(ezr.csv(csv_path))
    tbl: ezr.Tbl = ezr.Tbl(rows)

    # identify goal columns
    goal_cols = [c for c in tbl.cols.values()
                 if c.txt.endswith("-") or c.txt.endswith("+")]
    if not goal_cols:
        print("No goal column found.", file=sys.stderr)
        sys.exit(1)
    # For this data set we expect a single goal column
    goal_col = goal_cols[0]

    nrows = len(rows)
    arrival_count = round(0.2 * nrows)
    arrivals = rows[-arrival_count:]
    history = rows[:-arrival_count]

    # ------------------------------------------------------------------
    # APACHE-07: alert on unusually poor arrivals
    # ------------------------------------------------------------------
    print("== APACHE-07 ==")
    assessments = []

    # Build cumulative history as we iterate arrivals
    cumulative_rows = history.copy()
    for j, arrival_row in enumerate(arrivals):
        row_number = len(history) + j + 1  # 1‑based
        # compute stats from cumulative_rows
        perf_vals = [r[goal_col.at] for r in cumulative_rows if r[goal_col.at] != "?"]
        if not perf_vals:
            mean = std = None
        else:
            mean = sum(perf_vals) / len(perf_vals)
            if len(perf_vals) > 1:
                var = sum((v - mean) ** 2 for v in perf_vals) / (len(perf_vals) - 1)
                std = math.sqrt(var)
            else:
                std = 0.0

        arrival_perf = arrival_row[goal_col.at]
        if arrival_perf == "?" or mean is None:
            alert = False
        else:
            threshold = mean + 2 * (std if std else 0)
            alert = arrival_perf > threshold

        assessments.append({"row": row_number, "alert": alert})

        # record this arrival for subsequent statistics
        cumulative_rows.append(arrival_row)

        # print a brief line for the user
        print(f"Row {row_number} perf={arrival_perf} alert={alert}")

    print("ANSWER APACHE-07",
          json.dumps({"assessments": assessments}, separators=(",", ":")))

    # ------------------------------------------------------------------
    # APACHE-08: incremental summaries and shortlists
    # ------------------------------------------------------------------
    print("\n== APACHE-08 ==")
    checkpoints = [0, 19, 38]
    # cap at the number of arrivals available
    checkpoints = [m for m in checkpoints if m <= arrival_count]
    if not checkpoints:
        checkpoints = [0]
    results = []

    for m in checkpoints:
        rows_up_to_m = history + arrivals[:m]
        summary = {}
        for gc in goal_cols:
            stats = compute_stats(rows_up_to_m, gc)
            # round floats to avoid excessive precision
            stats = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in stats.items()}
            summary[gc.txt] = stats
        shortlist = best_rows(rows_up_to_m, goal_col, count=5)
        print(f"Checkpoint after {m} arrivals:")
        print(f"  Summary: {summary}")
        print(f"  Shortlist: {shortlist}")
        results.append({
            "after": m,
            "summary": summary,
            "shortlist": shortlist
        })

    print("ANSWER APACHE-08",
          json.dumps({"checkpoints": results}, separators=(",", ":")))


# ----------------------------------------------------------------------
# Entry point ------------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
