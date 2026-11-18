#!/usr/bin/env python3
"""
skill_04.py

Adds two new requests to the Apache analysis toolkit:
  - APACHE-07: As new benchmark results arrive, assess each configuration immediately and
               alert me when its measured performance is unusually poor compared with
               previously tested configurations.
  - APACHE-08: Keep our performance summaries and recommended-configuration shortlist
               updated as additional benchmark records become available, without
               requiring a complete manual restart of the analysis.

The implementation re‑uses helper functions from skill_01.py.
"""

import sys
import json
import numpy as np
from skill_01 import read_csv, parse_column_values, compute_stats

def arrivals_and_history(rows):
    """
    Split the rows into history and arrivals.

    Returns (history_rows, arrivals_rows)
    """
    total = len(rows)
    arrivals_count = round(0.2 * total)
    arrivals_start = total - arrivals_count
    return rows[:arrivals_start], rows[arrivals_start:]

def assess_arrivals(rows, history, arrivals, perf_col):
    """
    For each arrival, compute whether its Performance- is unusually poor compared
    with all rows before it (history + earlier arrivals).

    Unusually poor is defined as value < mean - 2*std of previous values.
    If no previous values, alert is False.
    If arrival has missing performance, alert is False.
    """
    assessments = []
    # Build a cumulative list of previous rows
    cumulative = history[:]  # start with history
    for i, row in enumerate(arrivals):
        # compute stats on cumulative
        vals = parse_column_values(cumulative, perf_col)
        if np.isnan(vals).all():
            mean = np.nan
            std = np.nan
        else:
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
        perf = row[perf_col]
        if perf == '?':
            alert = False
        else:
            perf_val = float(perf)
            threshold = mean - 2 * std if not np.isnan(mean) and not np.isnan(std) else None
            alert = threshold is not None and perf_val < threshold
        assessments.append({"row": len(history) + i + 1, "alert": alert})
        cumulative.append(row)
    return assessments

def compute_checkpoint_summary(rows_subset, goal_cols):
    """
    Compute summary statistics for each goal column in the subset.
    Returns a dict: goal -> {n, mean, min, max}
    """
    summary = {}
    for col in goal_cols:
        vals = parse_column_values(rows_subset, col)
        stats = compute_stats(vals)
        # stats may contain None if all NaN; replace with None explicitly
        n = len([v for v in vals if not np.isnan(v)])
        summary[col] = {
            "n": n,
            "mean": stats["mean"],
            "min": stats["min"],
            "max": stats["max"]
        }
    return summary

def shortlist_best(rows_subset, goal_col, best_n=5):
    """
    Return the row numbers (1‑based) of the best rows seen so far
    according to the goal column direction.
    """
    vals = parse_column_values(rows_subset, goal_col)
    is_minimize = goal_col.endswith('-')
    # collect (value, row_num) pairs where value is not NaN
    pairs = [(float(v), idx + 1) for idx, v in enumerate(vals) if not np.isnan(v)]
    if not pairs:
        return []
    # sort
    pairs.sort(key=lambda x: x[0], reverse=not is_minimize)
    return [row for _, row in pairs[:best_n]]

def main(csv_path):
    rows, headers = read_csv(csv_path)

    # Identify goal columns
    goal_cols = [h for h in headers if h.endswith('+') or h.endswith('-')]
    if not goal_cols:
        print("No goal columns found.")
        sys.exit(1)
    perf_col = goal_cols[0]  # use first goal column for assessment & shortlist

    # Split into history and arrivals
    history, arrivals = arrivals_and_history(rows)

    # ---------------- APACHE-07 ----------------
    print("== APACHE-07 ==")
    assessments = assess_arrivals(rows, history, arrivals, perf_col)
    for a in assessments:
        print(f"Row {a['row']}: alert={str(a['alert']).lower()}")
    print("ANSWER APACHE-07", json.dumps({"assessments": assessments}, separators=(',', ':')))

    # ---------------- APACHE-08 ----------------
    print("\n== APACHE-08 ==")
    checkpoints = [0, 19, 38]
    checkpoint_results = []
    for m in checkpoints:
        # Ensure we don't ask for more arrivals than we have
        m = min(m, len(arrivals))
        subset = rows[:len(history) + m]
        summary = compute_checkpoint_summary(subset, goal_cols)
        shortlist = shortlist_best(subset, perf_col, best_n=5)
        print(f"After {m} arrivals:")
        for col, stats in summary.items():
            print(f"  {col}: n={stats['n']}, mean={stats['mean']:.6g}, min={stats['min']:.6g}, max={stats['max']:.6g}")
        print(f"  Shortlist: {shortlist}")
        checkpoint_results.append({
            "after": m,
            "summary": summary,
            "shortlist": shortlist
        })
    print("ANSWER APACHE-08", json.dumps({"checkpoints": checkpoint_results}, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
