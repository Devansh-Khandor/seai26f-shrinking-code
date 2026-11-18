#!/usr/bin/env python3
"""
skill_04.py: Implements CAR-07 and CAR-08 for used‑car dealership data.

Usage:
    python skill_04.py <path_to_csv>

Outputs:
    == CAR-07 ==
    ... results ...
    ANSWER CAR-07 <json>

    == CAR-08 ==
    ... results ...
    ANSWER CAR-08 <json>
"""

import csv
import json
import math
import sys
from typing import List, Tuple, Dict, Any, Iterable

# ------------------------------------------------------------------
# CSV reading utilities
# ------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[str], List[List[Any]]]:
    """Read CSV, convert numeric cells, keep '?' as None."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows: List[List[Any]] = []
        for row in reader:
            conv_row: List[Any] = []
            for v in row:
                if v == "?":
                    conv_row.append(None)
                else:
                    try:
                        if "." in v or "e" in v.lower():
                            conv_row.append(float(v))
                        else:
                            conv_row.append(int(v))
                    except ValueError:
                        conv_row.append(v)
            rows.append(conv_row)
    return header, rows

# ------------------------------------------------------------------
# Helper functions for goal statistics and scoring
# ------------------------------------------------------------------
def goal_indices(header: List[str]) -> List[int]:
    """Return indices of goal columns (ending with '+' or '-')."""
    return [i for i, h in enumerate(header) if h[-1] in ("+", "-")]

def goal_is_max(header: List[str]) -> Dict[int, bool]:
    """Map goal index to whether it should be maximized."""
    return {i: header[i][-1] == "+" for i in goal_indices(header)}

def compute_goal_stats(rows: List[List[Any]], gidx: List[int]) -> Dict[int, Dict[str, Any]]:
    """Return stats (n, mean, min, max) for each goal column."""
    stats: Dict[int, Dict[str, Any]] = {}
    for idx in gidx:
        vals = [r[idx] for r in rows if r[idx] is not None]
        n = len(vals)
        if n == 0:
            stats[idx] = {"n": 0, "mean": None, "min": None, "max": None}
        else:
            mn = min(vals)
            mx = max(vals)
            mean_val = sum(vals) / n
            stats[idx] = {"n": n, "mean": mean_val, "min": mn, "max": mx}
    return stats

def compute_scores(rows: List[List[Any]], row_nums: List[int],
                   gidx: List[int], mins: Dict[int, float],
                   maxs: Dict[int, float], ismax: Dict[int, bool]) -> List[Tuple[float, int]]:
    """Return list of (score, row_number) for rows."""
    scores: List[Tuple[float, int]] = []
    for r, rn in zip(rows, row_nums):
        total = 0.0
        for idx in gidx:
            v = r[idx]
            if v is None:
                continue
            mn = mins[idx]
            mx = maxs[idx]
            if mn == mx:
                norm = 0.5
            else:
                if ismax[idx]:
                    norm = (v - mn) / (mx - mn)
                else:
                    norm = (mx - v) / (mx - mn)
            total += norm
        scores.append((total, rn))
    return scores

def round4(x: Any) -> Any:
    """Round floats to 4 decimals, leave ints unchanged."""
    if isinstance(x, float):
        return round(x, 4)
    return x

# ------------------------------------------------------------------
# CAR-07
# ------------------------------------------------------------------
def assess_arrivals(history: List[List[Any]], arrivals: List[List[Any]],
                    arrival_nums: List[int], header: List[str]) -> List[Dict[str, Any]]:
    """Return assessments for each arrival in order."""
    gidx = goal_indices(header)
    ismax = goal_is_max(header)
    processed_rows = history.copy()
    processed_nums = []  # row numbers
    assessments: List[Dict[str, Any]] = []

    # Build a map from row number to its data for history
    hist_nums = list(range(1, len(history) + 1))
    processed_rows.extend(arrivals)
    processed_nums.extend(arrival_nums)

    # We will evaluate one by one
    # Reset processed_rows to history only first
    processed_rows = history.copy()
    processed_nums = list(range(1, len(history) + 1))

    for r, rn in zip(arrivals, arrival_nums):
        # Add this arrival to processed lists
        processed_rows.append(r)
        processed_nums.append(rn)

        # Compute min/max over current inventory
        stats = compute_goal_stats(processed_rows, gidx)
        mins = {idx: stats[idx]["min"] for idx in gidx}
        maxs = {idx: stats[idx]["max"] for idx in gidx}

        # Compute scores for all processed rows
        scores = compute_scores(processed_rows, processed_nums, gidx, mins, maxs, ismax)
        # Sort by descending score, tie by row number ascending
        sorted_scores = sorted(scores, key=lambda x: (-x[0], x[1]))

        # Determine top 5
        top5 = [rn for _, rn in sorted_scores[:5]]
        desirable = rn in top5

        assessments.append({"row": rn, "desirable": desirable})

    return assessments

# ------------------------------------------------------------------
# CAR-08
# ------------------------------------------------------------------
def checkpoints_summary(history: List[List[Any]], arrivals: List[List[Any]],
                        arrival_nums: List[int], header: List[str]) -> List[Dict[str, Any]]:
    """Return list of checkpoints after 0, 40, 80 arrivals."""
    gidx = goal_indices(header)
    ismax = goal_is_max(header)

    checkpoints = [0, 40, 80]
    results: List[Dict[str, Any]] = []

    total_arrivals = len(arrivals)
    for m in checkpoints:
        m = min(m, total_arrivals)
        # inventory = history + first m arrivals
        inv_rows = history + arrivals[:m]
        inv_nums = list(range(1, len(history) + 1)) + arrival_nums[:m]
        # Compute stats
        stats = compute_goal_stats(inv_rows, gidx)
        summary = {}
        for idx in gidx:
            goal_name = header[idx]
            summary[goal_name] = {
                "n": stats[idx]["n"],
                "mean": round4(stats[idx]["mean"]),
                "min": round4(stats[idx]["min"]),
                "max": round4(stats[idx]["max"]),
            }
        # Compute scores
        mins = {idx: stats[idx]["min"] for idx in gidx}
        maxs = {idx: stats[idx]["max"] for idx in gidx}
        scores = compute_scores(inv_rows, inv_nums, gidx, mins, maxs, ismax)
        sorted_scores = sorted(scores, key=lambda x: (-x[0], x[1]))
        shortlist = [rn for _, rn in sorted_scores[:5]]

        results.append({
            "after": m,
            "summary": summary,
            "shortlist": shortlist
        })
    return results

# ------------------------------------------------------------------
# Main driver
# ------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    # Determine arrivals and history
    N = len(rows)
    arrivals_cnt = round(0.2 * N)
    history_rows = rows[:N - arrivals_cnt]
    arrivals_rows = rows[N - arrivals_cnt:]
    # Row numbers (1-based)
    arrival_nums = list(range(N - arrivals_cnt + 1, N + 1))

    # ---- CAR-07 ----
    print("== CAR-07 ==")
    assessments = assess_arrivals(history_rows, arrivals_rows, arrival_nums, header)
    for a in assessments:
        # compute score for printing
        # reuse score from assessment? we didn't store
        # we recompute for printing purpose
        # We'll recompute using full inventory at that point
        # For simplicity, just print row number and desirability
        print(f"Row {a['row']}: desirable={a['desirable']}")
    print(f"ANSWER CAR-07 {json.dumps({'assessments': assessments}, separators=(',', ':'))}")

    # ---- CAR-08 ----
    print("\n== CAR-08 ==")
    ckpts = checkpoints_summary(history_rows, arrivals_rows, arrival_nums, header)
    for ck in ckpts:
        print(f"Checkpoint after {ck['after']} arrivals:")
        for goal, s in ck['summary'].items():
            print(f"  {goal}: n={s['n']}, mean={s['mean']}, min={s['min']}, max={s['max']}")
        print(f"  Shortlist: {ck['shortlist']}")
    print(f"\nANSWER CAR-08 {json.dumps({'checkpoints': ckpts}, separators=(',', ':'))}")

if __name__ == "__main__":
    main()
