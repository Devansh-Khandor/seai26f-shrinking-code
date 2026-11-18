#!/usr/bin/env python3
"""
Skill 04: NASA-07 and NASA-08

This script answers the NASA-07 and NASA-08 requests.

NASA-07: For every arriving project (the last 20% of the rows),
          predict Effort- and Defects- using the mean of the
          historical values of those columns.  An alert is raised
          when the prediction is unusually high, defined as
          exceeding the historical mean by more than twice the
          historical standard deviation.

NASA-08: After 0, 9 and 19 arrivals have been incorporated
          into the historical data, output a statistical
          summary for each goal column (Kloc+, Effort-, Defects-)
          and the 5 best projects seen so far.  Projects are
          ranked using the same combined‑rank rule that is used
          in skill_01.py.

The program prints a human‑readable section for each request
and a single line with a JSON answer for each request, as
required by the evaluation harness.
"""

import csv
import json
import math
import sys
from typing import Any, Dict, List, Tuple, Optional

# ------------------------------------------------------------------
# Helper functions – copied from skill_01.py

GOAL_COLUMNS = {
    "Kloc+": "max",      # maximize
    "Effort-": "min",    # minimize
    "Defects-": "min",   # minimize
}

def read_data(csv_path: str) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Read CSV file and return a list of row dictionaries and a list
    of their 1‑based row numbers.
    """
    rows: List[Dict[str, Any]] = []
    row_nums: List[int] = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for rn, rec in enumerate(reader, start=1):
            row_nums.append(rn)
            parsed: Dict[str, Any] = {}
            for key, val in rec.items():
                # strip whitespace
                val = val.strip()
                if val == "?":
                    parsed[key] = None
                else:
                    # Try numeric conversion if possible
                    try:
                        if '.' in val or 'e' in val.lower():
                            parsed[key] = float(val)
                        else:
                            parsed[key] = int(val)
                    except ValueError:
                        parsed[key] = val  # keep as string
            rows.append(parsed)
    return rows, row_nums

# ------------------------------------------------------------------
# Statistics helpers

def mean_std(values: List[float]) -> Tuple[float, float]:
    """Return mean and standard deviation of a numeric list."""
    if not values:
        return (float('nan'), float('nan'))
    n = len(values)
    mean_val = sum(values) / n
    var = sum((x - mean_val) ** 2 for x in values) / n
    std_val = math.sqrt(var)
    return mean_val, std_val

def goal_stats(rows: List[Dict[str, Any]], goal_col: str) -> Dict[str, Any]:
    """
    Compute n, mean, min, max for a goal column over given rows.
    Missing values are ignored.
    """
    vals = [row[goal_col] for row in rows if isinstance(row.get(goal_col), (int, float))]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "min": None, "max": None}
    mean_val = sum(vals) / n
    min_val = min(vals)
    max_val = max(vals)
    return {"n": n, "mean": round(mean_val, 6), "min": min_val, "max": max_val}

# ------------------------------------------------------------------
# NASA-07 logic

def predict_and_alert(rows: List[Dict[str, Any]], arrivals: List[int]) -> List[Dict[str, Any]]:
    """
    For each arriving row, predict Effort- and Defects- as the
    historical mean of those columns.  An alert is raised when
    the prediction exceeds the historical mean by more than
    two standard deviations.
    """
    # Separate history from arrivals
    history_rows = [rows[i-1] for i in arrivals]  # not used here
    # Determine the split: all rows before the first arrival are history
    first_arrival = arrivals[0]
    history = rows[:first_arrival-1]
    # Compute historical statistics for Effort- and Defects-
    effort_vals = [r["Effort-"] for r in history if isinstance(r.get("Effort-"), (int, float))]
    defect_vals = [r["Defects-"] for r in history if isinstance(r.get("Defects-"), (int, float))]

    effort_mean, effort_std = mean_std(effort_vals)
    defect_mean, defect_std = mean_std(defect_vals)

    assessments = []
    for arr_row_num in arrivals:
        row = rows[arr_row_num-1]
        pred_effort = effort_mean
        pred_defects = defect_mean
        alert = False
        # Define "unusually high" as > mean + 2*std
        if not math.isnan(effort_std) and pred_effort > effort_mean + 2 * effort_std:
            alert = True
        if not math.isnan(defect_std) and pred_defects > defect_mean + 2 * defect_std:
            alert = True
        assessments.append({
            "row": arr_row_num,
            "predicted_effort": round(pred_effort, 6),
            "predicted_defects": round(pred_defects, 6),
            "alert": alert
        })
    return assessments

# ------------------------------------------------------------------
# NASA-08 logic

def compute_checkpoints(rows: List[Dict[str, Any]], row_nums: List[int],
                       arrival_counts: List[int]) -> List[Dict[str, Any]]:
    """
    Build checkpoints after the specified numbers of arrivals.
    """
    # Determine the split
    total_rows = len(rows)
    n_arrivals = total_rows - sum(arrival_counts)
    # Compute the starting indices for arrivals
    arrival_start = total_rows - n_arrivals
    checkpoints = []
    for m in arrival_counts:
        # Rows considered: history + first m arrivals
        idx_cut = arrival_start + m
        subset = rows[:idx_cut]
        summary = {}
        for goal in GOAL_COLUMNS:
            summary[goal] = goal_stats(subset, goal)
        checkpoints.append({"after": m, "summary": summary})
    return checkpoints

def shortlist_best(rows: List[Dict[str, Any]], row_nums: List[int]) -> List[int]:
    """
    Return the 5 best rows seen so far, based on combined ranking of
    Kloc+ (max), Effort- (min), Defects- (min).
    """
    # Filter rows with all three goal values present
    valid = [idx for idx, row in enumerate(rows) if all(isinstance(row.get(c), (int, float)) for c in GOAL_COLUMNS)]
    if not valid:
        return []

    # Compute rank for each column
    ranks = {c: {} for c in GOAL_COLUMNS}
    for c in GOAL_COLUMNS:
        # Prepare list of (value, idx)
        vals = [(rows[i][c], i) for i in valid]
        # Sort ascending if minimize, descending if maximize
        reverse = GOAL_COLUMNS[c] == "max"
        sorted_vals = sorted(vals, key=lambda x: x[0], reverse=reverse)
        for rank, (_, idx) in enumerate(sorted_vals, start=1):
            ranks[c][idx] = rank

    # Combined rank sum
    combined = []
    for idx in valid:
        total_rank = sum(ranks[c][idx] for c in GOAL_COLUMNS)
        combined.append((total_rank, rows[idx]["Kloc+"], idx))

    # Sort by combined rank asc, then Kloc+ asc for tie-breaking
    combined_sorted = sorted(combined, key=lambda x: (x[0], x[1]))
    top_indices = [idx for _, _, idx in combined_sorted[:5]]
    # Convert to 1‑based row numbers
    return [row_nums[i] for i in top_indices]

# ------------------------------------------------------------------
# Main entry point

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    rows, row_nums = read_data(csv_path)

    total_rows = len(rows)
    # Arrivals are the last round(0.2 * total_rows) rows
    n_arrivals = round(0.2 * total_rows)
    arrival_start_idx = total_rows - n_arrivals
    arrivals = [row_nums[i] for i in range(arrival_start_idx, total_rows)]

    # ------------------------------------------------------------------
    # NASA-07
    print("== NASA-07 ==")
    assessments = predict_and_alert(rows, arrivals)
    for a in assessments:
        print(f"Row {a['row']}: predicted Effort-={a['predicted_effort']}, "
              f"predicted Defects-={a['predicted_defects']}, alert={a['alert']}")
    answer_nasa07 = {"assessments": assessments}
    print(f"ANSWER NASA-07 {json.dumps(answer_nasa07)}")

    # ------------------------------------------------------------------
    # NASA-08
    print("\n== NASA-08 ==")
    # Checkpoints after 0, 9, 19 arrivals
    checkpoints_needed = [0, 9, 19]
    checkpoints = compute_checkpoints(rows, row_nums, checkpoints_needed)
    for cp in checkpoints:
        print(f"After {cp['after']} arrivals:")
        for goal, stats in cp['summary'].items():
            print(f"  {goal}: n={stats['n']}, mean={stats['mean']}, min={stats['min']}, max={stats['max']}")
    shortlist = shortlist_best(rows, row_nums)
    print(f"Shortlist: {shortlist}")

    checkpoints_json = [{"after": cp["after"], "summary": cp["summary"]} for cp in checkpoints]
    answer_nasa08 = {"checkpoints": checkpoints_json, "shortlist": shortlist}
    print(f"ANSWER NASA-08 {json.dumps(answer_nasa08)}")

if __name__ == "__main__":
    main()
