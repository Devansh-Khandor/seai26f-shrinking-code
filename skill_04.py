#!/usr/bin/env python3
"""
skill_04.py

Entry point for APACHE-07 and APACHE-08 skill.

Usage:
    python skill_04.py <path_to_csv>
"""

import csv
import json
import sys
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# --------------------------------------------------------------------------- #
# Utility functions (copied from skill_01)
# --------------------------------------------------------------------------- #

def read_csv(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Read the CSV file and return list of rows (dict) and list of column names.

    Missing numeric cells ('?') are stored as None.
    Each row dict contains an additional key '_row_num' with its 1-based
    position among data rows.
    """
    rows: List[Dict[str, Any]] = []
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        if headers is None:
            raise ValueError("CSV file has no header.")
        for idx, raw_row in enumerate(reader, start=1):
            row: Dict[str, Any] = {}
            for col, val in raw_row.items():
                if val == '?':
                    row[col] = None
                else:
                    try:
                        row[col] = int(val)
                    except ValueError:
                        # Treat non-integer cells as None (should not happen here)
                        row[col] = None
            row['_row_num'] = idx
            rows.append(row)
    return rows, headers

def determine_top_rows(rows: List[Dict[str, Any]], goal_col: str, k: int) -> List[Dict[str, Any]]:
    """Return the k rows with the lowest (best) goal values."""
    valid_rows = [row for row in rows if row[goal_col] is not None]
    if not valid_rows:
        return []
    sorted_rows = sorted(valid_rows, key=lambda r: r[goal_col])
    return sorted_rows[:k]

def compute_goal_stats(rows: List[Dict[str, Any]], goal_col: str) -> Dict[str, Any]:
    """Compute statistics for a goal column."""
    values = [row[goal_col] for row in rows if row[goal_col] is not None]
    if not values:
        raise ValueError(f"No values found for goal column '{goal_col}'.")
    values_sorted = sorted(values)
    mean_val = sum(values_sorted) / len(values_sorted)
    median_val = sorted(values)[len(values_sorted)//2] if len(values_sorted)%2==1 else (sorted(values)[len(values_sorted)//2-1] + sorted(values)[len(values_sorted)//2]) / 2
    min_val = min(values_sorted)
    max_val = max(values_sorted)
    if goal_col.endswith('-'):
        best_val = min_val
        worst_val = max_val
    else:
        best_val = max_val
        worst_val = min_val
    return {
        "mean": mean_val,
        "median": median_val,
        "min": min_val,
        "max": max_val,
        "best": best_val,
        "worst": worst_val
    }

# --------------------------------------------------------------------------- #
# Helper functions for APACHE-07
# --------------------------------------------------------------------------- #

def prior_stats(rows: List[Dict[str, Any]], goal_col: str, up_to_index: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute mean and population standard deviation of goal values
    in rows with index < up_to_index (0-based).
    Returns (mean, std). If fewer than 2 values, std is 0.0.
    """
    values = [rows[i][goal_col] for i in range(up_to_index) if rows[i][goal_col] is not None]
    if not values:
        return None, None
    n = len(values)
    mean_val = sum(values) / n
    if n > 1:
        var = sum((x - mean_val) ** 2 for x in values) / n
        std_val = math.sqrt(var)
    else:
        std_val = 0.0
    return mean_val, std_val

# --------------------------------------------------------------------------- #
# Helper functions for APACHE-08
# --------------------------------------------------------------------------- #

def compute_summary_for_goal(rows: List[Dict[str, Any]], goal_col: str) -> Dict[str, Any]:
    """Return n, mean, min, max for given goal column in the provided rows."""
    values = [r[goal_col] for r in rows if r[goal_col] is not None]
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    mean_val = sum(values) / n
    min_val = min(values)
    max_val = max(values)
    return {"n": n, "mean": mean_val, "min": min_val, "max": max_val}

# --------------------------------------------------------------------------- #
# Main skill logic
# --------------------------------------------------------------------------- #

def main(csv_path: str) -> None:
    rows, headers = read_csv(csv_path)

    # Identify goal columns (ends with '-' or '+')
    goal_cols = [col for col in headers if col.endswith('-') or col.endswith('+')]
    if not goal_cols:
        raise ValueError("No goal column found.")
    goal_col = goal_cols[0]   # assume first

    # Identify all goal columns for summaries
    all_goal_cols = goal_cols

    num_rows = len(rows)
    arrivals_count = round(0.2 * num_rows)
    # Ensure we do not exceed available rows
    arrivals_count = min(arrivals_count, num_rows)
    history_rows = rows[:-arrivals_count] if arrivals_count > 0 else rows[:]
    arrival_rows = rows[-arrivals_count:] if arrivals_count > 0 else []

    # --------------------------------------------------------------------- #
    # APACHE-07
    # --------------------------------------------------------------------- #
    assessments = []
    # We will iterate arrivals in order
    for i, arrival in enumerate(arrival_rows):
        # index of arrival in full list
        arrival_idx = len(history_rows) + i
        mean_prev, std_prev = prior_stats(rows, goal_col, arrival_idx)
        alert = False
        val = arrival[goal_col]
        if val is not None and mean_prev is not None:
            if goal_col.endswith('-'):  # minimize
                threshold = mean_prev + 2 * std_prev
                if val > threshold:
                    alert = True
            else:  # maximize
                threshold = mean_prev - 2 * std_prev
                if val < threshold:
                    alert = True
        assessments.append({"row": arrival['_row_num'], "alert": alert})

    print("== APACHE-07 ==")
    for a in assessments:
        print(f"Row {a['row']}: alert={a['alert']}")
    # Summary of alerts
    num_alerts = sum(1 for a in assessments if a['alert'])
    print(f"Total arrivals: {len(assessments)}, Alerts: {num_alerts}")

    answer_07 = {"assessments": assessments}
    print("ANSWER APACHE-07", json.dumps(answer_07, separators=(',', ':')))

    # --------------------------------------------------------------------- #
    # APACHE-08
    # --------------------------------------------------------------------- #
    checkpoints = [0, 19, 38]
    checkpoint_results = []
    for m in checkpoints:
        if m > arrivals_count:
            # If requested checkpoint beyond arrivals, skip it
            continue
        # Determine rows seen so far
        seen_rows = history_rows + arrival_rows[:m]
        summary = {}
        for col in all_goal_cols:
            stats = compute_summary_for_goal(seen_rows, col)
            # Round numeric values to 6 decimal places for consistency
            summary[col] = {
                "n": stats["n"],
                "mean": round(stats["mean"], 6),
                "min": stats["min"],
                "max": stats["max"]
            }
        # Shortlist best 5 rows so far
        top_rows = determine_top_rows(seen_rows, goal_col, 5)
        shortlist = [r['_row_num'] for r in top_rows]
        checkpoint_results.append({
            "after": m,
            "summary": summary,
            "shortlist": shortlist
        })

    print("\n== APACHE-08 ==")
    for ckpt in checkpoint_results:
        print(f"After {ckpt['after']} arrivals: short list {ckpt['shortlist']}")
    answer_08 = {"checkpoints": checkpoint_results}
    print("ANSWER APACHE-08", json.dumps(answer_08, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
