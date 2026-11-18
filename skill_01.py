#!/usr/bin/env python3
"""
skill_01.py

Entry point for APACHE-01 and APACHE-02 skill.

Usage:
    python skill_01.py <path_to_csv>
"""

import csv
import json
import sys
from statistics import median
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# --------------------------------------------------------------------------- #
# Utility functions
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

def compute_goal_stats(rows: List[Dict[str, Any]], goal_col: str) -> Dict[str, Any]:
    """Compute statistics for a goal column."""
    values = [row[goal_col] for row in rows if row[goal_col] is not None]
    if not values:
        raise ValueError(f"No values found for goal column '{goal_col}'.")
    values_sorted = sorted(values)
    mean_val = sum(values_sorted) / len(values_sorted)
    median_val = median(values_sorted)
    min_val = min(values_sorted)
    max_val = max(values_sorted)
    # For goal columns ending with '-' we minimize; with '+' we maximize.
    # Here we assume '-' means minimize.
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

def determine_top_rows(rows: List[Dict[str, Any]], goal_col: str, k: int) -> List[Dict[str, Any]]:
    """Return the k rows with the lowest (best) goal values."""
    # Filter rows that have a defined goal value
    valid_rows = [row for row in rows if row[goal_col] is not None]
    if not valid_rows:
        return []
    # Sort by goal ascending (best first)
    sorted_rows = sorted(valid_rows, key=lambda r: r[goal_col])
    return sorted_rows[:k]

def compute_setting_shares(
    top_rows: List[Dict[str, Any]],
    rest_rows: List[Dict[str, Any]],
    setting_cols: List[str]
) -> List[Dict[str, Any]]:
    """Compute shares for each setting column."""
    shares = []
    for col in setting_cols:
        # Top share
        top_non_missing = [r for r in top_rows if r[col] is not None]
        top_total = len(top_non_missing)
        top_ones = sum(1 for r in top_non_missing if r[col] == 1)
        top_share = top_ones / top_total if top_total > 0 else 0.0

        # Rest share
        rest_non_missing = [r for r in rest_rows if r[col] is not None]
        rest_total = len(rest_non_missing)
        rest_ones = sum(1 for r in rest_non_missing if r[col] == 1)
        rest_share = rest_ones / rest_total if rest_total > 0 else 0.0

        shares.append({
            "setting": col,
            "top_share": top_share,
            "rest_share": rest_share
        })
    # Sort by absolute difference descending, then by setting name
    shares.sort(key=lambda d: abs(d["top_share"] - d["rest_share"]), reverse=True)
    # Keep at least 2 settings
    return shares[:max(2, len(shares))]

# --------------------------------------------------------------------------- #
# Main skill logic
# --------------------------------------------------------------------------- #

def main(csv_path: str) -> None:
    rows, headers = read_csv(csv_path)

    # Identify goal column (ends with '-' or '+')
    goal_cols = [col for col in headers if col.endswith('-') or col.endswith('+')]
    if not goal_cols:
        raise ValueError("No goal column found.")
    goal_col = goal_cols[0]  # assume first
    # --------------------------------------------------------------------- #
    # APACHE-01
    # --------------------------------------------------------------------- #
    stats = compute_goal_stats(rows, goal_col)
    print("== APACHE-01 ==")
    # Print at least three lines of results
    print(f"Mean performance: {stats['mean']:.2f}")
    print(f"Median performance: {stats['median']:.2f}")
    print(f"Min/Max performance: {stats['min']} / {stats['max']}")
    # Machine-checked answer
    answer_01 = {goal_col: stats}
    print("ANSWER APACHE-01", json.dumps(answer_01, separators=(',', ':')))

    # --------------------------------------------------------------------- #
    # APACHE-02
    # --------------------------------------------------------------------- #
    total_rows = len(rows)
    k_candidate = int(0.2 * total_rows)
    k = max(5, min(20, k_candidate))
    # Ensure k does not exceed number of rows with valid goal
    valid_goal_rows = [r for r in rows if r[goal_col] is not None]
    k = min(k, len(valid_goal_rows))
    top_rows = determine_top_rows(rows, goal_col, k)
    top_row_nums = sorted([r['_row_num'] for r in top_rows])

    # Remaining rows for rest
    top_row_set = set(r['_row_num'] for r in top_rows)
    rest_rows = [r for r in rows if r['_row_num'] not in top_row_set]

    # Settings are all columns except goal and those ending with 'X'
    setting_cols = [col for col in headers if col != goal_col and not col.endswith('X')]
    setting_shares = compute_setting_shares(top_rows, rest_rows, setting_cols)

    print("\n== APACHE-02 ==")
    print(f"Top rows (unusually favorable): {top_row_nums}")
    print(f"Number of top rows: {len(top_row_nums)}")
    print(f"Total settings considered: {len(setting_cols)}")

    answer_02 = {
        "top": top_row_nums,
        "settings": setting_shares
    }
    print("ANSWER APACHE-02", json.dumps(answer_02, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
