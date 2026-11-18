#!/usr/bin/env python3
"""
skill_07.py

Entry point for APACHE-13 and APACHE-14 skill.

Usage:
    python skill_07.py <path_to_csv>
"""

import csv
import json
import sys
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Reuse the CSV reading routine from skill_01
from skill_01 import read_csv


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def identify_goal_column(headers: List[str]) -> str:
    """Return the first column name that ends with '-' or '+'."""
    for col in headers:
        if col.endswith('-') or col.endswith('+'):
            return col
    raise ValueError("No goal column found.")


def identify_setting_columns(headers: List[str], goal_col: str) -> List[str]:
    """Return setting columns (exclude goal and columns ending with 'X')."""
    return [c for c in headers if c != goal_col and not c.endswith('X')]


# --------------------------------------------------------------------------- #
# APACHE-13 logic
# --------------------------------------------------------------------------- #

def apache_13(rows: List[Dict[str, Any]],
              goal_col: str) -> Tuple[List[int], int]:
    """
    Evaluate rows sequentially until we reach the budget or have seen all rows.
    Return the list of evaluated row numbers and the best (lowest for a min goal)
    row number among those evaluated.
    """
    total_rows = len(rows)
    # budget: at most min(30, number of rows // 5)
    budget = min(30, total_rows // 5)

    # Simple strategy: evaluate the first `budget` rows in order.
    evaluated = [r['_row_num'] for r in rows[:budget]]

    # Find best row among evaluated
    best_row_num = None
    best_value = None
    for r in rows[:budget]:
        val = r[goal_col]
        if val is None:
            continue
        if best_value is None or val < best_value:  # minimize
            best_value = val
            best_row_num = r['_row_num']

    return evaluated, best_row_num


# --------------------------------------------------------------------------- #
# APACHE-14 logic
# --------------------------------------------------------------------------- #

def apache_14(rows: List[Dict[str, Any]],
              goal_col: str) -> List[int]:
    """
    Recommend an evaluation order: top rows sorted by goal value (ascending for a min goal)
    up to the allowed budget (min(30, n//5)).
    """
    total_rows = len(rows)
    budget = min(30, total_rows // 5)

    # Filter rows with defined goal value
    valid_rows = [r for r in rows if r[goal_col] is not None]
    # Sort by goal ascending (best first)
    sorted_rows = sorted(valid_rows, key=lambda r: r[goal_col])
    order = [r['_row_num'] for r in sorted_rows[:budget]]
    return order


# --------------------------------------------------------------------------- #
# Main skill logic
# --------------------------------------------------------------------------- #

def main(csv_path: str) -> None:
    rows, headers = read_csv(csv_path)

    goal_col = identify_goal_column(headers)
    # setting_cols = identify_setting_columns(headers, goal_col)  # not needed for current tasks

    # --------------------------------------------------------------------- #
    # APACHE-13
    # --------------------------------------------------------------------- #
    evaluated, best_row_num = apache_13(rows, goal_col)

    print("== APACHE-13 ==")
    print(f"Evaluated {len(evaluated)} rows")
    print(f"Best row among evaluated: {best_row_num}")
    answer_13 = {"evaluated": evaluated, "best": best_row_num}
    print("ANSWER APACHE-13", json.dumps(answer_13, separators=(',', ':')))

    # --------------------------------------------------------------------- #
    # APACHE-14
    # --------------------------------------------------------------------- #
    order = apache_14(rows, goal_col)

    print("\n== APACHE-14 ==")
    print(f"Order length: {len(order)}")
    print(f"First 5 rows: {order[:5]}")
    answer_14 = {"order": order}
    print("ANSWER APACHE-14", json.dumps(answer_14, separators=(',', ':')))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
