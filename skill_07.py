#!/usr/bin/env python3
"""
Skill 07: NASA-13 and NASA-14

This script answers the NASA-13 and NASA-14 requests for
the given CSV file.

Usage:
    python skill_07.py <csv path>
"""

import csv
import json
import math
import sys
from typing import Any, Dict, List, Tuple
from collections import defaultdict, Counter

# Import helper functions and goal definitions from skill_01
try:
    from skill_01 import read_data, GOAL_COLUMNS
except Exception as e:
    # Fallback: minimal implementation if import fails
    def read_data(csv_path: str) -> Tuple[List[Dict[str, Any]], List[int]]:
        rows: List[Dict[str, Any]] = []
        row_nums: List[int] = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for rn, rec in enumerate(reader, start=1):
                row_nums.append(rn)
                parsed: Dict[str, Any] = {}
                for key, val in rec.items():
                    val = val.strip()
                    if val == "?":
                        parsed[key] = None
                    else:
                        try:
                            if '.' in val or 'e' in val.lower():
                                parsed[key] = float(val)
                            else:
                                parsed[key] = int(val)
                        except ValueError:
                            parsed[key] = val
                rows.append(parsed)
        return rows, row_nums
    GOAL_COLUMNS = {
        "Kloc+": "max",
        "Effort-": "min",
        "Defects-": "min",
    }

# ------------------------------------------------------------------
# Helpers

def is_numeric(val: Any) -> bool:
    return isinstance(val, (int, float))

def compute_global_minmax(rows: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    """Return dict col -> (min, max) over all numeric values."""
    mins: Dict[str, float] = {}
    maxs: Dict[str, float] = {}
    for row in rows:
        for col, val in row.items():
            if is_numeric(val):
                if col not in mins or val < mins[col]:
                    mins[col] = val
                if col not in maxs or val > maxs[col]:
                    maxs[col] = val
    return {col: (mins[col], maxs[col]) for col in mins}

def group_stats(rows: List[Dict[str, Any]], condition) -> Dict[str, float]:
    """Return mean of each goal column for rows satisfying condition."""
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        if condition(row):
            for g in GOAL_COLUMNS:
                val = row.get(g)
                if is_numeric(val):
                    sums[g] += val
                    counts[g] += 1
    means: Dict[str, float] = {}
    for g, cnt in counts.items():
        if cnt > 0:
            means[g] = sums[g] / cnt
    return means

def normalized_score(mean: float, g_min: float, g_max: float, direction: str) -> float:
    """Normalize a mean value to [0,1] where higher is better."""
    if g_max == g_min:
        return 0.0
    if direction == "max":
        return (mean - g_min) / (g_max - g_min)
    else:  # min
        return (g_max - mean) / (g_max - g_min)

def evaluate_condition(rows: List[Dict[str, Any]],
                       global_minmax: Dict[str, Tuple[float, float]],
                       col: str,
                       cond_repr: str) -> Tuple[float, int, Any]:
    """
    Evaluate a single condition on column 'col'.
    Return (total_norm_score, matching_count, cond_value)
    cond_value is the value (str) for categorical or [low, high] for numeric.
    """
    # Determine if numeric
    numeric_vals = [row[col] for row in rows if row.get(col) is not None and is_numeric(row[col])]
    if not numeric_vals:
        # Treat as categorical
        values = set(row[col] for row in rows if row.get(col) is not None)
        best = None
        best_score = -1.0
        best_match = 0
        for val in values:
            def cond(r: Dict[str, Any]) -> bool:
                return r.get(col) == val
            means = group_stats(rows, cond)
            if not means:
                continue
            score = 0.0
            for g, mean_val in means.items():
                g_min, g_max = global_minmax.get(g, (0, 0))
                score += normalized_score(mean_val, g_min, g_max, GOAL_COLUMNS[g])
            if score > best_score:
                best_score = score
                best_match = sum(1 for r in rows if r.get(col) == val)
                best = val
        return best_score, best_match, best
    else:
        # Numeric column: evaluate two ranges [min, median] and [median, max]
        numeric_vals_sorted = sorted(numeric_vals)
        med = numeric_vals_sorted[len(numeric_vals_sorted)//2]
        min_val = numeric_vals_sorted[0]
        max_val = numeric_vals_sorted[-1]
        ranges = [(min_val, med), (med, max_val)]
        best = None
        best_score = -1.0
        best_match = 0
        for low, high in ranges:
            def cond(r: Dict[str, Any]) -> bool:
                v = r.get(col)
                return v is not None and is_numeric(v) and low <= v <= high
            means = group_stats(rows, cond)
            if not means:
                continue
            score = 0.0
            for g, mean_val in means.items():
                g_min, g_max = global_minmax.get(g, (0, 0))
                score += normalized_score(mean_val, g_min, g_max, GOAL_COLUMNS[g])
            match = sum(1 for r in rows if cond(r))
            if score > best_score:
                best_score = score
                best_match = match
                best = [low, high]
        return best_score, best_match, best

# ------------------------------------------------------------------
# Main logic

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    rows, row_nums = read_data(csv_path)

    # Identify non-goal columns
    all_columns = rows[0].keys()
    non_goal_cols = [c for c in all_columns if c not in GOAL_COLUMNS and not c.endswith('X')]

    # Compute global min/max for goals
    global_minmax = compute_global_minmax(rows)

    # Evaluate each non-goal column
    best_overall_score = -1.0
    best_rule = None
    best_match_count = 0

    for col in non_goal_cols:
        score, match, cond_val = evaluate_condition(rows, global_minmax, col, str(col))
        if score > best_overall_score:
            best_overall_score = score
            best_rule = {col: cond_val}
            best_match_count = match

    # NASA-13 output
    print("== NASA-13 ==")
    print(f"Non-goal columns examined: {len(non_goal_cols)}")
    print(f"Chosen rule: {best_rule}")
    print(f"Matching rows: {best_match_count}")

    answer_n13 = {
        "rule": best_rule,
        "matching": best_match_count
    }
    print(f"ANSWER NASA-13 {json.dumps(answer_n13)}")

    # NASA-14 logic
    held_out_rows = [rn for rn in row_nums if rn % 5 == 0]
    # Limit to budget
    budget = min(30, len(rows) // 5)
    order = sorted(held_out_rows)[:budget]

    print("\n== NASA-14 ==")
    print(f"Number of held‑out rows: {len(held_out_rows)}")
    print(f"Order (first to evaluate): {order}")

    answer_n14 = {
        "order": order
    }
    print(f"ANSWER NASA-14 {json.dumps(answer_n14)}")

if __name__ == "__main__":
    main()
