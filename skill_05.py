#!/usr/bin/env python3
"""
Skill 05: NASA-09 and NASA-10

This script analyses projects that required unusually high
development effort (NASA-09) and projects with unusually high
defect counts (NASA-10).  It builds on the helper functions
from the previous skill scripts and outputs human‑readable
information as well as machine‑checked JSON answers.
"""

import csv
import json
import math
import sys
import statistics
from collections import Counter
from typing import List, Dict, Any, Tuple

# ------------------------------------------------------------------
# Helper functions copied from skill_01 / skill_04

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
            parsed = {}
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

def mean_std(values: List[float]) -> Tuple[float, float]:
    """Return mean and standard deviation of a numeric list."""
    if not values:
        return (float('nan'), float('nan'))
    n = len(values)
    mean_val = sum(values) / n
    var = sum((x - mean_val) ** 2 for x in values) / n
    std_val = math.sqrt(var)
    return mean_val, std_val

# ------------------------------------------------------------------
# Utility helpers

def get_header(csv_path: str) -> List[str]:
    """Return the header fields from the CSV file."""
    with open(csv_path, newline='') as f:
        return f.readline().strip().split(',')

def is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float))

def compute_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float('nan')

def compute_mode(values: List[Any]) -> Any:
    if not values:
        return None
    cnt = Counter(values)
    return cnt.most_common(1)[0][0]

def typical_value(values: List[Any]) -> Any:
    """Return mean for numeric values, mode for others."""
    if not values:
        return None
    if all(is_numeric(v) for v in values):
        return round(compute_mean(values), 6)
    return compute_mode(values)

# ------------------------------------------------------------------
# NASA-09 & NASA-10 core logic

def identify_high_rows(rows: List[Dict[str, Any]],
                      goal_col: str,
                      threshold_multiplier: float = 2.0,
                      min_count: int = 5) -> List[int]:
    """
    Identify rows whose goal_col value is unusually high.
    Return a list of 1‑based row numbers.
    """
    # Collect numeric values and indices
    vals = [(i, row[goal_col])
            for i, row in enumerate(rows)
            if is_numeric(row.get(goal_col))]

    if not vals:
        return []

    values_only = [v for _, v in vals]
    mean_val, std_val = mean_std(values_only)
    threshold = mean_val + threshold_multiplier * std_val

    high = [i for i, v in vals if v > threshold]

    # If fewer than min_count, take the top min_count largest values
    if len(high) < min_count:
        high = [i for i, _ in sorted(vals, key=lambda x: x[1], reverse=True)][:min_count]

    return high

def identify_low_rows(rows: List[Dict[str, Any]],
                      goal_col: str,
                      min_count: int = 5) -> List[int]:
    """
    Identify rows with the lowest goal_col values.
    Return a list of 1‑based row numbers.
    """
    vals = [(i, row[goal_col])
            for i, row in enumerate(rows)
            if is_numeric(row.get(goal_col))]

    if not vals:
        return []

    low = [i for i, _ in sorted(vals, key=lambda x: x[1])][:min_count]
    return low

def group_stats_for_col(group_indices: List[int],
                        col: str,
                        rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean/median/typical values for a group on a column."""
    values = [rows[i][col] for i in group_indices if rows[i][col] is not None]
    if not values:
        return {"values": [], "mean": None, "median": None, "typical": None}
    numeric_vals = [v for v in values if is_numeric(v)]
    if numeric_vals:
        mean_val = round(compute_mean(numeric_vals), 6)
        median_val = statistics.median(numeric_vals)
        typical = round(median_val, 6)  # use median as typical
        return {"values": numeric_vals, "mean": mean_val, "median": median_val, "typical": typical}
    else:
        mode_val = compute_mode(values)
        return {"values": values, "mean": None, "median": None, "typical": mode_val}

def diff_metric(poor_stats: Dict[str, Any], nonpoor_stats: Dict[str, Any]) -> float:
    """Return absolute difference between means for numeric,
       or difference in mode frequencies for categorical."""
    if poor_stats["mean"] is not None and nonpoor_stats["mean"] is not None:
        return abs(poor_stats["mean"] - nonpoor_stats["mean"])
    # categorical
    vals = poor_stats["values"] + nonpoor_stats["values"]
    if not vals:
        return 0.0
    all_cnt = Counter(vals)
    # compute proportion of each mode
    poor_mode = compute_mode(poor_stats["values"])
    nonpoor_mode = compute_mode(nonpoor_stats["values"])
    poor_freq = all_cnt[poor_mode] / len(vals)
    nonpoor_freq = all_cnt[nonpoor_mode] / len(vals)
    return abs(poor_freq - nonpoor_freq)

def diff_metric_bad_good(bad_stats: Dict[str, Any], good_stats: Dict[str, Any]) -> float:
    """Same as diff_metric but between bad and good groups."""
    return diff_metric(bad_stats, good_stats)

# ------------------------------------------------------------------
# Main entry point

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    rows, row_nums = read_data(csv_path)
    header = get_header(csv_path)

    # Identify non‑goal columns for analysis
    goal_cols = [c for c in header if c in GOAL_COLUMNS]
    non_goal_cols = [c for c in header if c not in goal_cols and not c.endswith('X')]

    # ------------------------------------------------------------------
    # NASA-09: high Effort- (poor)
    effort_col = "Effort-"
    poor_indices = identify_high_rows(rows, effort_col, threshold_multiplier=2.0, min_count=5)
    poor_row_nums = [row_nums[i] for i in poor_indices]

    # Compute ranking of features most associated with poor Effort-
    ranking = []
    for col in non_goal_cols:
        # Compute stats for poor and non‑poor groups
        nonpoor_indices = [i for i in range(len(rows)) if i not in poor_indices]
        poor_stats = group_stats_for_col(poor_indices, col, rows)
        nonpoor_stats = group_stats_for_col(nonpoor_indices, col, rows)
        diff = diff_metric(poor_stats, nonpoor_stats)
        ranking.append((diff, col, poor_stats, nonpoor_stats))
    ranking_sorted = sorted(ranking, key=lambda x: x[0], reverse=True)
    top_features = [col for _, col, _, _ in ranking_sorted[:3]]

    # Human‑readable NASA‑09
    print("== NASA-09 ==")
    print(f"Number of poor effort projects: {len(poor_row_nums)}")
    print(f"Top associated features: {top_features}")
    if poor_indices:
        poor_efforts = [rows[i][effort_col] for i in poor_indices]
        avg_poor = round(compute_mean(poor_efforts), 2)
        print(f"Average Effort- of poor projects: {avg_poor}")
    # Machine‑checked answer
    answer_n09 = {"poor": poor_row_nums, "ranking": top_features}
    print(f"ANSWER NASA-09 {json.dumps(answer_n09)}")

    # ------------------------------------------------------------------
    # NASA-10: high Defects- (bad) vs low Defects- (good)
    defect_col = "Defects-"
    bad_indices = identify_high_rows(rows, defect_col, threshold_multiplier=2.0, min_count=5)
    good_indices = identify_low_rows(rows, defect_col, min_count=5)

    bad_row_nums = [row_nums[i] for i in bad_indices]
    good_row_nums = [row_nums[i] for i in good_indices]

    # Compute differences between bad and good groups
    differences = []
    for col in non_goal_cols:
        bad_stats = group_stats_for_col(bad_indices, col, rows)
        good_stats = group_stats_for_col(good_indices, col, rows)
        diff = diff_metric_bad_good(bad_stats, good_stats)
        differences.append((diff, col, bad_stats, good_stats))
    differences_sorted = sorted(differences, key=lambda x: x[0], reverse=True)
    top_diff = []
    for _, col, bad_stats, good_stats in differences_sorted[:3]:
        bad_val = bad_stats["typical"]
        good_val = good_stats["typical"]
        top_diff.append({"feature": col, "bad": bad_val, "good": good_val})

    # Human‑readable NASA‑10
    print("\n== NASA-10 ==")
    print(f"Number of bad defect projects: {len(bad_row_nums)}")
    print(f"Number of good defect projects: {len(good_row_nums)}")
    print(f"Top differences: {top_diff}")

    # Machine‑checked answer
    answer_n10 = {"bad": bad_row_nums, "good": good_row_nums, "differences": top_diff}
    print(f"ANSWER NASA-10 {json.dumps(answer_n10)}")

if __name__ == "__main__":
    main()
