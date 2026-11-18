#!/usr/bin/env python3
"""
skill_02.py

Entry point for APACHE-03 and APACHE-04 skill.

Usage:
    python skill_02.py <path_to_csv>
"""

import csv
import json
import sys
import math
from statistics import mean, median
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
    # For goal columns ending with `-` we minimize; with `+` we maximize.
    # Here we assume `-` means minimize.
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
# New skill logic
# --------------------------------------------------------------------------- #

def predict_performance(rows: List[Dict[str, Any]], goal_col: str) -> Tuple[Dict[str, float], float]:
    """Predict performance for held-out rows (row number divisible by 5)."""
    # Determine training and held-out sets
    training_rows = [r for r in rows if r['_row_num'] % 5 != 0]
    heldout_rows = [r for r in rows if r['_row_num'] % 5 == 0]

    # Compute mean performance on training data
    training_values = [r[goal_col] for r in training_rows if r[goal_col] is not None]
    if not training_values:
        raise ValueError("No training values available for prediction.")
    training_mean = sum(training_values) / len(training_values)

    # Predictions: same mean for all held-out rows
    predictions = {str(r['_row_num']): training_mean for r in heldout_rows}

    # Compute MAE against actual values for held-out rows
    errors = []
    for r in heldout_rows:
        actual = r[goal_col]
        if actual is not None:
            errors.append(abs(training_mean - actual))
    mae = sum(errors) / len(errors) if errors else 0.0

    return predictions, mae


def find_pairs_differ_by_one_setting(rows: List[Dict[str, Any]],
                                     setting_cols: List[str],
                                     goal_col: str) -> List[Tuple[int, int, str]]:
    """Return list of tuples (row_a_num, row_b_num, diff_setting) where rows differ in exactly one setting."""
    pairs = []
    # Pre-filter rows with complete data
    valid_rows = [r for r in rows
                  if r[goal_col] is not None and all(r[col] is not None for col in setting_cols)]
    # Build mapping from row number to row dict
    row_map = {r['_row_num']: r for r in valid_rows}
    row_nums = sorted(row_map.keys())

    for i, a_num in enumerate(row_nums):
        a = row_map[a_num]
        for b_num in row_nums[i+1:]:
            b = row_map[b_num]
            diff_cols = [col for col in setting_cols if a[col] != b[col]]
            if len(diff_cols) == 1:
                pairs.append((a_num, b_num, diff_cols[0]))
    return pairs


def compute_setting_stats(rows: List[Dict[str, Any]],
                          setting_cols: List[str],
                          goal_col: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute coefficient (mean diff) and std deviation of actual diffs for each setting."""
    coeffs = {}
    stds = {}
    # Precompute means for each setting value
    for col in setting_cols:
        ones = [r[goal_col] for r in rows if r[col] == 1 and r[goal_col] is not None]
        zeros = [r[goal_col] for r in rows if r[col] == 0 and r[goal_col] is not None]
        if ones and zeros:
            coeff = (sum(ones)/len(ones)) - (sum(zeros)/len(zeros))
            coeffs[col] = coeff
        else:
            coeffs[col] = 0.0

    # Compute actual differences for each pair
    pair_diffs_by_setting: Dict[str, List[float]] = {col: [] for col in setting_cols}
    pairs = find_pairs_differ_by_one_setting(rows, setting_cols, goal_col)
    for a_num, b_num, diff_col in pairs:
        a = next(r for r in rows if r['_row_num'] == a_num)
        b = next(r for r in rows if r['_row_num'] == b_num)
        # Identify which row has setting=1
        if a[diff_col] == 1:
            diff = b[goal_col] - a[goal_col]
        else:
            diff = a[goal_col] - b[goal_col]
        pair_diffs_by_setting[diff_col].append(diff)

    for col, diffs in pair_diffs_by_setting.items():
        if diffs:
            # compute standard deviation
            mean_diff = sum(diffs) / len(diffs)
            var = sum((x - mean_diff) ** 2 for x in diffs) / len(diffs)
            stds[col] = math.sqrt(var)
        else:
            stds[col] = 0.0

    return coeffs, stds


def estimate_pairs(rows: List[Dict[str, Any]],
                   setting_cols: List[str],
                   goal_col: str,
                   min_pairs: int = 5) -> List[Dict[str, Any]]:
    """Estimate performance differences for pairs of configurations that differ by one setting."""
    coeffs, stds = compute_setting_stats(rows, setting_cols, goal_col)
    pairs_raw = find_pairs_differ_by_one_setting(rows, setting_cols, goal_col)

    estimates = []
    for a_num, b_num, diff_col in pairs_raw:
        est_diff = coeffs.get(diff_col, 0.0)
        sd = stds.get(diff_col, 0.0)
        low = est_diff - sd
        high = est_diff + sd
        estimates.append({
            "a": a_num,
            "b": b_num,
            "setting": diff_col,
            "difference": est_diff,
            "low": low,
            "high": high
        })
        if len(estimates) >= min_pairs:
            break
    return estimates


def main(csv_path: str) -> None:
    rows, headers = read_csv(csv_path)

    # Identify goal column
    goal_cols = [col for col in headers if col.endswith('-') or col.endswith('+')]
    if not goal_cols:
        raise ValueError("No goal column found.")
    goal_col = goal_cols[0]

    # Identify setting columns
    setting_cols = [col for col in headers if col != goal_col and not col.endswith('X')]

    # --------------------------------------------------------------------- #
    # APACHE-03
    # --------------------------------------------------------------------- #
    predictions, mae = predict_performance(rows, goal_col)
    print("== APACHE-03 ==")
    # Print predictions and MAE
    print("Predicted performance for held-out rows:")
    print(json.dumps({"predictions": predictions, "mae": mae}, separators=(',', ':')))
    answer_03 = {"predictions": predictions, "mae": mae}
    print("ANSWER APACHE-03", json.dumps(answer_03, separators=(',', ':')))

    # --------------------------------------------------------------------- #
    # APACHE-04
    # --------------------------------------------------------------------- #
    pair_estimates = estimate_pairs(rows, setting_cols, goal_col, min_pairs=5)
    print("\n== APACHE-04 ==")
    print("Estimated differences for selected pairs:")
    print(json.dumps({"pairs": pair_estimates}, separators=(',', ':')))
    answer_04 = {"pairs": pair_estimates}
    print("ANSWER APACHE-04", json.dumps(answer_04, separators=(',', ':')))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
