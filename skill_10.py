#!/usr/bin/env python3
"""
skill_10.py

Entry point for APACHE-19 and APACHE-20 skill.

Usage:
    python skill_10.py <csv_path>
"""

import csv
import json
import sys
import random
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Utilities (copying from earlier skills)
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
                        row[col] = None
            row['_row_num'] = idx
            rows.append(row)
    return rows, headers

def identify_goal_column(headers: List[str]) -> str:
    """Return the first column name that ends with '-' or '+'."""
    for col in headers:
        if col.endswith('-') or col.endswith('+'):
            return col
    raise ValueError("No goal column found.")

# --------------------------------------------------------------------------- #
# APACHE-19: configuration selection comparison
# --------------------------------------------------------------------------- #

def best_row_in_set(row_nums: List[int], goal_col: str, rows: List[Dict[str, Any]]) -> int:
    """Return the row number with the best goal value among given rows."""
    best_num = None
    best_val = None
    for rn in row_nums:
        r = rows[rn-1]
        val = r[goal_col]
        if val is None:
            continue
        if best_val is None or (goal_col.endswith('-') and val < best_val) or \
                (goal_col.endswith('+') and val > best_val):
            best_val = val
            best_num = rn
    return best_num

def run_apache_19(rows: List[Dict[str, Any]], goal_col: str,
                  budget: int, repeats: int = 20) -> Dict[str, Any]:
    random.seed(42)  # deterministic
    all_row_nums = list(range(1, len(rows)+1))

    # Our method: sorted by goal ascending (min) or descending (max)
    sorted_rows = sorted(rows, key=lambda r: r[goal_col])
    ours_best_rows = []
    random_best_rows = []

    for _ in range(repeats):
        # Our best from first 'budget' rows in sorted order
        our_selection = [r['_row_num'] for r in sorted_rows[:budget]]
        ours_best = best_row_in_set(our_selection, goal_col, rows)
        ours_best_rows.append(ours_best)

        # Random selection
        random_selection = random.sample(all_row_nums, budget)
        random_best = best_row_in_set(random_selection, goal_col, rows)
        random_best_rows.append(random_best)

    # Determine verdict
    ours_count = 0
    random_count = 0
    for o, r in zip(ours_best_rows, random_best_rows):
        o_val = rows[o-1][goal_col]
        r_val = rows[r-1][goal_col]
        if goal_col.endswith('-'):
            if o_val < r_val:
                ours_count += 1
            elif r_val < o_val:
                random_count += 1
        else:
            if o_val > r_val:
                ours_count += 1
            elif r_val > o_val:
                random_count += 1
    if ours_count > random_count:
        verdict = "ours better"
    elif random_count > ours_count:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    return {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best_rows,
        "random": random_best_rows,
        "verdict": verdict
    }

# --------------------------------------------------------------------------- #
# APACHE-20: prediction comparison
# --------------------------------------------------------------------------- #

def compute_baseline_mean(rows: List[Dict[str, Any]], goal_col: str,
                          heldout_indices: List[int]) -> float:
    values = [rows[i][goal_col] for i in range(len(rows))
              if (i+1) not in heldout_indices and rows[i][goal_col] is not None]
    return sum(values) / len(values) if values else 0.0

def mae_of_predictions(preds: Dict[str, float],
                       rows: List[Dict[str, Any]],
                       goal_col: str) -> float:
    errors = []
    for rn_str, pred in preds.items():
        rn = int(rn_str)
        actual = rows[rn-1][goal_col]
        if actual is not None:
            errors.append(abs(actual - pred))
    return sum(errors) / len(errors) if errors else 0.0

def run_apache_20(rows: List[Dict[str, Any]], goal_col: str) -> Dict[str, Any]:
    # Identify held-out rows (divisible by 5)
    heldout_indices = [i+1 for i in range(len(rows)) if (i+1) % 5 == 0]

    # Predictions using simple mean of training data
    training_rows = [r for r in rows if r['_row_num'] not in heldout_indices]
    training_values = [r[goal_col] for r in training_rows if r[goal_col] is not None]
    training_mean = sum(training_values) / len(training_values) if training_values else 0.0
    predictions = {str(r['_row_num']): training_mean for r in rows
                   if r['_row_num'] in heldout_indices}

    # Baseline mean (same as training mean for this simple estimator)
    baseline = training_mean

    mae_model = mae_of_predictions(predictions, rows, goal_col)
    mae_baseline = mae_of_predictions({str(r['_row_num']): baseline for r in rows
                                      if r['_row_num'] in heldout_indices}, rows, goal_col)

    # Verdict
    if mae_model < mae_baseline:
        verdict = "better"
    elif mae_baseline < mae_model:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    return {
        "predictions": predictions,
        "baseline": baseline,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict
    }

# --------------------------------------------------------------------------- #
# Main skill logic
# --------------------------------------------------------------------------- #

def main(csv_path: str) -> None:
    rows, headers = read_csv(csv_path)
    goal_col = identify_goal_column(headers)

    # --------------------------------------------------------------------- #
    # APACHE-19
    # --------------------------------------------------------------------- #
    budget = min(30, len(rows) // 5)
    result_19 = run_apache_19(rows, goal_col, budget)

    print("== APACHE-19 ==")
    print(f"Budget: {result_19['budget']}")
    print(f"Repeats: {result_19['repeats']}")
    print(f"Our best rows: {result_19['ours']}")
    print(f"Random best rows: {result_19['random']}")
    print(f"Verdict: {result_19['verdict']}")
    print("ANSWER APACHE-19", json.dumps(result_19, separators=(',', ':')))

    # --------------------------------------------------------------------- #
    # APACHE-20
    # --------------------------------------------------------------------- #
    result_20 = run_apache_20(rows, goal_col)

    print("\n== APACHE-20 ==")
    print(f"Number of held-out rows: {len([i for i in range(len(rows)) if (i+1)%5==0])}")
    print(f"Model MAE: {result_20['mae_model']:.4f}")
    print(f"Baseline MAE: {result_20['mae_baseline']:.4f}")
    print(f"Verdict: {result_20['verdict']}")
    print("ANSWER APACHE-20", json.dumps(result_20, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
