#!/usr/bin/env python3
"""
skill_07.py

Entry point for APACHE-13 and APACHE-14 requests.

Usage:
    python skill_07.py <csv_path>
"""

import csv
import sys
import json
import math
from typing import List, Tuple, Dict
from collections import defaultdict

# --------------------------------------------------------------------------- #
# Shared utilities (copy of earlier functions)

def parse_cell(val: str):
    """Parse a CSV cell value: convert to int or None if '?'."""
    val = val.strip()
    if val == "?":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def load_csv(path: str) -> List[Tuple[int, Dict[str, int]]]:
    """Load CSV, return list of (row_num, dict_of_columns)."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for i, raw in enumerate(reader, start=1):
            row = {}
            for k, v in raw.items():
                row[k] = parse_cell(v)
            rows.append((i, row))
    return rows


def identify_columns(rows):
    """Return goal_cols, input_cols as lists."""
    col_keys = rows[0][1].keys()
    goal_cols = [col for col in col_keys if col.endswith('+') or col.endswith('-')]
    input_cols = [col for col in col_keys if not (col.endswith('+') or col.endswith('-') or col.endswith('X'))]
    return goal_cols, input_cols


def build_feature_matrix(rows, input_cols, target_col):
    """Build X, y from rows where target_col is not None and all inputs are present."""
    X = []
    y = []
    for rnum, row in rows:
        if row.get(target_col) is None:
            continue
        feat = []
        missing = False
        for col in input_cols:
            val = row.get(col)
            if val is None:
                missing = True
                break
            feat.append(val)
        if missing:
            continue
        X.append(feat)
        y.append(row[target_col])
    return X, y


# --------------------------------------------------------------------------- #
# Main logic for adaptive evaluation

def adaptive_evaluation(rows, input_cols, target_col, budget=30):
    """
    Greedy strategy: start with 5 seeded rows, then iteratively
    train a linear model on evaluated rows and pick next row
    with lowest predicted target among un‑evaluated candidates.
    """
    # Prepare mapping of row number to row dict
    row_dict = {rnum: row for rnum, row in rows}

    # Find evaluatable rows (full features)
    evaluatable = [rnum for rnum, row in rows
                   if row.get(target_col) is not None and
                   all(row.get(col) is not None for col in input_cols)]

    # Seed with first 5 evaluatable rows (or fewer if not enough)
    seed = evaluatable[:5]
    evaluated = []
    evaluated_set = set()

    for rnum in seed:
        evaluated.append(rnum)
        evaluated_set.add(rnum)

    # Remaining candidates
    remaining = [r for r in evaluatable if r not in evaluated_set]

    # Iterative greedy loop
    while len(evaluated) < budget and remaining:
        # Train linear regression on evaluated data
        train_rows = [(rnum, row_dict[rnum]) for rnum in evaluated]
        X_train, y_train = build_feature_matrix(train_rows, input_cols, target_col)
        if X_train:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X_train, y_train)
            # Predict remaining
            preds = []
            for rnum in remaining:
                row = row_dict[rnum]
                feat = [row[col] for col in input_cols]
                pred = model.predict([feat])[0]
                preds.append((pred, rnum))
            # Choose row with lowest predicted value
            preds.sort()
            _, next_rnum = preds[0]
        else:
            # No model available, pick arbitrarily the first remaining
            next_rnum = remaining[0]

        evaluated.append(next_rnum)
        evaluated_set.add(next_rnum)
        remaining = [r for r in remaining if r not in evaluated_set]

    return evaluated


def find_best_row(rows, target_col, evaluated):
    """Return the row number with lowest actual target among evaluated."""
    best_val = None
    best_row = None
    for rnum in evaluated:
        val = rows[rnum-1][1].get(target_col)
        if val is None:
            continue
        if best_val is None or val < best_val:
            best_val = val
            best_row = rnum
    return best_row


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)
    goal_cols, input_cols = identify_columns(rows)
    target_col = 'Performance-'

    # ------------------------------------------------------------------ APACHE-13
    print("== APACHE-13 ==")

    evaluated_rows = adaptive_evaluation(rows, input_cols, target_col, budget=30)
    best_row = find_best_row(rows, target_col, evaluated_rows)

    # Some simple output lines
    print(f"Evaluated {len(evaluated_rows)} rows: {evaluated_rows[:5]} ...")
    print(f"Best evaluated row: {best_row}")

    answer_13 = {"evaluated": evaluated_rows, "best": best_row}
    print("\nANSWER APACHE-13", json.dumps(answer_13))

    # ------------------------------------------------------------------ APACHE-14
    print("\n== APACHE-14 ==")

    order = evaluated_rows  # same order as we would evaluate
    print(f"Recommended evaluation order (first 10): {order[:10]} ...")

    answer_14 = {"order": order}
    print("\nANSWER APACHE-14", json.dumps(answer_14))

if __name__ == "__main__":
    main()
