#!/usr/bin/env python3
"""
skill_07.py

Usage:
    python skill_07.py <csv_path>
"""

import csv
import json
import sys

import numpy as np
from sklearn.linear_model import LinearRegression


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def is_goal_col(col_name):
    """Goal columns end with '+' or '-'."""
    return col_name.endswith('+') or col_name.endswith('-')


def is_input_col(col_name):
    """Input columns are not goal columns and do not end with 'X'."""
    return not col_name.endswith('X') and not is_goal_col(col_name)


def parse_int(val):
    """Parse binary 0/1, missing '?' -> 0."""
    return 0 if val == '?' else int(val)


def parse_float(val):
    """Parse float or missing '?' -> None."""
    return None if val == '?' else float(val)


# --------------------------------------------------------------------------- #
# Core logic
# --------------------------------------------------------------------------- #
def generate_order(parsed_rows, goal_col, input_cols, budget):
    """
    Generate an evaluation order using a simple linear‑regression guided
    strategy.  Only rows with a non‑missing goal value are considered.
    """
    n_rows = len(parsed_rows)
    # Candidates that can be evaluated (have a goal value)
    candidates = [i + 1 for i in range(n_rows) if parsed_rows[i][goal_col] is not None]

    selected = []

    while len(selected) < budget and candidates:
        if len(selected) < 2:
            # No model yet – pick the smallest index
            next_idx = min(candidates)
        else:
            # Train linear regression on already selected rows
            X_train = np.array(
                [[parsed_rows[i - 1][c] for c in input_cols] for i in selected]
            )
            y_train = np.array([parsed_rows[i - 1][goal_col] for i in selected])
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict on remaining candidates
            best_val = None
            best_idx = None
            for cand in candidates:
                X_cand = np.array(
                    [parsed_rows[cand - 1][c] for c in input_cols]
                ).reshape(1, -1)
                pred = model.predict(X_cand)[0]
                if (
                    best_val is None
                    or pred < best_val
                    or (pred == best_val and cand < best_idx)
                ):
                    best_val = pred
                    best_idx = cand
            next_idx = best_idx

        selected.append(next_idx)
        candidates.remove(next_idx)

    return selected


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]

    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        raw_rows = list(reader)

    # Identify goal and input columns
    goal_cols = [c for c in fieldnames if is_goal_col(c)]
    if not goal_cols:
        print("No goal column found.", file=sys.stderr)
        sys.exit(1)

    # Pick the minimization target (ends with '-').  Fall back to first goal.
    goal_col = None
    for c in goal_cols:
        if c.endswith('-'):
            goal_col = c
            break
    if not goal_col:
        goal_col = goal_cols[0]

    input_cols = [c for c in fieldnames if is_input_col(c)]

    # Parse all rows
    parsed_rows = []
    for row in raw_rows:
        parsed = {}
        for col in fieldnames:
            if is_goal_col(col):
                parsed[col] = parse_float(row[col])
            else:
                parsed[col] = parse_int(row[col])
        parsed_rows.append(parsed)

    # Budget: at most 30 or number of rows // 5
    budget = min(30, len(parsed_rows) // 5)

    # Generate evaluation order
    order = generate_order(parsed_rows, goal_col, input_cols, budget)

    # ---- APACHE-13 ----
    evaluated_rows = order
    best_row = None
    best_value = None
    for idx in evaluated_rows:
        val = parsed_rows[idx - 1][goal_col]
        if val is not None:
            if best_row is None or val < best_value:
                best_row = idx
                best_value = val

    print("== APACHE-13 ==")
    print(f"Evaluated rows: {evaluated_rows}")
    print(f"Best row: {best_row} (performance {best_value if best_row is not None else None})")
    print(f"Number evaluated: {len(evaluated_rows)}")
    answer_13 = {"evaluated": evaluated_rows, "best": best_row}
    print("ANSWER APACHE-13", json.dumps(answer_13, separators=(',', ':')))

    # ---- APACHE-14 ----
    print("\n== APACHE-14 ==")
    print(f"Evaluation order (first to last): {order}")
    print(f"Total rows to evaluate: {len(order)}")
    answer_14 = {"order": order}
    print("ANSWER APACHE-14", json.dumps(answer_14, separators=(',', ':')))


if __name__ == "__main__":
    main()
