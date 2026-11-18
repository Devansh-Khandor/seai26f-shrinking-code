#!/usr/bin/env python3
"""
Apache configuration monitoring and summary updating.

Usage:  python skill_04.py <csv_path>
"""

import csv
import json
import math
import sys

# ---------------------------------------------------------------------------

def read_csv(csv_path):
    """Read CSV and return list of row dicts and fieldnames."""
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows, reader.fieldnames

# ---------------------------------------------------------------------------

def to_float(val):
    """Convert string to float or return None if missing."""
    if val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

# ---------------------------------------------------------------------------

def find_goal_columns(fieldnames):
    """Return list of columns ending with '+' or '-'."""
    return [c for c in fieldnames if c.endswith('+') or c.endswith('-')]

# ---------------------------------------------------------------------------

def mean_std(values):
    """Return mean and population standard deviation of a list of floats."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = sum(values) / n
    var = sum((x - m) ** 2 for x in values) / n
    return m, math.sqrt(var)

# ---------------------------------------------------------------------------

def compute_stats(values):
    """Return dict of n, mean, min, max."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    m = sum(values) / n
    return {"n": n, "mean": m, "min": min(values), "max": max(values)}

# ---------------------------------------------------------------------------

def main(csv_path):
    rows, fieldnames = read_csv(csv_path)
    total_rows = len(rows)

    # Identify goal columns
    goal_cols = find_goal_columns(fieldnames)
    if not goal_cols:
        print("No goal columns found", file=sys.stderr)
        sys.exit(1)
    # For this data, we assume one goal column
    goal_col = goal_cols[0]

    # Precompute goal values per row
    goal_values = [to_float(r[goal_col]) for r in rows]

    # Determine arrivals
    arrivals_count = round(0.2 * total_rows)
    arrivals_start_idx = total_rows - arrivals_count  # 0-based index where arrivals start

    # ----------------- APACHE-07 -----------------
    print("== APACHE-07 ==")

    prior_values = []
    assessments = []

    for idx, (row, gval) in enumerate(zip(rows, goal_values), start=1):
        # check if this row is an arrival
        is_arrival = idx > arrivals_start_idx
        if is_arrival:
            # compute alert
            if len(prior_values) >= 2:
                mean_prior, std_prior = mean_std(prior_values)
                if std_prior > 0 and gval is not None and gval > mean_prior + 2 * std_prior:
                    alert = True
                else:
                    alert = False
            else:
                alert = False
            assessments.append({"row": idx, "alert": alert})
        # update prior list after processing (so arrival uses all before)
        if gval is not None:
            prior_values.append(gval)

    # Print at least three lines
    for a in assessments[:3]:
        print(f"Row {a['row']} alert={a['alert']}")
    # Output answer JSON
    print(f"ANSWER APACHE-07 {json.dumps({'assessments': assessments}, separators=(',', ':'))}")

    # ----------------- APACHE-08 -----------------
    print("\n== APACHE-08 ==")

    # Define checkpoints: after 0, 19, 38 arrivals
    checkpoints_arrival_counts = [0, 19, 38]
    checkpoints = []

    # Compute checkpoint row numbers (1-based)
    history_end_idx = arrivals_start_idx  # index of first arrival, 0-based
    checkpoint_rows = {m: history_end_idx + m for m in checkpoints_arrival_counts}

    # Prepare running stats
    current_goal_vals = []
    current_rows_with_goals = []  # list of tuples (row_num, goal_value)
    current_row_index = 0

    # We'll build checkpoints on the fly
    for idx, (row, gval) in enumerate(zip(rows, goal_values), start=1):
        # add current row to stats if goal present
        if gval is not None:
            current_goal_vals.append(gval)
            current_rows_with_goals.append((idx, gval))

        # If this index matches a checkpoint, capture summary and shortlist
        if idx in checkpoint_rows.values():
            # Determine m for this checkpoint
            m = [k for k, v in checkpoint_rows.items() if v == idx][0]
            # Compute summary for each goal column (there may be more than one)
            summary = {}
            for col in goal_cols:
                # Extract values for this column up to current idx
                col_vals = [to_float(r[col]) for r in rows[:idx] if to_float(r[col]) is not None]
                stats = compute_stats(col_vals)
                summary[col] = stats
            # Shortlist: best 5 rows so far based on goal_col
            best_rows = sorted(current_rows_with_goals, key=lambda x: (x[1], x[0]))[:5]
            shortlist = [row_num for row_num, _ in best_rows]
            checkpoints.append({
                "after": m,
                "summary": summary,
                "shortlist": shortlist
            })

    # Print at least three lines
    for cp in checkpoints[:3]:
        print(f"After {cp['after']} arrivals, best rows: {cp['shortlist']}")
    # Output answer JSON
    print(f"ANSWER APACHE-08 {json.dumps({'checkpoints': checkpoints}, separators=(',', ':'))}")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
