#!/usr/bin/env python3
"""
skill_04.py

Usage:
    python skill_04.py <csv path>

Adds two new analysis sections:
  * CAR-07: Assess each new arrival immediately to see if it belongs in the
    top‑5 shortlist of desirable vehicles.
  * CAR-08: Keep inventory summaries and the 5‑vehicle shortlist up‑to‑date
    as arrivals come in, printing checkpoints after 0, 40, and 80 arrivals.

The script prints each section header, a few descriptive lines, and a
machine‑checked ANSWER line in JSON format.
"""

import sys
import json
import numpy as np
import csv

# Reuse helpers from skill_01
from skill_01 import read_csv, parse_float, normalize


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def collect_values(rows, idx):
    """Return a numpy array of float values for column idx, skipping missing."""
    vals = []
    for r in rows:
        v = parse_float(r[idx])
        if v is not None:
            vals.append(v)
    return np.array(vals)


def compute_goal_stats(rows, idx):
    """Return dict with n, mean, min, max for column idx."""
    vals = collect_values(rows, idx)
    if vals.size == 0:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals))
    }


def compute_score(row, mins, maxs):
    """
    Compute combined performance score for a single row.
    Returns None if any goal is missing.
    """
    lbs = parse_float(row[mins['lbs_idx']])
    acc = parse_float(row[mins['acc_idx']])
    mpg = parse_float(row[mins['mpg_idx']])
    if lbs is None or acc is None or mpg is None:
        return None

    norm_lbs = normalize(lbs, mins['lbs_min'], maxs['lbs_max'])
    norm_acc = normalize(acc, mins['acc_min'], maxs['acc_max'])
    norm_mpg = normalize(mpg, mins['mpg_min'], maxs['mpg_max'])

    return (1 - norm_lbs) + (1 - norm_acc) + norm_mpg


def get_scoreable_rows(rows, idx_lbs, idx_acc, idx_mpg):
    """Return rows (index, row) tuples that have all three goal values."""
    result = []
    for i, r in enumerate(rows):
        if (parse_float(r[idx_lbs]) is not None and
                parse_float(r[idx_acc]) is not None and
                parse_float(r[idx_mpg]) is not None):
            result.append((i, r))
    return result


def top_n_rows(rows, n, idx_lbs, idx_acc, idx_mpg):
    """
    Return the list of 1‑based row numbers for the top n rows based on combined score.
    """
    # Determine overall min and max for scoring
    lbs_vals = collect_values(rows, idx_lbs)
    acc_vals = collect_values(rows, idx_acc)
    mpg_vals = collect_values(rows, idx_mpg)

    mins = {
        'lbs_idx': idx_lbs,
        'acc_idx': idx_acc,
        'mpg_idx': idx_mpg,
        'lbs_min': float(np.min(lbs_vals)),
        'acc_min': float(np.min(acc_vals)),
        'mpg_min': float(np.min(mpg_vals))
    }
    maxs = {
        'lbs_max': float(np.max(lbs_vals)),
        'acc_max': float(np.max(acc_vals)),
        'mpg_max': float(np.max(mpg_vals))
    }

    # Score all scoreable rows
    scored = []
    for idx, r in get_scoreable_rows(rows, idx_lbs, idx_acc, idx_mpg):
        sc = compute_score(r, mins, maxs)
        if sc is not None:
            scored.append((sc, idx + 1))  # 1‑based row number

    scored.sort(reverse=True, key=lambda x: x[0])
    return [rn for _, rn in scored[:n]]


def checkpoint_summary(rows, idx_lbs, idx_acc, idx_mpg):
    """
    Return dict mapping each goal column to its summary stats.
    """
    return {
        'Lbs-': compute_goal_stats(rows, idx_lbs),
        'Acc+': compute_goal_stats(rows, idx_acc),
        'Mpg+': compute_goal_stats(rows, idx_mpg)
    }


# --------------------------------------------------------------------------- #
# Main logic for CAR-07 and CAR-08
# --------------------------------------------------------------------------- #

def main(csv_path):
    header, data = read_csv(csv_path)
    n_rows = len(data)

    # Identify goal column indices
    idx_lbs = header.index('Lbs-')
    idx_acc = header.index('Acc+')
    idx_mpg = header.index('Mpg+')

    # Define arrivals
    arrivals_count = int(round(0.2 * n_rows))
    arrivals_start = n_rows - arrivals_count
    history_rows = data[:arrivals_start]
    arrival_rows = data[arrivals_start:]

    # -------------------------------------------------- CAR-07 --------------------------------------------------
    print("== CAR-07 ==")
    assessments = []

    # Keep a list of all rows seen so far (history + processed arrivals)
    seen_rows = history_rows.copy()

    for i, arr in enumerate(arrival_rows):
        # Row number in full file
        row_num = arrivals_start + i + 1

        # Add arrival to seen set
        seen_rows.append(arr)

        # Compute top 5 among seen rows
        top5 = top_n_rows(seen_rows, 5, idx_lbs, idx_acc, idx_mpg)

        desirable = row_num in top5
        assessments.append({"row": row_num, "desirable": desirable})

        # Print a simple line for each assessment
        print(f"Row {row_num}: {'desirable' if desirable else 'not desirable'}")

    # Machine‑checked answer
    answer_car07 = {"assessments": assessments}
    print(f"ANSWER CAR-07 {json.dumps(answer_car07, separators=(',', ':'), sort_keys=True)}")

    # -------------------------------------------------- CAR-08 --------------------------------------------------
    print("\n== CAR-08 ==")
    checkpoints = [0, 40, 80]
    checkpoint_results = []

    for m in checkpoints:
        # Determine rows up to m arrivals
        rows_up_to = history_rows + arrival_rows[:m]

        # Compute summary stats
        summary = checkpoint_summary(rows_up_to, idx_lbs, idx_acc, idx_mpg)

        # Determine 5‑vehicle shortlist
        shortlist = top_n_rows(rows_up_to, 5, idx_lbs, idx_acc, idx_mpg)

        # Print human‑readable summary
        print(f"\nAfter {m} arrivals:")
        for col in ['Lbs-', 'Acc+', 'Mpg+']:
            stats = summary[col]
            print(f"{col}: n={stats['n']} mean={stats['mean']:.3f} min={stats['min']:.1f} max={stats['max']:.1f}")

        print(f"Shortlist: {shortlist}")

        # Record for machine‑checked answer
        checkpoint_results.append({
            "after": m,
            "summary": {
                col: {
                    "n": summary[col]["n"],
                    "mean": summary[col]["mean"],
                    "min": summary[col]["min"],
                    "max": summary[col]["max"]
                } for col in ['Lbs-', 'Acc+', 'Mpg+']
            },
            "shortlist": shortlist
        })

    # Machine‑checked answer
    answer_car08 = {"checkpoints": checkpoint_results}
    print(f"ANSWER CAR-08 {json.dumps(answer_car08, separators=(',', ':'), sort_keys=True)}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
