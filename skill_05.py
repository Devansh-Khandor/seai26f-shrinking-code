#!/usr/bin/env python3
"""
Apache configuration analysis – poor vs. good performance and key setting combinations.

Usage:  python skill_05.py <csv_path>
"""

import csv
import json
import math
import sys
import itertools

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

def find_goal_column(fieldnames):
    """Return the column that ends with '+' or '-'."""
    for c in fieldnames:
        if c.endswith('+') or c.endswith('-'):
            return c
    return None

# ---------------------------------------------------------------------------

def get_non_goal_columns(fieldnames, goal_col):
    """Return list of columns that are inputs (not goal and not ending in 'X')."""
    return [c for c in fieldnames
            if c != goal_col and not c.endswith('X')]

# ---------------------------------------------------------------------------

def mean(values):
    if not values:
        return None
    return sum(values) / len(values)

# ---------------------------------------------------------------------------

def compute_poor_rows(rows, goal_col):
    """Return list of indices (1‑based) of poorly performing rows."""
    # collect goal values
    goals = [to_float(r[goal_col]) for r in rows]
    # filter out missing
    valid_goals = [g for g in goals if g is not None]
    if not valid_goals:
        return []  # no valid data

    m = mean(valid_goals)
    std = math.sqrt(mean([(g - m)**2 for g in valid_goals]))

    # Rows that are > mean + std
    poor = [i+1 for i, g in enumerate(goals) if g is not None and g > m + std]

    # If fewer than 5, take worst 5 rows
    if len(poor) < 5:
        # sort all rows by goal descending (worst)
        all_rows = [(i+1, g) for i, g in enumerate(goals) if g is not None]
        all_rows.sort(key=lambda x: -x[1])  # worst first
        poor = [idx for idx, _ in all_rows[:5]]
    return poor

# ---------------------------------------------------------------------------

def compute_good_rows(rows, goal_col, k=5):
    """Return list of indices (1‑based) of best performing rows (lowest goal)."""
    goals = [to_float(r[goal_col]) for r in rows]
    all_rows = [(i+1, g) for i, g in enumerate(goals) if g is not None]
    all_rows.sort(key=lambda x: x[1])  # best first
    good = [idx for idx, _ in all_rows[:k]]
    return good

# ---------------------------------------------------------------------------

def ranking_of_settings(rows, goal_col, non_goal_cols, poor_indices):
    """Return list of setting names sorted by association with poor performance."""
    # compute mean goal for 1 vs 0 for each setting
    associations = []
    for col in non_goal_cols:
        values1 = []
        values0 = []
        for idx in poor_indices:
            r = rows[idx-1]
            goal_val = to_float(r[goal_col])
            if goal_val is None:
                continue
            v = r[col]
            if v == '?' or v == '':
                continue
            if v == '1':
                values1.append(goal_val)
            elif v == '0':
                values0.append(goal_val)
        if not values1 and not values0:
            continue
        mean1 = mean(values1) if values1 else None
        mean0 = mean(values0) if values0 else None
        if mean1 is None or mean0 is None:
            diff = None
        else:
            diff = mean1 - mean0  # positive means 1 worse
        associations.append((col, diff))
    # Sort by absolute diff descending, keep those with diff not None
    associations = [(c,d) for c,d in associations if d is not None]
    associations.sort(key=lambda x: abs(x[1]), reverse=True)
    ranked = [c for c,_ in associations[:3]]  # at least 3
    return ranked

# ---------------------------------------------------------------------------

def find_common_combinations(rows, bad_indices, good_indices, non_goal_cols):
    """Find at least one combination of two settings most common in bad rows."""
    bad_counts = {}
    good_counts = {}
    # iterate over pairs of columns
    for col1, col2 in itertools.combinations(non_goal_cols, 2):
        bad_dict = {}
        good_dict = {}
        # bad rows
        for idx in bad_indices:
            r = rows[idx-1]
            v1 = r[col1]
            v2 = r[col2]
            if v1 in ('?', '') or v2 in ('?', ''):
                continue
            key = (v1, v2)
            bad_dict[key] = bad_dict.get(key, 0) + 1
        # good rows
        for idx in good_indices:
            r = rows[idx-1]
            v1 = r[col1]
            v2 = r[col2]
            if v1 in ('?', '') or v2 in ('?', ''):
                continue
            key = (v1, v2)
            good_dict[key] = good_dict.get(key, 0) + 1
        # combine keys
        for key in set(bad_dict.keys()).union(good_dict.keys()):
            bad_counts[(col1, col2, key)] = bad_dict.get(key, 0)
            good_counts[(col1, col2, key)] = good_dict.get(key, 0)
    # Find the pair/key with highest bad fraction
    best = None
    best_bad_frac = -1.0
    for key_tuple, bad_count in bad_counts.items():
        col1, col2, key = key_tuple
        bad_frac = bad_count / len(bad_indices) if bad_indices else 0.0
        if bad_frac > best_bad_frac:
            best = key_tuple
            best_bad_frac = bad_frac
    if best is None:
        return []
    col1, col2, key = best
    bad_frac = bad_counts[best] / len(bad_indices) if bad_indices else 0.0
    good_frac = good_counts[best] / len(good_indices) if good_indices else 0.0
    combo = {
        "settings": {col1: int(key[0]), col2: int(key[1])},
        "bad_share": bad_frac,
        "good_share": good_frac
    }
    return [combo]

# ---------------------------------------------------------------------------

def main(csv_path):
    rows, fieldnames = read_csv(csv_path)
    goal_col = find_goal_column(fieldnames)
    if goal_col is None:
        print("No goal column found", file=sys.stderr)
        sys.exit(1)

    non_goal_cols = get_non_goal_columns(fieldnames, goal_col)

    # ----------------------------------
    # APACHE-09
    # ----------------------------------
    print("== APACHE-09 ==")

    poor_indices = compute_poor_rows(rows, goal_col)
    ranking = ranking_of_settings(rows, goal_col, non_goal_cols, poor_indices)

    # Print at least three lines
    print(f"Poor rows: {poor_indices}")
    print(f"Top associated settings: {ranking}")

    answer_09 = {"poor": poor_indices, "ranking": ranking}
    print(f"ANSWER APACHE-09 {json.dumps(answer_09, separators=(',', ':'))}")

    # ----------------------------------
    # APACHE-10
    # ----------------------------------
    print("\n== APACHE-10 ==")

    good_indices = compute_good_rows(rows, goal_col, k=5)

    combinations = find_common_combinations(rows, poor_indices, good_indices, non_goal_cols)

    # Print at least three lines
    print(f"Good rows: {good_indices}")
    print(f"Bad rows: {poor_indices}")
    print(f"Top combinations: {combinations}")

    answer_10 = {"bad": poor_indices, "good": good_indices, "combinations": combinations}
    print(f"ANSWER APACHE-10 {json.dumps(answer_10, separators=(',', ':'))}")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
