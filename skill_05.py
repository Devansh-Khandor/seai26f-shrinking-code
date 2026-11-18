#!/usr/bin/env python3
"""
skill_05.py

Entry point for APACHE-09 and APACHE-10 requests.

Usage:
    python skill_05.py <csv_path>
"""

import csv
import sys
import json
import math
from collections import defaultdict
from itertools import combinations

# --------------------------------------------------------------------------- #
# Shared utilities

def parse_cell(val: str):
    """Parse a CSV cell value: convert to int or None if '?'."""
    val = val.strip()
    if val == "?":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def load_csv(path: str):
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


# --------------------------------------------------------------------------- #
# Helper: select worst or best rows

def select_rows_by_performance(rows, goal_col, percentage=0.10, min_rows=5, max_rows=20, reverse=False):
    """Return row numbers of worst (reverse=True) or best (reverse=False) performance rows."""
    filtered = [(rnum, row) for rnum, row in rows if row.get(goal_col) is not None]
    sorted_rows = sorted(filtered, key=lambda x: x[1][goal_col], reverse=reverse)
    total = len(sorted_rows)
    n_sel = max(min_rows, math.ceil(percentage * total))
    n_sel = min(n_sel, max_rows)
    selected = sorted_rows[:n_sel]
    return [rnum for rnum, _ in selected]


# --------------------------------------------------------------------------- #
# APACHE-09 logic

def compute_ranking(rows, bad_row_nums, input_cols):
    """Return top 3 columns most strongly associated with poor performance."""
    bad_rows = [row for rnum, row in rows if rnum in bad_row_nums]
    all_rows = [row for _, row in rows]
    # Compute average setting value (percentage of 1s) for bad rows and all rows
    bad_means = {}
    all_means = {}
    for col in input_cols:
        bad_vals = [row[col] for row in bad_rows if row[col] is not None]
        all_vals = [row[col] for row in all_rows if row[col] is not None]
        bad_means[col] = sum(bad_vals) / len(bad_vals) if bad_vals else 0.0
        all_means[col] = sum(all_vals) / len(all_vals) if all_vals else 0.0

    # Compute absolute difference
    diffs = [(col, abs(bad_means[col] - all_means[col])) for col in input_cols]
    diffs.sort(key=lambda x: x[1], reverse=True)
    # Return column names only
    return [col for col, _ in diffs[:3]]


# --------------------------------------------------------------------------- #
# APACHE-10 logic

def find_best_combination(rows, bad_rows, good_rows, input_cols):
    """Find one two‑setting combination that best distinguishes bad from good rows."""
    bad_set = set(bad_rows)
    good_set = set(good_rows)

    # Pre‑filter rows that have complete data for all input columns
    def valid_row(row):
        return all(row[col] is not None for col in input_cols)

    bad_rows_vals = [row for _, row in rows if _ in bad_set and valid_row(row)]
    good_rows_vals = [row for _, row in rows if _ in good_set and valid_row(row)]

    total_bad = len(bad_rows_vals)
    total_good = len(good_rows_vals)

    best = None
    best_diff = -1e9

    for col1, col2 in combinations(input_cols, 2):
        # Count occurrences of each 2‑bit pattern
        pattern_counts_bad = defaultdict(int)
        for row in bad_rows_vals:
            key = (row[col1], row[col2])
            pattern_counts_bad[key] += 1
        pattern_counts_good = defaultdict(int)
        for row in good_rows_vals:
            key = (row[col1], row[col2])
            pattern_counts_good[key] += 1

        # Evaluate each pattern present in bad rows
        for pattern, bad_cnt in pattern_counts_bad.items():
            good_cnt = pattern_counts_good.get(pattern, 0)
            bad_share = bad_cnt / total_bad if total_bad else 0.0
            good_share = good_cnt / total_good if total_good else 0.0
            diff = bad_share - good_share
            if diff > best_diff:
                best_diff = diff
                best = {
                    "settings": {col1: pattern[0], col2: pattern[1]},
                    "bad_share": bad_share,
                    "good_share": good_share
                }
    return best if best else None


# --------------------------------------------------------------------------- #
# Main

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)
    goal_cols, input_cols = identify_columns(rows)
    target_col = 'Performance-'

    # --------------------------------------------------- APACHE-09
    print("== APACHE-09 ==")

    # Identify poorly performing rows (worst 10%, at least 5)
    poor_row_nums = select_rows_by_performance(rows, target_col, reverse=True)
    print(f"Poor rows (worst 10%): {poor_row_nums}")

    # Ranking of settings
    ranking_cols = compute_ranking(rows, poor_row_nums, input_cols)
    print("Ranking of settings most associated with poor performance:")
    for col in ranking_cols:
        print(f"  {col}")

    answer_09 = {"poor": poor_row_nums, "ranking": ranking_cols}
    print("\nANSWER APACHE-09", json.dumps(answer_09))

    # --------------------------------------------------- APACHE-10
    print("\n== APACHE-10 ==")

    # Identify good rows (best 10%, at least 5)
    good_row_nums = select_rows_by_performance(rows, target_col, reverse=False)
    print(f"Good rows (best 10%): {good_row_nums}")

    # Find distinguishing combination
    combo = find_best_combination(rows, poor_row_nums, good_row_nums, input_cols)
    if combo is None:
        combos = []
    else:
        combos = [combo]

    print("Distinguishing combinations (at least 2 settings):")
    for c in combos:
        print(f"  Settings: {c['settings']}, bad_share: {c['bad_share']:.3f}, good_share: {c['good_share']:.3f}")

    answer_10 = {
        "bad": poor_row_nums,
        "good": good_row_nums,
        "combinations": combos
    }
    print("\nANSWER APACHE-10", json.dumps(answer_10))

if __name__ == "__main__":
    main()
