#!/usr/bin/env python3
"""
skill_08.py

Entry point for APACHE-15 and APACHE-16 requests.

Usage:
    python skill_08.py <csv_path>
"""

import csv
import sys
import json
import math
from collections import defaultdict
from itertools import combinations

# --------------------------------------------------------------------------- #
# Shared utilities (copy from earlier skills)

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
# Helper functions for selecting poor rows

def select_rows_by_performance(rows, goal_col, percentage=0.10, min_rows=5, max_rows=20, reverse=True):
    """Return row numbers of worst (reverse=True) or best (reverse=False) performance rows."""
    filtered = [(rnum, row) for rnum, row in rows if row.get(goal_col) is not None]
    sorted_rows = sorted(filtered, key=lambda x: x[1][goal_col], reverse=reverse)
    total = len(sorted_rows)
    n_sel = max(min_rows, math.ceil(percentage * total))
    n_sel = min(n_sel, max_rows)
    selected = sorted_rows[:n_sel]
    return [rnum for rnum, _ in selected]


# --------------------------------------------------------------------------- #
# Hamming distance over input columns

def hamming_distance(row1, row2, cols):
    """Count differing values over specified columns."""
    dist = 0
    for c in cols:
        v1 = row1.get(c)
        v2 = row2.get(c)
        if v1 != v2:
            dist += 1
    return dist


# --------------------------------------------------------------------------- #
# Main logic

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)
    goal_cols, input_cols = identify_columns(rows)
    target_col = 'Performance-'

    # Find the set of poorly performing rows
    poor_rows = select_rows_by_performance(rows, target_col, reverse=True)
    # Pick the first poor row as the target
    if not poor_rows:
        print("== APACHE-15 ==")
        print("No poor rows found.")
        print("\nANSWER APACHE-15 {}",)
        print("\n== APACHE-16 ==")
        print("No poor rows found.")
        print("\nANSWER APACHE-16 {}",)
        return

    poor_row_num = poor_rows[0]
    poor_row = rows[poor_row_num - 1][1]
    poor_perf = poor_row.get(target_col)

    # ------------------------------------------------------------------ APACHE-15
    print("== APACHE-15 ==")

    # Find alternatives differing in 1 or 2 settings with lower performance
    best_alt = None
    best_diff = None
    best_perf = None

    for rnum, row in rows:
        if rnum == poor_row_num:
            continue
        alt_perf = row.get(target_col)
        if alt_perf is None:
            continue
        if alt_perf >= poor_perf:
            continue
        diff = hamming_distance(poor_row, row, input_cols)
        if diff not in (1, 2):
            continue
        if best_alt is None or diff < best_diff or (diff == best_diff and alt_perf < best_perf):
            best_alt = (rnum, row)
            best_diff = diff
            best_perf = alt_perf

    if best_alt is None:
        # Fall back: no alternative found
        alt_row_num = poor_row_num
        changes = {}
        after_perf = poor_perf
    else:
        alt_row_num, alt_row = best_alt
        # Determine changes
        changes = {}
        for col in input_cols:
            v1 = poor_row.get(col)
            v2 = alt_row.get(col)
            if v1 != v2:
                changes[col] = [v1, v2]
        after_perf = alt_row.get(target_col)

    # Print results
    print(f"Poor row: {poor_row_num}")
    print(f"Alternative row: {alt_row_num}")
    print(f"Changes: {json.dumps(changes)}")
    print(f"Performance before: {poor_perf}")
    print(f"Performance after: {after_perf}")

    answer_15 = {
        "row": poor_row_num,
        "alternative": alt_row_num,
        "changes": changes,
        "performance": {"before": poor_perf, "after": after_perf}
    }
    print("\nANSWER APACHE-15", json.dumps(answer_15))

    # ------------------------------------------------------------------ APACHE-16
    print("\n== APACHE-16 ==")

    # Target: best (lowest) observed performance
    target_perf = min(row.get(target_col) for _, row in rows if row.get(target_col) is not None)

    # Find alternatives closer to target than poor row, with minimal changes
    best_alt2 = None
    best_diff2 = None
    best_perf2 = None

    for rnum, row in rows:
        if rnum == poor_row_num:
            continue
        alt_perf = row.get(target_col)
        if alt_perf is None:
            continue
        if alt_perf >= poor_perf:   # need better than poor row
            continue
        diff = hamming_distance(poor_row, row, input_cols)
        if best_alt2 is None or diff < best_diff2 or (diff == best_diff2 and alt_perf < best_perf2):
            best_alt2 = (rnum, row)
            best_diff2 = diff
            best_perf2 = alt_perf

    if best_alt2 is None:
        alt_row_num2 = poor_row_num
        changed = []
    else:
        alt_row_num2, alt_row2 = best_alt2
        changed = [col for col in input_cols if poor_row.get(col) != alt_row2.get(col)]

    print(f"Poor row: {poor_row_num}")
    print(f"Target performance: {target_perf}")
    print(f"Alternative row: {alt_row_num2}")
    print(f"Changed settings: {changed}")

    answer_16 = {
        "row": poor_row_num,
        "target": target_perf,
        "alternative": alt_row_num2,
        "changed": changed
    }
    print("\nANSWER APACHE-16", json.dumps(answer_16))

if __name__ == "__main__":
    main()
