#!/usr/bin/env python3
"""
skill_09.py

Entry point for APACHE-17 and APACHE-18 requests.

Usage:
    python skill_09.py <csv_path>
"""

import csv
import sys
import json
import math
from collections import defaultdict

# --------------------------------------------------------------------------- #
# Shared utilities (copied from earlier skills)

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
        for i, raw in enumerate(reader, start=1):  # row numbers start at 1
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
# Helper functions

def hamming_distance(row1, row2, cols):
    """Count differing values over specified columns."""
    dist = 0
    for c in cols:
        v1 = row1.get(c)
        v2 = row2.get(c)
        if v1 != v2:
            dist += 1
    return dist

def complete_row(row, cols):
    """Return True if all cols have non‑None values in the row."""
    return all(row.get(c) is not None for c in cols)

# --------------------------------------------------------------------------- #
# APACHE-17: unexpected performance

def compute_expected(row_num, row, history, input_cols, target_col, k=3):
    """
    Find the k nearest historical rows (by hamming distance) and
    return the mean performance- of those neighbors.
    """
    if not complete_row(row, input_cols):
        return None

    distances = []
    for hnum, hrow in history:
        if not complete_row(hrow, input_cols):
            continue
        if hnum == row_num:
            continue
        d = hamming_distance(row, hrow, input_cols)
        distances.append((d, hnum, hrow))

    if not distances:
        return None

    distances.sort(key=lambda x: x[0])
    selected = distances[:k]
    vals = [hrow.get(target_col) for _, _, hrow in selected if hrow.get(target_col) is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)

def flagged_unexpected(rows, history, input_cols, target_col, max_flags=20):
    """Return list of flagged rows for APACHE-17."""
    residuals = []
    flags = []

    for rnum, row in rows:
        actual = row.get(target_col)
        if actual is None:
            continue
        expected = compute_expected(rnum, row, history, input_cols, target_col)
        if expected is None:
            continue
        resid = actual - expected
        residuals.append(resid)
        flags.append((rnum, actual, expected, resid))

    # Compute statistics on residuals
    if not residuals:
        return []

    mean_res = sum(residuals) / len(residuals)
    var = sum((x - mean_res) ** 2 for x in residuals) / len(residuals)
    std_res = math.sqrt(var)

    # Flag those with |residual - mean| > 2*std
    flagged = []
    for rnum, actual, expected, resid in flags:
        if std_res == 0:
            continue
        if abs(resid - mean_res) > 2 * std_res:
            flagged.append({
                "row": rnum,
                "column": target_col,
                "actual": actual,
                "expected": round(expected, 3)
            })

    flagged.sort(key=lambda x: abs(x["actual"] - x["expected"]), reverse=True)
    return flagged[:max_flags]

# --------------------------------------------------------------------------- #
# APACHE-18: unusual setting combinations

def compute_distance_stats(rows, history, input_cols):
    """Return list of (row_num, distance) for all rows (excluding those without data)."""
    distances = []
    for rnum, row in rows:
        if not complete_row(row, input_cols):
            continue
        # Find nearest history row
        best_dist = None
        for hnum, hrow in history:
            if not complete_row(hrow, input_cols):
                continue
            if rnum == hnum:
                continue
            d = hamming_distance(row, hrow, input_cols)
            if best_dist is None or d < best_dist:
                best_dist = d
        if best_dist is not None:
            distances.append((rnum, best_dist))
    return distances

def flagged_unusual_combinations(rows, history, input_cols, max_flags=20):
    """Return list of flagged row numbers for APACHE-18 with unusualness scores."""
    distances = compute_distance_stats(rows, history, input_cols)
    if not distances:
        return [], {}

    # Compute mean and std of distances
    ds = [d for _, d in distances]
    mean_d = sum(ds) / len(ds)
    var = sum((x - mean_d) ** 2 for x in ds) / len(ds)
    std_d = math.sqrt(var) if var > 0 else 0.0

    flagged = []
    scores = {}
    for rnum, d in distances:
        if std_d == 0:
            continue
        z = (d - mean_d) / std_d
        if z > 2:  # threshold
            flagged.append(rnum)
            scores[str(rnum)] = round(z, 3)

    flagged.sort(key=lambda r: scores[str(r)], reverse=True)
    return flagged[:max_flags], scores

# --------------------------------------------------------------------------- #
# Main

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)
    goal_cols, input_cols = identify_columns(rows)
    target_col = 'Performance-'

    # Determine history vs arrivals
    total_rows = len(rows)
    arrivals_count = round(0.2 * total_rows)
    history_rows = rows[:total_rows - arrivals_count]
    arrivals_rows = rows[total_rows - arrivals_count:]

    # ------------------------------------------------------------------ APACHE-17
    print("== APACHE-17 ==")

    flagged17 = flagged_unexpected(rows, history_rows, input_cols, target_col, max_flags=20)
    for f in flagged17:
        print(f"Row {f['row']}: actual={f['actual']} expected={f['expected']}")

    answer_17 = {"flagged": flagged17}
    print("\nANSWER APACHE-17", json.dumps(answer_17))

    # ------------------------------------------------------------------ APACHE-18
    print("\n== APACHE-18 ==")

    flagged18, scores = flagged_unusual_combinations(rows, history_rows, input_cols, max_flags=20)
    print("Unusual rows:", flagged18)
    for r in flagged18:
        print(f"Row {r} score={scores[str(r)]}")

    answer_18 = {"flagged": flagged18, "scores": scores}
    print("\nANSWER APACHE-18", json.dumps(answer_18))

if __name__ == "__main__":
    main()
