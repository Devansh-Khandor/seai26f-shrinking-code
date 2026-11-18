#!/usr/bin/env python3
"""
skill_07.py
------------
Command line tool that reads a CSV file of NASA software project data
and answers two new requests:

    NASA-13: Help me identify promising combinations of project characteristics
             that historically delivered more software with lower development effort
             and fewer defects.
    NASA-14: When detailed project evaluations are expensive, recommend which
             project proposals should be evaluated first to discover promising
             alternatives with a limited evaluation budget.

Usage:
    python skill_07.py <csv path>
"""

import csv
import json
import sys
import math
import statistics

import numpy as np

from skill_02 import load_csv, parse_float


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def identify_columns(rows, goal_cols):
    """
    Return lists of numeric and categorical feature names.
    Exclude goal columns and columns ending with 'X'.
    """
    header = list(rows[0].keys())
    # filter out goal and X columns
    feature_names = [c for c in header if c not in goal_cols and not c.endswith("X")]
    numeric = []
    categorical = []
    for col in feature_names:
        is_num = False
        for r in rows:
            val = r[col]
            if val == "?":
                continue
            if parse_float(val) is not None:
                is_num = True
                break
        if is_num:
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def mean_by_category(rows, col, target_col):
    """
    For each distinct value of `col`, compute the mean of `target_col`
    over rows where both are present.
    """
    sums = {}
    counts = {}
    for r in rows:
        val = r[col]
        if val == "?":
            continue
        target = r[target_col]
        if target == "?":
            continue
        t = parse_float(target)
        if t is None:
            continue
        sums[val] = sums.get(val, 0.0) + t
        counts[val] = counts.get(val, 0) + 1
    means = {v: sums[v] / counts[v] for v in sums}
    return means


def rule_from_categorical(rows, categorical, goal_cols):
    """
    Choose one categorical feature/value pair that maximizes
    (mean Kloc+ - mean Effort- - mean Defects-).
    Return (col, value, count_of_matching_rows).
    """
    best_score = None
    best_col = None
    best_val = None
    best_count = 0

    for col in categorical:
        # compute means for each value
        mean_k = mean_by_category(rows, col, "Kloc+")
        mean_e = mean_by_category(rows, col, "Effort-")
        mean_d = mean_by_category(rows, col, "Defects-")
        for val in mean_k.keys():
            # values must exist in all three dicts
            if val in mean_e and val in mean_d:
                score = mean_k[val] - mean_e[val] - mean_d[val]
                if best_score is None or score > best_score:
                    best_score = score
                    best_col = col
                    best_val = val
    if best_col is None:
        # fallback: use first categorical column, first value
        if categorical:
            best_col = categorical[0]
            best_val = next(iter(mean_by_category(rows, best_col, "Kloc+")))
    # count matching rows
    count = sum(1 for r in rows if r[best_col] == best_val)
    return best_col, best_val, count


def compute_nasa13(rows):
    goal_cols = ["Kloc+", "Effort-", "Defects-"]
    _, categorical = identify_columns(rows, goal_cols)
    col, val, count = rule_from_categorical(rows, categorical, goal_cols)
    return {"rule": {col: val}, "matching": count}


def compute_nasa14(rows):
    """
    Return order of rows to evaluate first.
    Order is determined by a static score computed from goal values.
    """
    # filter rows with all goals present
    scored = []
    for idx, r in enumerate(rows, start=1):
        k = parse_float(r["Kloc+"])
        e = parse_float(r["Effort-"])
        d = parse_float(r["Defects-"])
        if k is None or e is None or d is None:
            continue
        # simple score: higher Kloc+, lower effort, lower defects
        score = k - e - d
        scored.append((score, idx))
    # sort descending
    scored.sort(key=lambda x: x[0], reverse=True)
    # budget
    budget = min(30, len(rows) // 5)
    order = [idx for _, idx in scored[:budget]]
    return order


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)

    # NASA-13
    print("== NASA-13 ==")
    ans_13 = compute_nasa13(rows)
    # print at least three lines
    print(f"Rule: {json.dumps(ans_13['rule'], separators=(',', ':'))}")
    print(f"Matching rows: {ans_13['matching']}")
    print("Example of matched rows: ", end="")
    # show first 3 matching row numbers
    rule_col, rule_val = next(iter(ans_13['rule'].items()))
    matched_rows = [i + 1 for i, r in enumerate(rows) if r[rule_col] == rule_val][:3]
    print(matched_rows)
    print(f"ANSWER NASA-13 {json.dumps(ans_13, separators=(',', ':'))}")

    # NASA-14
    print("\n== NASA-14 ==")
    order = compute_nasa14(rows)
    print(f"Budget: {len(order)} rows")
    print("First 5 rows to evaluate:", order[:5])
    print(f"ANSWER NASA-14 {json.dumps({'order': order}, separators=(',', ':'))}")


if __name__ == "__main__":
    main()
