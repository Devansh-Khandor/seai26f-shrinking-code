#!/usr/bin/env python3
"""
Skill 07 – NASA-13 & NASA-14

NASA-13: Identify promising combinations of project characteristics that historically
delivered more software with lower development effort and fewer defects.

NASA-14: Recommend which project proposals should be evaluated first
to discover promising alternatives within a limited evaluation budget.
"""

import sys
import csv
import json
import math
import numpy as np

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def parse_float(val):
    """Parse a CSV cell into a float, handling missing data."""
    if val is None or val == '':
        return None
    val = val.strip()
    if val == '?' or val == '?':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def read_csv(path):
    """Return list of (row_idx, row_dict) tuples and fieldnames."""
    data = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            data.append((idx, row))
    return data, reader.fieldnames

def is_goal(col):
    return col.endswith('+') or col.endswith('-')

def is_excluded(col):
    return col.endswith('X')

def numeric_columns(fieldnames, data):
    """Return list of columns that are numeric across most rows."""
    nums = []
    for col in fieldnames:
        if is_goal(col) or is_excluded(col):
            continue
        # Check if at least 80% of non‑missing values can be parsed as float
        parsed = 0
        total = 0
        for _, row in data:
            val = row.get(col)
            f = parse_float(val)
            if f is not None:
                parsed += 1
            total += 1 if val is not None and val != '?' else 0
        if total > 0 and parsed / total >= 0.8:
            nums.append(col)
    return nums

# ----------------------------------------------------------------------
# NASA-13 implementation
# ----------------------------------------------------------------------
def nasa_13(data, fieldnames):
    print("== NASA-13 ==")

    # Identify promising rows: low Effort- and low Defects-
    effort_vals = [parse_float(row['Effort-']) for _, row in data if parse_float(row['Effort-']) is not None]
    defect_vals = [parse_float(row['Defects-']) for _, row in data if parse_float(row['Defects-']) is not None]
    if not effort_vals or not defect_vals:
        print("Insufficient data for NASA-13.")
        print("ANSWER NASA-13 {}")
        return

    eff_q1 = float(np.percentile(effort_vals, 25))
    def_q1 = float(np.percentile(defect_vals, 25))

    promising = set()
    for idx, row in data:
        eff = parse_float(row['Effort-'])
        defc = parse_float(row['Defects-'])
        if eff is not None and defc is not None and eff <= eff_q1 and defc <= def_q1:
            promising.add(idx)

    # Find the best single non‑goal column rule that captures many promising rows
    candidates = []
    non_goal_cols = [c for c in fieldnames if not is_goal(c) and not is_excluded(c)]

    for col in non_goal_cols:
        # Determine if column is numeric
        vals_prom = []
        for _, row in data:
            if row.get(col) is None or row.get(col) == '?':
                continue
            f = parse_float(row[col])
            if f is not None:
                vals_prom.append(f)
        if len(vals_prom) == 0:
            continue  # skip columns with no numeric values in any row

        # Check if column is numeric by seeing if values are numbers in a good fraction
        numeric = all(isinstance(v, (int, float)) for v in vals_prom)
        if not numeric:
            # treat as categorical
            # Find most common value among promising rows
            cat_counts = {}
            for idx, row in data:
                if idx not in promising:
                    continue
                val = row.get(col)
                if val is None or val == '?':
                    continue
                cat_counts[val] = cat_counts.get(val, 0) + 1
            if not cat_counts:
                continue
            best_val, best_cnt = max(cat_counts.items(), key=lambda x: x[1])

            # Count how many rows overall satisfy this value
            match_cnt = sum(1 for _, row in data if row.get(col) == best_val)
            candidates.append(('cat', col, best_val, match_cnt))
        else:
            # Numeric: use min/max over promising rows
            low = min(vals_prom)
            high = max(vals_prom)
            # Count how many rows overall within inclusive range
            match_cnt = sum(1 for _, row in data if parse_float(row[col]) is not None and low <= parse_float(row[col]) <= high)
            candidates.append(('num', col, (low, high), match_cnt))

    if not candidates:
        print("No suitable non‑goal columns found.")
        print("ANSWER NASA-13 {}")
        return

    # Pick candidate with highest match count
    best = max(candidates, key=lambda x: x[3])
    rule_type, col, rule_value, match_cnt = best

    # Prepare JSON rule
    if rule_type == 'cat':
        rule_obj = {col: rule_value}
    else:
        # round to 2 decimals
        low, high = rule_value
        rule_obj = {col: [round(low, 2), round(high, 2)]}

    # Print at least three lines
    print(f"Promising rows count: {len(promising)}")
    print(f"Selected rule: {rule_obj}")
    print(f"Rows matching rule: {match_cnt}")

    # Machine‑checked answer
    ans = {"rule": rule_obj, "matching": match_cnt}
    print(f"ANSWER NASA-13 {json.dumps(ans, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# NASA-14 implementation
# ----------------------------------------------------------------------
def nasa_14(data, fieldnames):
    print("\n== NASA-14 ==")

    # Budget: min(30, n_rows // 5)
    total_rows = len(data)
    budget = min(30, total_rows // 5)

    # Gather rows with both goal values present
    rows_with_goals = []
    for idx, row in data:
        eff = parse_float(row['Effort-'])
        defc = parse_float(row['Defects-'])
        if eff is not None and defc is not None:
            rows_with_goals.append((idx, eff, defc))
    if not rows_with_goals:
        print("No rows with complete goal values for NASA-14.")
        print(f"ANSWER NASA-14 {json.dumps({'order': []}, separators=(',', ':'))}")
        return

    # Normalize Effort- and Defects- (lower better)
    eff_vals = np.array([e for _, e, _ in rows_with_goals])
    def_vals = np.array([d for _, _, d in rows_with_goals])
    eff_min, eff_max = eff_vals.min(), eff_vals.max()
    def_min, def_max = def_vals.min(), def_vals.max()
    if eff_max == eff_min:
        eff_norm = np.zeros_like(eff_vals)
    else:
        eff_norm = (eff_vals - eff_min) / (eff_max - eff_min)
    if def_max == def_min:
        def_norm = np.zeros_like(def_vals)
    else:
        def_norm = (def_vals - def_min) / (def_max - def_min)

    # Score: lower sum of normalized values is better
    scores = eff_norm + def_norm
    ranked = sorted(zip(scores, rows_with_goals), key=lambda x: x[0])
    order = [idx for _, (idx, _, _) in ranked[:budget]]

    print(f"Evaluation budget: {budget}")
    print(f"First 5 rows to evaluate: {order[:5]}")
    print(f"Last 5 rows in order: {order[-5:]}")

    ans = {"order": order}
    print(f"ANSWER NASA-14 {json.dumps(ans, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv path>")
        sys.exit(1)
    path = sys.argv[1]
    data, fieldnames = read_csv(path)

    nasa_13(data, fieldnames)
    nasa_14(data, fieldnames)

if __name__ == "__main__":
    main()
