#!/usr/bin/env python3
"""
Skill 07: NASA-13 & NASA-14 project selection

Usage:
    python skill_07.py <path-to-nasa93dem.csv>
"""

import sys
import json
import numpy as np

# Reuse helper functions from skill_01
try:
    from skill_01 import read_csv, format_number
except ImportError:
    # Minimal reimplementation if skill_01 is not available
    import csv
    def read_csv(path):
        rows = []
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = {}
                for k, v in row.items():
                    if v.strip() == '?':
                        parsed[k] = None
                    else:
                        try:
                            parsed[k] = float(v)
                        except ValueError:
                            parsed[k] = v
                rows.append(parsed)
        return rows

    def format_number(x):
        if x is None:
            return "null"
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        return f"{x:.6f}".rstrip('0').rstrip('.')

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def is_numeric(value):
    return isinstance(value, float)

def get_non_goal_columns(rows):
    """Return columns that are not goals (+/-) and not ending with X."""
    cols = list(rows[0].keys())
    return [c for c in cols if not (c.endswith('+') or c.endswith('-')) and not c.endswith('X')]

def compute_composite_score(row):
    """Compute a score that rewards high Kloc+ and penalises Effort- and Defects-."""
    k = row.get('Kloc+')
    e = row.get('Effort-')
    d = row.get('Defects-')
    if k is None or e is None or d is None or e == 0 or d == 0:
        return -float('inf')
    return k / (e * d)

# ----------------------------------------------------------------------
# NASA-13: identify promising combinations of characteristics
# ----------------------------------------------------------------------
def nasa_13(rows):
    # 1. Select top 5 rows by composite score
    scored = [(idx, r, compute_composite_score(r)) for idx, r in enumerate(rows, start=1)
              if compute_composite_score(r) != -float('inf')]
    scored.sort(key=lambda t: t[2], reverse=True)
    top_rows = [s[1] for s in scored[:5]]

    # 2. Find common conditions
    non_goal_cols = get_non_goal_columns(rows)
    conditions = []

    for col in non_goal_cols:
        values = [r[col] for r in top_rows if r[col] is not None]
        if not values:
            continue
        # categorical
        if all(not is_numeric(v) for v in values):
            # if all equal, add condition
            if len(set(values)) == 1:
                conditions.append((col, values[0]))
        else:
            # numeric: compute inclusive range
            lows = [v for v in values if isinstance(v, float)]
            low, high = min(lows), max(lows)
            # if range is zero, treat as categorical
            if low == high:
                conditions.append((col, low))
            else:
                conditions.append((col, [low, high]))

    # Sort conditions by spread (numeric) or by value frequency (categorical)
    def spread(cond):
        _, val = cond
        if isinstance(val, list):
            return val[1] - val[0]
        return 0.0
    conditions.sort(key=spread)

    # Keep up to 3
    conditions = conditions[:3]

    # Build rule dict
    rule = {}
    for col, val in conditions:
        rule[col] = val

    # Count matching rows
    def matches(r):
        for col, val in rule.items():
            v = r.get(col)
            if v is None:
                return False
            if isinstance(val, list):
                low, high = val
                if not (low <= v <= high):
                    return False
            else:
                if v != val:
                    return False
        return True

    matching = sum(1 for r in rows if matches(r))

    # Print
    print("== NASA-13 ==")
    print(f"Rule conditions (up to 3): {rule}")
    print(f"Matching rows: {matching}")

    # JSON answer
    json_obj = {"rule": rule, "matching": matching}
    json_str = json.dumps(json_obj, separators=(',', ':'))
    print(f"ANSWER NASA-13 {json_str}")

# ----------------------------------------------------------------------
# NASA-14: recommend evaluation order
# ----------------------------------------------------------------------
def nasa_14(rows):
    # Budget: at most min(30, number of rows // 5)
    budget = min(30, len(rows) // 5)
    # Compute composite scores for all rows
    scored = [(idx, r, compute_composite_score(r)) for idx, r in enumerate(rows, start=1)
              if compute_composite_score(r) != -float('inf')]
    # Sort by descending score
    scored.sort(key=lambda t: t[2], reverse=True)
    # Take top 'budget' rows
    order = [idx for idx, _, _ in scored[:budget]]

    # Print
    print("== NASA-14 ==")
    print(f"Evaluation order (first {budget} projects): {order}")

    # JSON answer
    json_obj = {"order": order}
    json_str = json.dumps(json_obj, separators=(',', ':'))
    print(f"ANSWER NASA-14 {json_str}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)
    path = sys.argv[1]
    rows = read_csv(path)
    nasa_13(rows)
    nasa_14(rows)

if __name__ == "__main__":
    main()
