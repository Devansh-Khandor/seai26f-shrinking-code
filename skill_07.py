#!/usr/bin/env python3
"""
skill_07.py <csv path>

Implements:
    NASA-13: Identify promising combinations of project characteristics that
             historically delivered more software with lower development effort
             and fewer defects.

    NASA-14: With a limited evaluation budget, recommend the order in which to
             evaluate new project proposals to discover promising alternatives.

The script prints each request header, a few human‑readable result lines,
and then a machine‑checked JSON answer.
"""

import csv
import json
import math
import sys
from statistics import median
from collections import Counter

# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------
def to_float(v):
    """Convert a string to float or return None for '?'."""
    return None if v == '?' else float(v)

def is_numeric(v):
    """Return True if v can be parsed as a float."""
    if v == '?' or v is None:
        return False
    try:
        float(v)
        return True
    except ValueError:
        return False

# ------------------------------------------------------------------
# Core logic
# ------------------------------------------------------------------
def main(csv_path):
    # Load rows
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)

    total_rows = len(rows)

    # Identify goal and non‑goal columns
    goal_cols = [c for c in header if c.endswith('+') or c.endswith('-')]
    non_goal_cols = [c for c in header if not c.endswith('X') and c not in goal_cols]

    # Pre‑compute numeric vs categorical for non‑goal columns
    numeric_cols = set()
    for col in non_goal_cols:
        for row in rows:
            v = row[col]
            if v == '?':
                continue
            try:
                float(v)
                numeric_cols.add(col)
                break
            except ValueError:
                break  # non‑numeric

    # ------------------------------------------------------------------
    # NASA-13
    # ------------------------------------------------------------------
    # Step 1: compute joint score for each row (Kloc+ maximize, others minimize)
    goal_vals = {c: [] for c in goal_cols}
    for row in rows:
        for g in goal_cols:
            v = to_float(row[g])
            if v is not None:
                goal_vals[g].append(v)

    # Find min/max per goal
    goal_min = {}
    goal_max = {}
    for g in goal_cols:
        vals = goal_vals[g]
        if not vals:
            goal_min[g] = goal_max[g] = 0.0
        else:
            goal_min[g] = min(vals)
            goal_max[g] = max(vals)

    def normalize_value(g, v):
        mn = goal_min[g]
        mx = goal_max[g]
        if mx == mn:
            return 1.0
        if g.endswith('+'):  # maximize
            return (v - mn) / (mx - mn)
        else:  # minimize
            return (mx - v) / (mx - mn)

    scores = []
    for idx, row in enumerate(rows, start=1):
        # Need all three goals present
        missing = False
        s = 0.0
        for g in goal_cols:
            v = to_float(row[g])
            if v is None:
                missing = True
                break
            s += normalize_value(g, v)
        if missing:
            continue
        scores.append((s, idx))

    # Take top 10 rows
    scores.sort(reverse=True)
    top_n = min(10, len(scores))
    top_rows = [idx for _, idx in scores[:top_n]]

    # Build candidate conditions
    conditions = []

    for col in non_goal_cols:
        # gather values in top rows
        vals = []
        for r in top_rows:
            v = rows[r-1][col]
            if v == '?':
                continue
            vals.append(v)

        if not vals:
            continue

        if col in numeric_cols:
            low = min(float(v) for v in vals)
            high = max(float(v) for v in vals)
            # Support: number of rows where value within [low, high]
            support = sum(1 for row in rows
                          if row[col] != '?' and low <= float(row[col]) <= high)
            conditions.append((col, [low, high], support))
        else:
            # categorical: pick most common value among top rows
            most_common, freq = Counter(vals).most_common(1)[0]
            support = sum(1 for row in rows
                          if row[col] != '?' and row[col] == most_common)
            conditions.append((col, most_common, support))

    # Sort conditions by support descending, pick up to 3
    conditions.sort(key=lambda x: -x[2])
    chosen = conditions[:3]

    # Compute matching count (rows satisfying all chosen conditions)
    def row_matches(row):
        for col, val, _ in chosen:
            if row[col] == '?':
                return False
            if col in numeric_cols:
                num = float(row[col])
                low, high = val
                if num < low or num > high:
                    return False
            else:
                if row[col] != val:
                    return False
        return True

    matching = sum(1 for row in rows if row_matches(row))

    rule_dict = {}
    for col, val, _ in chosen:
        if col in numeric_cols:
            rule_dict[col] = [round(val[0], 2), round(val[1], 2)]
        else:
            rule_dict[col] = val

    nasa13_ans = {"rule": rule_dict, "matching": matching}

    # ------------------------------------------------------------------
    # NASA-14
    # ------------------------------------------------------------------
    # Compute min/max for each goal across all rows
    goal_min_all = {}
    goal_max_all = {}
    for g in goal_cols:
        vals = [to_float(row[g]) for row in rows if to_float(row[g]) is not None]
        if vals:
            goal_min_all[g] = min(vals)
            goal_max_all[g] = max(vals)
        else:
            goal_min_all[g] = goal_max_all[g] = 0.0

    # Compute score for each row that has all goals
    scored_rows = []
    for idx, row in enumerate(rows, start=1):
        missing = False
        score = 0.0
        for g in goal_cols:
            v = to_float(row[g])
            if v is None:
                missing = True
                break
            mn = goal_min_all[g]
            mx = goal_max_all[g]
            if mx == mn:
                norm = 1.0
            elif g.endswith('+'):
                norm = (v - mn) / (mx - mn)
            else:
                norm = (mx - v) / (mx - mn)
            score += norm
        if missing:
            continue
        scored_rows.append((score, idx))

    # Sort descending
    scored_rows.sort(reverse=True)

    # Budget: min(30, number of rows // 5)
    budget = min(30, total_rows // 5)
    order = [idx for _, idx in scored_rows[:budget]]

    nasa14_ans = {"order": order}

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    # NASA-13
    print("== NASA-13 ==")
    print(f"Top {top_n} rows by joint score: {top_rows}")
    print("Candidate conditions (column: value/range, support):")
    for col, val, sup in chosen:
        if isinstance(val, list):
            print(f"  {col}: [{val[0]:.2f}, {val[1]:.2f}] support={sup}")
        else:
            print(f"  {col}: {val} support={sup}")
    print(f"Matching rows count: {matching}")
    print(f"ANSWER NASA-13 {json.dumps(nasa13_ans)}")

    # NASA-14
    print("\n== NASA-14 ==")
    print(f"Evaluation budget (rows to evaluate): {budget}")
    print("Recommended order of evaluation (row numbers):")
    print(f"  {order}")
    print(f"ANSWER NASA-14 {json.dumps(nasa14_ans)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
