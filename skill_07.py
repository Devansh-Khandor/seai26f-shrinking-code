#!/usr/bin/env python3
"""
skill_07.py <csv_path>

Performs two NASA analysis requests on the provided CSV file:
NASA-13 – Identify promising combinations of project characteristics that historically
          delivered more software with lower development effort and fewer defects.
NASA-14 – When detailed project evaluations are expensive, recommend which project
          proposals should be evaluated first to discover promising alternatives
          with a limited evaluation budget.

The script prints each request with the required header, at least three lines of
human‑readable results, and a single‑line machine‑checked JSON answer.
"""

import csv
import json
import sys
from statistics import mean, median
from math import isnan

# --------------------------------------------------------------------------- #
# Helper functions (copied from previous skills)
# --------------------------------------------------------------------------- #

def parse_value(v):
    """Parse a cell value; return float or None for missing."""
    v = v.strip()
    if v == '?' or v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None

def normalize(v, vmin, vmax, best_first=True):
    """Normalize a value to [0,1]. If best_first is True, 1 is best; else 0 is best."""
    if vmax == vmin or vmin is None or vmax is None:
        return 0.5
    if best_first:
        return (v - vmin) / (vmax - vmin)
    else:
        return (vmax - v) / (vmax - vmin)

# --------------------------------------------------------------------------- #
# CSV loader
# --------------------------------------------------------------------------- #

def load_rows(csv_path):
    """Read CSV and return list of parsed rows (dict) with row numbers."""
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        all_columns = reader.fieldnames
        for idx, raw in enumerate(reader, start=1):
            parsed = {}
            for col in all_columns:
                if col.endswith('X'):      # non‑input
                    parsed[col] = None
                else:
                    parsed[col] = parse_value(raw[col])
            parsed['row_num'] = idx
            rows.append(parsed)
    return rows, all_columns

# --------------------------------------------------------------------------- #
# NASA-13
# --------------------------------------------------------------------------- #

def nasa_13(rows, all_columns):
    """Identify promising rule based on historical data."""
    # Identify goal columns
    goal_cols = [c for c in all_columns if (c.endswith('+') or c.endswith('-')) and not c.endswith('X')]

    # Separate goal values
    def vals(col):
        return [r[col] for r in rows if r[col] is not None]

    # Thresholds: top 25% Kloc+, bottom 25% Effort- and Defects-
    thresholds = {}
    for col in goal_cols:
        vlist = vals(col)
        if not vlist:
            thresholds[col] = None
            continue
        vlist.sort()
        if col.endswith('+'):  # maximize
            thresholds[col] = vlist[int(0.75 * len(vlist))]
        else:  # minimize
            thresholds[col] = vlist[int(0.25 * len(vlist))]

    # Filter rows that satisfy thresholds
    good_rows = []
    for r in rows:
        ok = True
        for col in goal_cols:
            val = r[col]
            if val is None or thresholds[col] is None:
                ok = False
                break
            if col.endswith('+'):
                if val < thresholds[col]:
                    ok = False
                    break
            else:
                if val > thresholds[col]:
                    ok = False
                    break
        if ok:
            good_rows.append(r)

    matching = len(good_rows)

    # Find rule conditions from non‑goal columns
    non_goal_cols = [c for c in all_columns
                     if not c.endswith('X') and not (c.endswith('+') or c.endswith('-'))]
    rule = {}

    # Helper to determine numeric vs categorical
    def is_numeric(col):
        return all(isinstance(r[col], float) for r in rows if r[col] is not None)

    # For numeric columns: compute range in good rows
    numeric_ranges = {}
    for col in non_goal_cols:
        if is_numeric(col):
            vals_good = [r[col] for r in good_rows if r[col] is not None]
            if vals_good:
                numeric_ranges[col] = (min(vals_good), max(vals_good))
    # Choose up to 3 numeric columns with the tightest range relative to overall range
    numeric_candidates = []
    for col, (low, high) in numeric_ranges.items():
        overall = [r[col] for r in rows if r[col] is not None]
        overall_low, overall_high = min(overall), max(overall)
        rel_range = (high - low) / (overall_high - overall_low) if overall_high != overall_low else 0
        numeric_candidates.append((rel_range, col, low, high))
    numeric_candidates.sort()
    for _, col, low, high in numeric_candidates[:3]:
        rule[col] = [low, high]

    # For categorical columns: mode
    for col in non_goal_cols:
        if not is_numeric(col):
            vals_good = [r[col] for r in good_rows if r[col] is not None]
            if vals_good:
                counts = {}
                for v in vals_good:
                    counts[v] = counts.get(v, 0) + 1
                mode_val = max(counts.items(), key=lambda kv: kv[1])[0]
                rule[col] = mode_val

    # Keep only 1–3 conditions
    if len(rule) > 3:
        # prioritize numeric columns with tightest ranges first
        ordered = list(rule.items())
        ordered.sort(key=lambda kv: (0 if isinstance(kv[1], list) else 1, kv[0]))
        rule = dict(ordered[:3])

    return {"rule": rule, "matching": matching}

# --------------------------------------------------------------------------- #
# NASA-14
# --------------------------------------------------------------------------- #

def nasa_14(rows, all_columns):
    """Recommend evaluation order under a limited budget using only goal values."""
    # Identify goal columns
    goal_cols = [c for c in all_columns if (c.endswith('+') or c.endswith('-')) and not c.endswith('X')]

    # Compute min/max per goal
    mins = {}
    maxs = {}
    for col in goal_cols:
        vals = [r[col] for r in rows if r[col] is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
        else:
            mins[col] = maxs[col] = None

    # Scoring: higher Kloc+, lower Effort-, lower Defects-
    def score_row(r):
        s = 0.0
        for col in goal_cols:
            val = r[col]
            if val is None:
                return None  # cannot score
            if col.endswith('+'):
                norm = normalize(val, mins[col], maxs[col], best_first=True)
            else:
                norm = normalize(val, mins[col], maxs[col], best_first=False)
            s += norm
        return s

    # Compute scores for rows with all goals present
    scored = [(r['row_num'], score_row(r)) for r in rows]
    scored = [(rn, sc) for rn, sc in scored if sc is not None]
    # Sort by descending score
    scored.sort(key=lambda x: -x[1])
    # Determine budget
    n = len(rows)
    budget = min(30, n // 5)
    # Select top budget rows
    order = [rn for rn, _ in scored[:budget]]
    return {"order": order}

# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main(csv_path):
    rows, all_columns = load_rows(csv_path)

    # NASA-13
    print("== NASA-13 ==")
    result_13 = nasa_13(rows, all_columns)
    print("Rule conditions:")
    for k, v in result_13["rule"].items():
        if isinstance(v, list):
            print(f"  {k}: [{v[0]:.2f}, {v[1]:.2f}]")
        else:
            print(f"  {k}: {v}")
    print(f"Number of rows matching rule: {result_13['matching']}")
    print("ANSWER NASA-13", json.dumps(result_13, ensure_ascii=False))

    # NASA-14
    print("\n== NASA-14 ==")
    result_14 = nasa_14(rows, all_columns)
    print(f"Evaluation budget: {min(30, len(rows)//5)} rows")
    print("First 5 rows to evaluate:")
    print("  " + ", ".join(map(str, result_14["order"][:5])))
    print("Full order:")
    print("  " + ", ".join(map(str, result_14["order"])))
    print("ANSWER NASA-14", json.dumps(result_14, ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
