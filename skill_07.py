#!/usr/bin/env python3
"""
skill_07.py – NASA-13 & NASA-14

This script analyses a CSV file containing software project data and
produces two reports:

* NASA-13 – Identify promising combinations of project characteristics that
  historically delivered more software (Kloc+) with lower development effort
  (Effort-) and fewer defects (Defects-).

* NASA-14 – Recommend which project proposals should be evaluated first
  to discover promising alternatives when the evaluation budget is limited.

The program prints, for each request, a header line `== <ID> ==`,
at least three human‑readable lines, and then a machine‑checked JSON answer
line starting with `ANSWER <ID>`.  All computations are derived from the
data in the supplied CSV file; nothing is hard‑coded.

Usage:  python skill_07.py <csv_path>
"""

import csv
import json
import math
import sys

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def read_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def to_float(val):
    if val is None:
        return None
    v = val.strip()
    if v == '' or v == '?':
        return None
    try:
        return float(v)
    except ValueError:
        return None

def is_numeric(values):
    """Return True if all non‑None values can be converted to float."""
    for v in values:
        if v is None:
            continue
        try:
            float(v)
        except ValueError:
            return False
    return True

# ----------------------------------------------------------------------
# Core analysis
# ----------------------------------------------------------------------
def identify_columns(rows):
    """Return goal columns and non‑goal columns."""
    header = rows[0].keys()
    goal_cols = [c for c in header if c.endswith(('+', '-'))]
    non_goal_cols = [c for c in header
                     if not c.endswith(('+', '-')) and not c.endswith('X')]
    return goal_cols, non_goal_cols

def compute_min_max_for_goals(rows, goal_cols):
    """Return dict mapping goal column to (min, max) ignoring None."""
    mins, maxs = {}, {}
    for col in goal_cols:
        vals = [to_float(r[col]) for r in rows]
        filtered = [v for v in vals if v is not None]
        if not filtered:
            mins[col] = maxs[col] = 0.0
        else:
            mins[col] = min(filtered)
            maxs[col] = max(filtered)
    return mins, maxs

def normalize(val, mn, mx):
    if val is None or mn == mx:
        return None
    return (val - mn) / (mx - mn)

def compute_composite_score(rows, goal_cols, mins, maxs):
    """Return list of tuples (row_index, score)."""
    scores = []
    for idx, r in enumerate(rows, start=1):
        v_k = to_float(r.get('Kloc+', None))
        v_e = to_float(r.get('Effort-', None))
        v_d = to_float(r.get('Defects-', None))
        if v_k is None or v_e is None or v_d is None:
            continue  # skip rows with missing any goal
        nk = normalize(v_k, mins['Kloc+'], maxs['Kloc+'])
        ne = normalize(v_e, mins['Effort-'], maxs['Effort-'])
        nd = normalize(v_d, mins['Defects-'], maxs['Defects-'])
        # Composite: want high Kloc+, low Effort-, low Defects-
        score = nk + (1 - ne) + (1 - nd)
        scores.append((idx, score))
    return scores

def get_top_promising_rows(rows, scores, fraction=0.20, min_count=5):
    """Return list of row indices that are top by score."""
    # Sort by score descending
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    n = max(min_count, int(len(rows) * fraction))
    n = min(n, len(sorted_scores))
    top_indices = [idx for idx, _ in sorted_scores[:n]]
    return top_indices

def derive_rules(rows, promising_indices, non_goal_cols):
    """Derive up to 3 simple conditions from promising rows."""
    # Gather promising rows
    prom_rows = [rows[i-1] for i in promising_indices]
    all_rows = rows

    condition_scores = []

    for col in non_goal_cols:
        # collect values from promising rows
        prom_vals = [r[col] for r in prom_rows]
        # filter missing
        prom_vals_nonnull = [v for v in prom_vals if v is not None and v.strip() != '?']

        if not prom_vals_nonnull:
            continue

        # Check if numeric
        numeric = True
        for v in prom_vals_nonnull:
            try:
                float(v)
            except ValueError:
                numeric = False
                break

        if numeric:
            # numeric range
            nums = [float(v) for v in prom_vals_nonnull]
            low = min(nums)
            high = max(nums)
            # compute proportion of all rows within this range
            all_vals = [r[col] for r in all_rows]
            count_in_range = 0
            total = 0
            for v in all_vals:
                if v is None or v.strip() == '?':
                    continue
                try:
                    num = float(v)
                except ValueError:
                    continue
                total += 1
                if low <= num <= high:
                    count_in_range += 1
            proportion = count_in_range / total if total else 0
            condition_scores.append((proportion, col, [low, high]))
        else:
            # categorical: mode
            counts = {}
            for v in prom_vals_nonnull:
                stripped = v.strip()
                counts[stripped] = counts.get(stripped, 0) + 1
            # mode value
            mode_val = max(counts.items(), key=lambda x: x[1])[0]
            # proportion of all rows equal to mode
            all_vals = [r[col] for r in all_rows]
            total = 0
            match = 0
            for v in all_vals:
                if v is None or v.strip() == '?':
                    continue
                total += 1
                if v.strip() == mode_val:
                    match += 1
            proportion = match / total if total else 0
            condition_scores.append((proportion, col, mode_val))

    # Pick up to 3 with highest proportion
    condition_scores.sort(key=lambda x: x[0], reverse=True)
    selected = condition_scores[:3]

    rule = {}
    for _, col, val in selected:
        rule[col] = val
    return rule

def count_matching_rows(rows, rule):
    """Count rows that satisfy all conditions in rule."""
    count = 0
    for r in rows:
        ok = True
        for col, cond in rule.items():
            val = r.get(col, None)
            if val is None or val.strip() == '?':
                ok = False
                break
            # numeric check
            try:
                num_val = float(val)
                is_num = True
            except ValueError:
                is_num = False
            if isinstance(cond, list):
                # numeric range
                try:
                    low, high = cond
                    if not (low <= num_val <= high):
                        ok = False
                        break
                except (TypeError, ValueError):
                    ok = False
                    break
            else:
                # categorical
                if val.strip() != cond:
                    ok = False
                    break
        if ok:
            count += 1
    return count

# ----------------------------------------------------------------------
# NASA-13
# ----------------------------------------------------------------------
def nasa13(rows):
    goal_cols, non_goal_cols = identify_columns(rows)
    mins, maxs = compute_min_max_for_goals(rows, goal_cols)
    scores = compute_composite_score(rows, goal_cols, mins, maxs)
    if not scores:
        return {"rule": {}, "matching": 0}
    top_indices = get_top_promising_rows(rows, scores)
    rule = derive_rules(rows, top_indices, non_goal_cols)
    matching = count_matching_rows(rows, rule)
    return {"rule": rule, "matching": matching}

# ----------------------------------------------------------------------
# NASA-14
# ----------------------------------------------------------------------
def nasa14(rows):
    goal_cols, _ = identify_columns(rows)
    mins, maxs = compute_min_max_for_goals(rows, goal_cols)
    scores = compute_composite_score(rows, goal_cols, mins, maxs)
    # Sort by score descending
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    n_rows = len(rows)
    budget = min(30, n_rows // 5)
    order = [idx for idx, _ in sorted_scores[:budget]]
    return order

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_csv(csv_path)
    if not rows:
        print("No data rows found.")
        sys.exit(1)

    # NASA-13
    print("== NASA-13 ==")
    res13 = nasa13(rows)
    # Human‑readable lines
    print(f"Derived rule with {len(res13['rule'])} condition(s):")
    for col, val in res13['rule'].items():
        if isinstance(val, list):
            print(f"  {col}: [{val[0]:.4f}, {val[1]:.4f}]")
        else:
            print(f"  {col}: {val}")
    print(f"Rows matching rule: {res13['matching']}")
    print("ANSWER NASA-13", json.dumps(res13, separators=(',', ':')))

    # NASA-14
    print("\n== NASA-14 ==")
    order = nasa14(rows)
    print(f"Evaluation order (top {len(order)} rows):")
    print("  " + ", ".join(str(r) for r in order))
    print("ANSWER NASA-14", json.dumps({"order": order}, separators=(',', ':')))

if __name__ == "__main__":
    main()
