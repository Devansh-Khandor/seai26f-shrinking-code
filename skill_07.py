#!/usr/bin/env python3
"""
skill_07.py <csv_path>

Implements NASA-13 (identify promising characteristics) and NASA-14
(recommend evaluation order under a limited budget).

The program prints for each request a header line `== <ID> ==`,
followed by at least three lines of textual output, and finally a
machine‑checked answer line in the form

    ANSWER <ID> <json>

The JSON structures are defined in the problem statement.
"""

import sys
import csv
import json
import math
from statistics import mean

# --------------------------------------------------------------------
# CSV utilities
# --------------------------------------------------------------------
def parse_csv(path):
    """Read CSV into list of dict rows (header omitted)."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def to_float(val):
    """Convert cell to float, return None for missing."""
    if val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

# --------------------------------------------------------------------
# Helper functions for NASA‑13
# --------------------------------------------------------------------
def get_non_goal_cols(header):
    """Return columns that are not goals (+/-) and not X columns."""
    return [c for c in header
            if not c.endswith('+') and not c.endswith('-') and not c.endswith('X')]

def is_numeric(col, rows):
    """True if at least one row can be parsed as float for this column."""
    for r in rows:
        if to_float(r[col]) is not None:
            return True
    return False

def compute_metric_for_rows(rows, indices):
    """Return average of Kloc+ - Effort- - Defects- for rows at indices (1‑based)."""
    vals = []
    for i in indices:
        r = rows[i-1]
        k = to_float(r['Kloc+'])
        e = to_float(r['Effort-'])
        d = to_float(r['Defects-'])
        if k is None or e is None or d is None:
            continue
        vals.append(k - e - d)
    return mean(vals) if vals else -math.inf

def find_best_symbolic_rule(rows, non_goal_cols):
    """Return (col, value) that maximizes the metric."""
    best = None
    best_metric = -math.inf
    for col in non_goal_cols:
        if is_numeric(col, rows):
            continue  # skip numeric for now
        # categorical
        values = set(r[col] for r in rows if r[col] != '' and r[col] != '?')
        for val in values:
            indices = [i for i, r in enumerate(rows, start=1) if r[col] == val]
            if not indices:
                continue
            m = compute_metric_for_rows(rows, indices)
            if m > best_metric:
                best_metric = m
                best = (col, val, indices)
    return best

# --------------------------------------------------------------------
# Helper functions for NASA‑14
# --------------------------------------------------------------------
def score_row(r):
    """Return composite score Kloc+ - Effort- - Defects-; None if missing."""
    k = to_float(r['Kloc+'])
    e = to_float(r['Effort-'])
    d = to_float(r['Defects-'])
    if k is None or e is None or d is None:
        return None
    return k - e - d

# --------------------------------------------------------------------
# Main implementations
# --------------------------------------------------------------------
def nasa_13(rows):
    """Identify a rule on non‑goal columns that selects rows with high Kloc+
    and low Effort- / Defects-."""
    header = rows[0].keys()
    non_goal_cols = get_non_goal_cols(header)

    result = find_best_symbolic_rule(rows, non_goal_cols)

    print("== NASA-13 ==")
    if result is None:
        print("No suitable rule found.")
        print("ANSWER NASA-13", json.dumps({"rule": {}, "matching": 0}, separators=(',', ':')))
        return

    col, val, indices = result
    matching_count = len(indices)
    matched_rows = [rows[i-1] for i in indices]
    kloc_vals = [to_float(r['Kloc+']) for r in matched_rows if to_float(r['Kloc+']) is not None]
    effort_vals = [to_float(r['Effort-']) for r in matched_rows if to_float(r['Effort-']) is not None]
    defect_vals = [to_float(r['Defects-']) for r in matched_rows if to_float(r['Defects-']) is not None]

    print(f"Rule: {col} == \"{val}\"")
    print(f"Matching rows: {matching_count}")
    print(f"Avg Kloc+: {mean(kloc_vals):.2f}, Avg Effort-: {mean(effort_vals):.2f}, Avg Defects-: {mean(defect_vals):.2f}")

    # Machine‑checked answer
    answer = {"rule": {col: val}, "matching": matching_count}
    print("ANSWER NASA-13", json.dumps(answer, separators=(',', ':')))

def nasa_14(rows):
    """Recommend evaluation order within a limited budget."""
    total_rows = len(rows)
    budget = min(30, total_rows // 5)
    # Compute scores, keep only rows with complete goals
    scored = []
    for idx, r in enumerate(rows, start=1):
        s = score_row(r)
        if s is not None:
            scored.append((s, idx))
    # Sort by score descending, then by index ascending
    scored.sort(key=lambda x: (-x[0], x[1]))
    order = [idx for _, idx in scored[:budget]]

    print("== NASA-14 ==")
    print(f"Evaluation budget: {budget} rows")
    print(f"Top 5 rows to evaluate: {order[:5]}")
    if len(order) > 5:
        print(f"Next 5 rows: {order[5:10]}")
    print(f"Full order: {order}")

    # Machine‑checked answer
    answer = {"order": order}
    print("ANSWER NASA-14", json.dumps(answer, separators=(',', ':')))

# --------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_07.py <csv_path>\n")
        sys.exit(1)
    path = sys.argv[1]
    rows = parse_csv(path)
    nasa_13(rows)
    nasa_14(rows)

if __name__ == "__main__":
    main()
