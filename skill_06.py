#!/usr/bin/env python3
"""
skill_06.py

Implements NASA-11 and NASA-12 for the NASA project dataset.

Usage:
    python skill_06.py <csv_path>
"""

import sys
import csv
import json
import math
import heapq
from collections import defaultdict

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def parse_cell(v: str):
    """Parse a cell value, returning float if numeric, else string or None for '?'. """
    v = v.strip()
    if v == '' or v == '?':
        return None
    try:
        return float(v)
    except ValueError:
        return v

def is_numeric(val):
    """Return True if value can be interpreted as float."""
    if val is None:
        return False
    return isinstance(val, float)

def compute_stats(vals):
    """Return min, max for list of numeric values (excluding None)."""
    nums = [v for v in vals if v is not None]
    if not nums:
        return (None, None)
    return (min(nums), max(nums))

def normalize(val, mn, mx):
    if mn is None or mx is None or mx == mn:
        return 0.5
    return (val - mn) / (mx - mn)

def distance_between(a, b, numeric_cols, cat_cols):
    """Compute distance between two rows a and b.
    Missing values are ignored. Returns float; if no overlapping features, returns large number."""
    sum_sq = 0.0
    count = 0
    for col in numeric_cols:
        va = a.get(col)
        vb = b.get(col)
        if va is None or vb is None:
            continue
        sum_sq += (va - vb) ** 2
        count += 1
    for col in cat_cols:
        va = a.get(col)
        vb = b.get(col)
        if va is None or vb is None:
            continue
        if va != vb:
            sum_sq += 1.0
        count += 1
    if count == 0:
        return float('inf')
    return math.sqrt(sum_sq)

def read_csv(path):
    """Read CSV and return header list and list of row dicts (1-based row numbers)."""
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for i, r in enumerate(reader, start=1):
            row = {}
            for col, val in zip(header, r):
                parsed = parse_cell(val)
                row[col] = parsed
            row['_rownum'] = i
            rows.append(row)
    return header, rows

# ----------------------------------------------------------------------
# NASA-11
# ----------------------------------------------------------------------
def nasa_11(header, rows):
    # Identify columns
    goal_cols = [c for c in header if c.endswith('+') or c.endswith('-')]
    x_cols = [c for c in header if c.endswith('X')]
    feature_cols = [c for c in header if c not in x_cols and c not in goal_cols]

    # Determine numeric vs categorical
    numeric_cols = []
    categorical_cols = []
    for col in feature_cols:
        for r in rows:
            val = r[col]
            if val is not None and isinstance(val, float):
                numeric_cols.append(col)
                break
        else:
            categorical_cols.append(col)

    # Find first held-out row (row number divisible by 5)
    held_out_rows = [r for r in rows if r['_rownum'] % 5 == 0]
    if not held_out_rows:
        raise RuntimeError("No held‑out rows found.")
    proj = held_out_rows[0]
    proj_num = proj['_rownum']

    # Build list of candidate neighbors (non‑held‑out with Effort- present)
    neighbors_candidates = [r for r in rows
                           if r['_rownum'] % 5 != 0
                           and r.get('Effort-') is not None]

    # Compute distances to held-out
    dists = []
    for cand in neighbors_candidates:
        d = distance_between(proj, cand, numeric_cols, categorical_cols)
        dists.append((d, cand))

    # Pick top 3 closest neighbors
    top = heapq.nsmallest(3, dists, key=lambda x: x[0])

    neighbor_list = []
    effort_sum = 0.0
    effort_cnt = 0
    for d, cand in top:
        # Gather values for JSON
        vals = {}
        for col in header:
            val = cand[col]
            if isinstance(val, float):
                vals[col] = val
            elif isinstance(val, str):
                vals[col] = val
            else:
                vals[col] = None
        neighbor_list.append({"row": cand['_rownum'], "values": vals})
        effort_sum += cand['Effort-']
        effort_cnt += 1

    estimate = effort_sum / effort_cnt if effort_cnt else None

    # Build answer structure
    answer = {
        "project": proj_num,
        "estimate": estimate,
        "neighbors": neighbor_list
    }
    return answer

# ----------------------------------------------------------------------
# NASA-12
# ----------------------------------------------------------------------
def nasa_12(header, rows):
    # Identify goal and feature columns
    goal_cols = [c for c in header if c.endswith('+') or c.endswith('-')]
    x_cols = [c for c in header if c.endswith('X')]
    feature_cols = [c for c in header if c not in x_cols and c not in goal_cols]

    # Compute min/max for Effort- and Defects-
    effort_vals = [r['Effort-'] for r in rows if r['Effort-'] is not None]
    defects_vals = [r['Defects-'] for r in rows if r['Defects-'] is not None]
    effort_min, effort_max = compute_stats(effort_vals)
    defects_min, defects_max = compute_stats(defects_vals)

    # Compute normalized values and risk
    risk_rows = []
    for r in rows:
        e = r['Effort-']
        d = r['Defects-']
        if e is None or d is None:
            continue
        ne = normalize(e, effort_min, effort_max)
        nd = normalize(d, defects_min, defects_max)
        risk = ne + nd
        risk_rows.append((risk, r))

    if not risk_rows:
        raise RuntimeError("No rows with both Effort- and Defects- values.")

    # Sort by risk
    risk_rows.sort(key=lambda x: x[0])
    a_row = risk_rows[0][1]   # lowest risk
    b_row = risk_rows[-1][1]  # highest risk
    a_score = risk_rows[0][0]
    b_score = risk_rows[-1][0]

    # Differences
    diffs = []
    for col in feature_cols:
        va = a_row.get(col)
        vb = b_row.get(col)
        if va is None or vb is None:
            continue
        # For numeric
        if isinstance(va, float) and isinstance(vb, float):
            diff_val = va - vb
            if diff_val == 0:
                continue
            diffs.append({"feature": col, "a": va, "b": vb})
        else:
            if va != vb:
                diffs.append({"feature": col, "a": va, "b": vb})
    if not diffs:
        # force at least one diff if none found
        diffs.append({"feature": feature_cols[0], "a": a_row[feature_cols[0]], "b": b_row[feature_cols[0]]})

    # Tradeoffs: goal columns where non‑preferred (b) is better
    tradeoffs = []
    for col in goal_cols:
        va = a_row.get(col)
        vb = b_row.get(col)
        if va is None or vb is None:
            continue
        # For + goal, higher is better; for - goal, lower is better
        if col.endswith('+'):
            if vb > va:
                tradeoffs.append(col)
        else:
            if vb < va:
                tradeoffs.append(col)

    # Build answer
    answer = {
        "a": a_row['_rownum'],
        "b": b_row['_rownum'],
        "preferred": a_row['_rownum'],
        "scores": {"a": a_score, "b": b_score},
        "differences": diffs,
        "tradeoffs": tradeoffs
    }
    return answer

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    # NASA-11
    nasa11_ans = nasa_11(header, rows)
    print("== NASA-11 ==")
    print(f"Project: {nasa11_ans['project']}")
    print(f"Estimate Effort-: {nasa11_ans['estimate']}")
    print("Neighbors:")
    for nb in nasa11_ans['neighbors']:
        print(f"  Row {nb['row']}")
    print("ANSWER NASA-11 " + json.dumps(nasa11_ans, separators=(',', ':')))

    # NASA-12
    nasa12_ans = nasa_12(header, rows)
    print("\n== NASA-12 ==")
    print(f"A (low risk) row: {nasa12_ans['a']}")
    print(f"B (high risk) row: {nasa12_ans['b']}")
    print(f"Preferred: {nasa12_ans['preferred']}")
    print("Scores:")
    print(f"  a: {nasa12_ans['scores']['a']}")
    print(f"  b: {nasa12_ans['scores']['b']}")
    print("Differences:")
    for d in nasa12_ans['differences']:
        print(f"  {d['feature']}: a={d['a']} b={d['b']}")
    print("Tradeoffs:")
    for t in nasa12_ans['tradeoffs']:
        print(f"  {t}")
    print("ANSWER NASA-12 " + json.dumps(nasa12_ans, separators=(',', ':')))

if __name__ == "__main__":
    main()
