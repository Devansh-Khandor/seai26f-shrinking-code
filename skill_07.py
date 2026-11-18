#!/usr/bin/env python3
"""
skill_07.py: NASA-13 and NASA-14 analyses.
Usage:  python skill_07.py <csv path>
"""

import csv
import json
import math
import sys

# Import helper functions from earlier skills
from skill_01 import parse_float, median, std_dev


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def percentile(values, p):
    """Return the pth percentile (0-100) of a non-empty list of numbers."""
    n = len(values)
    if n == 0:
        return None
    values = sorted(values)
    k = (n - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    d0 = values[int(f)] * (c - k)
    d1 = values[int(c)] * (k - f)
    return d0 + d1


# ----------------------------------------------------------------------
# NASA-13: identify promising combinations of characteristics
# ----------------------------------------------------------------------
def nasa13(rows, header):
    # Determine goal values
    kloc_vals = []
    effort_vals = []
    defect_vals = []

    for r in rows:
        k = parse_float(r.get("Kloc+"))
        e = parse_float(r.get("Effort-"))
        d = parse_float(r.get("Defects-"))
        if k is not None:
            kloc_vals.append(k)
        if e is not None:
            effort_vals.append(e)
        if d is not None:
            defect_vals.append(d)

    if not kloc_vals or not effort_vals or not defect_vals:
        # Fall back if missing data
        print("== NASA-13 ==")
        print("Insufficient goal data to identify promising rows.")
        print("ANSWER NASA-13 {}")
        return

    # Percentile thresholds: top 30% Kloc+, bottom 30% Effort- and Defects-
    kloc_thresh = percentile(kloc_vals, 70)   # 70th percentile
    effort_thresh = percentile(effort_vals, 30)  # 30th percentile
    defect_thresh = percentile(defect_vals, 30)

    # Select promising rows
    promising = []
    for idx, r in enumerate(rows, start=1):
        k = parse_float(r.get("Kloc+"))
        e = parse_float(r.get("Effort-"))
        d = parse_float(r.get("Defects-"))
        if k is None or e is None or d is None:
            continue
        if k >= kloc_thresh and e <= effort_thresh and d <= defect_thresh:
            promising.append((idx, r))

    matching_count = len(promising)

    # Non‑goal columns
    non_goal_cols = [
        c for c in header
        if not (c.endswith('+') or c.endswith('-') or c.endswith('X'))
    ]

    # For each non‑goal column, compute statistics within promising rows
    col_stats = {}
    for col in non_goal_cols:
        vals = [parse_float(r.get(col)) for _, r in promising]
        # Determine if numeric
        numeric_vals = [v for v in vals if v is not None]
        if numeric_vals:
            # Numeric: range
            col_stats[col] = {
                "type": "numeric",
                "min": min(numeric_vals),
                "max": max(numeric_vals),
                "range": max(numeric_vals) - min(numeric_vals),
                "count": len(numeric_vals)
            }
        else:
            # Categorical: most common value
            cat_vals = [v for v in vals if v is not None]
            if not cat_vals:
                continue
            freq = {}
            for v in cat_vals:
                freq[v] = freq.get(v, 0) + 1
            most_common_val, freq_count = max(freq.items(), key=lambda x: x[1])
            col_stats[col] = {
                "type": "categorical",
                "value": most_common_val,
                "count": freq_count,
                "total": len(cat_vals)
            }

    # Rank columns: categorical by frequency proportion, numeric by inverse range
    ranked = []
    for col, st in col_stats.items():
        if st["type"] == "categorical":
            prop = st["count"] / st["total"] if st["total"] else 0
            ranked.append((prop, col, st))
        else:
            # Avoid division by zero
            rng = st["range"]
            inv_range = 1 / rng if rng > 0 else float('inf')
            ranked.append((inv_range, col, st))
    ranked.sort(reverse=True, key=lambda x: x[0])

    # Build rule with up to 3 conditions
    rule = {}
    for _, col, st in ranked[:3]:
        if st["type"] == "categorical":
            rule[col] = st["value"]
        else:
            # numeric inclusive range
            rule[col] = [round(st["min"], 4), round(st["max"], 4)]

    # Output
    print("== NASA-13 ==")
    print(f"Promising rows found: {matching_count}")
    print("Rule conditions:")
    for k, v in rule.items():
        print(f"  {k} : {v}")
    print("ANSWER NASA-13", json.dumps({"rule": rule, "matching": matching_count}))


# ----------------------------------------------------------------------
# NASA-14: recommend evaluation order
# ----------------------------------------------------------------------
def nasa14(rows, header, budget):
    # Compute score for each row (Kloc+ - Effort- - Defects-)
    scored_rows = []
    for idx, r in enumerate(rows, start=1):
        k = parse_float(r.get("Kloc+"))
        e = parse_float(r.get("Effort-"))
        d = parse_float(r.get("Defects-"))
        if k is None or e is None or d is None:
            continue
        score = k - e - d
        scored_rows.append((score, idx))

    scored_rows.sort(reverse=True, key=lambda x: x[0])
    order = [idx for _, idx in scored_rows[:budget]]

    print("== NASA-14 ==")
    print(f"Evaluation budget: {budget}")
    print("Top rows to evaluate first:")
    for i, rnum in enumerate(order, start=1):
        print(f"  {i}. Row {rnum}")
    print("ANSWER NASA-14", json.dumps({"order": order}))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path):
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        for idx, r in enumerate(reader, start=1):
            rows.append(r)

    # Determine budget for NASA-14
    num_rows = len(rows)
    budget = min(30, num_rows // 5)

    nasa13(rows, header)
    print()
    nasa14(rows, header, budget)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv path>")
        sys.exit(1)
    main(sys.argv[1])
