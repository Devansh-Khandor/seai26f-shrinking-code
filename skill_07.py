#!/usr/bin/env python3
"""
skill_07.py - Identify promising project characteristics and recommend
evaluation order for new proposals

Usage:  python skill_07.py <csv path>
"""

import json
import sys
import math
import ezr

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def read_csv(path):
    """Return header and list of rows."""
    rows = list(ezr.csv(path))
    header = rows[0]
    data = rows[1:]
    return header, data

def numeric(v):
    """Return True if value is int/float."""
    return isinstance(v, (int, float))

def get_goal_ranges(header, rows):
    """Return dict of goal column -> (min, max)."""
    goal_cols = [c for c in header if c.endswith(('+', '-'))]
    ranges = {}
    for c in goal_cols:
        idx = header.index(c)
        vals = [row[idx] for row in rows if row[idx] != "?"]
        if vals:
            mn = min(vals)
            mx = max(vals)
            ranges[c] = (mn, mx)
    return ranges

def normalize_value(col, val, mn, mx):
    """Return normalized 0..1 for a goal value."""
    if mx == mn:
        return 0.5
    if col.endswith('+'):
        return (val - mn) / (mx - mn)
    else:  # '-'
        return (mx - val) / (mx - mn)

def composite_score(header, ranges, row):
    """Return composite score based on all goal columns."""
    score = 0.0
    cnt = 0
    for c, (mn, mx) in ranges.items():
        v = row[header.index(c)]
        if v == "?":
            continue
        score += normalize_value(c, v, mn, mx)
        cnt += 1
    return score / cnt if cnt else 0.0

# ----------------------------------------------------------------------
# NASA-13: Identify promising combinations of characteristics
# ----------------------------------------------------------------------
def nasa_13(header, rows):
    # Identify goal and non‑goal columns
    goal_cols = [c for c in header if c.endswith(('+', '-'))]
    non_goal_cols = [c for c in header if not c.endswith(('+', '-')) and not c.endswith('X')]

    ranges = get_goal_ranges(header, rows)

    # Compute composite scores for rows with complete goal data
    scored = []
    for r in rows:
        if all(r[header.index(c)] != "?" for c in goal_cols):
            sc = composite_score(header, ranges, r)
            scored.append(sc)

    # Take top 10% (or at least 5) as "top performers"
    if not scored:
        top_rows = []
    else:
        sorted_rows = sorted(((composite_score(header, ranges, r), r) for r in rows if all(r[header.index(c)] != "?" for c in goal_cols)),
                            key=lambda x: x[0], reverse=True)
        cutoff = max(5, math.ceil(0.10 * len(rows)))
        top_rows = [r for _, r in sorted_rows[:cutoff]]

    # Build conditions
    conditions = []
    for col in non_goal_cols:
        idx = header.index(col)
        vals = [r[idx] for r in top_rows if r[idx] != "?"]
        if not vals:
            continue
        if all(numeric(v) for v in vals):
            mn, mx = min(vals), max(vals)
            if mn == mx:
                continue
            # overall range for scoring
            all_vals = [r[idx] for r in rows if r[idx] != "?"]
            if not all_vals:
                continue
            overall_mn, overall_mx = min(all_vals), max(all_vals)
            range_score = (mx - mn) / (overall_mx - overall_mn) if overall_mx != overall_mn else 0
            conditions.append((col, [mn, mx], range_score))
        else:
            # symbolic
            freq = {}
            for v in vals:
                freq[v] = freq.get(v, 0) + 1
            most, count = max(freq.items(), key=lambda kv: kv[1])
            prop = count / len(vals)
            if prop > 0.5:
                conditions.append((col, most, prop))

    # Pick up to 3 best conditions
    conditions.sort(key=lambda x: x[2], reverse=True)
    selected = conditions[:3]

    rule = {col: (val if isinstance(val, str) else [round(val[0], 3), round(val[1], 3)]) for col, val, _ in selected}

    # Count matching rows
    match_count = 0
    for r in rows:
        ok = True
        for col, val in rule.items():
            idx = header.index(col)
            cell = r[idx]
            if cell == "?":
                ok = False
                break
            if isinstance(val, list):
                if not (val[0] <= float(cell) <= val[1]):
                    ok = False
                    break
            else:
                if cell != val:
                    ok = False
                    break
        if ok:
            match_count += 1

    # Human‑readable output
    print("== NASA-13 ==")
    print(f"Rule conditions: {rule}")
    print(f"Matching rows: {match_count}")

    # Machine‑checked answer
    answer = {"rule": rule, "matching": match_count}
    print(f"ANSWER NASA-13 {json.dumps(answer, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# NASA-14: Recommend evaluation order
# ----------------------------------------------------------------------
def nasa_14(header, rows):
    goal_cols = [c for c in header if c.endswith(('+', '-'))]
    ranges = get_goal_ranges(header, rows)

    # Compute scores for all rows (skip rows with any missing goal)
    row_scores = []
    for i, r in enumerate(rows, start=1):
        if any(r[header.index(c)] == "?" for c in goal_cols):
            continue
        sc = composite_score(header, ranges, r)
        row_scores.append((sc, i))

    # Sort descending by score
    sorted_rows = sorted(row_scores, key=lambda x: x[0], reverse=True)
    budget = min(30, len(rows) // 5)

    order = [i for _, i in sorted_rows[:budget]]

    # Human‑readable output
    print("== NASA-14 ==")
    print(f"Evaluation order (top {budget} rows): {order}")

    # Machine‑checked answer
    answer = {"order": order}
    print(f"ANSWER NASA-14 {json.dumps(answer, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    header, rows = read_csv(path)

    nasa_13(header, rows)
    nasa_14(header, rows)

if __name__ == "__main__":
    main()
