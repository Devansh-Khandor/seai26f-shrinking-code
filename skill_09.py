#!/usr/bin/env python3
"""
Skill 09 – Apache performance analytics (APACHE-17 & APACHE-18)

Usage:
    python skill_09.py <csv_path>
"""

import json
import sys
import ezr
import math
from collections import defaultdict

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
def goal_col(tbl: ezr.TBL) -> ezr.COL:
    """Return the first column whose name ends with '+' or '-'."""
    for c in tbl.cols.values():
        if c.txt.endswith(('+', '-')):
            return c
    raise RuntimeError("No goal column found.")

def setting_cols(tbl: ezr.TBL) -> list[ezr.COL]:
    """Return columns that are neither goals nor X columns."""
    return [c for c in tbl.cols.values()
            if not c.txt.endswith(('+', '-', 'X'))]

def row_index(tbl: ezr.TBL, row: ezr.ROW) -> int:
    """Return 1‑based index of a row in the table."""
    return tbl.rows.index(row) + 1

def get_row_goal_value(row: ezr.ROW, goal_at: int):
    val = row[goal_at]
    if val == "?":
        raise ValueError("Missing goal value.")
    return float(val)

# --------------------------------------------------------------------
# APACHE-17: unusually different performance
# --------------------------------------------------------------------
def apache_17(tbl: ezr.TBL):
    goal = goal_col(tbl)
    goal_at = goal.at
    settings = setting_cols(tbl)

    # Build groups keyed by tuple of setting values
    groups = defaultdict(list)   # key -> list of (row, goal_val)
    for row in tbl.rows:
        key = tuple(row[c.at] for c in settings)
        try:
            gv = get_row_goal_value(row, goal_at)
        except ValueError:
            continue
        groups[key].append((row, gv))

    # For each row compute expected performance and difference
    diffs = []
    for row in tbl.rows:
        key = tuple(row[c.at] for c in settings)
        group = groups[key]
        if len(group) <= 1:   # no other rows with same settings
            continue
        gv = get_row_goal_value(row, goal_at)
        # mean of other rows in group
        other_vals = [v for r, v in group if r is not row]
        expected = sum(other_vals) / len(other_vals)
        diff = abs(gv - expected)
        diffs.append({"row": row_index(tbl, row),
                      "actual": gv,
                      "expected": expected,
                      "diff": diff})

    if not diffs:
        flagged = []
    else:
        # Compute mean and std of diffs
        vals = [d["diff"] for d in diffs]
        mean_diff = sum(vals) / len(vals)
        std_diff = math.sqrt(sum((x - mean_diff) ** 2 for x in vals) / len(vals))
        threshold = mean_diff + 2 * std_diff
        flagged = [d for d in diffs if d["diff"] > threshold]
        # keep top 20 by largest diff
        flagged.sort(key=lambda d: d["diff"], reverse=True)
        flagged = flagged[:20]
        # format output
        flagged = [{"row": d["row"],
                    "column": goal.txt,
                    "actual": round(d["actual"], 2),
                    "expected": round(d["expected"], 2)} for d in flagged]

    # Print section
    print("== APACHE-17 ==")
    if flagged:
        for f in flagged:
            print(f"Row {f['row']}: {f['column']} actual={f['actual']} expected={f['expected']}")
    else:
        print("No unusually different performances found.")
    # Machine‑checked answer
    print(f"ANSWER APACHE-17 {json.dumps({'flagged': flagged})}")

# --------------------------------------------------------------------
# APACHE-18: unusual setting combinations
# --------------------------------------------------------------------
def apache_18(tbl: ezr.TBL):
    settings = setting_cols(tbl)

    n_settings = len(settings)
    # For each row compute similarity to nearest historical row
    similarities = {}
    for idx, row in enumerate(tbl.rows, start=1):
        # Find nearest other row (by xdist)
        best_dist = None
        best_row = None
        for jdx, other in enumerate(tbl.rows, start=1):
            if idx == jdx:
                continue
            d = ezr.xdist(tbl, row, other)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_row = other
        # Compute similarity: fraction of matching setting values
        matches = sum(1 for c in settings if row[c.at] == best_row[c.at])
        similarity = matches / n_settings if n_settings else 1.0
        similarities[idx] = similarity

    if not similarities:
        flagged = []
    else:
        vals = list(similarities.values())
        mean_sim = sum(vals) / len(vals)
        std_sim = math.sqrt(sum((x - mean_sim) ** 2 for x in vals) / len(vals))
        threshold = mean_sim - 2 * std_sim
        flagged = [idx for idx, sim in similarities.items() if sim < threshold]
        flagged.sort()
        flagged = flagged[:20]
        # Scores dictionary
        scores = {str(idx): round(similarities[idx], 2) for idx in flagged}

    # Print section
    print("\n== APACHE-18 ==")
    if flagged:
        print("Rows with unusual setting combinations:")
        for r in flagged:
            print(f"  Row {r}")
        print("Unusualness scores:")
        for r in flagged:
            print(f"  Row {r}: {scores[str(r)]}")
    else:
        print("No unusual setting combinations found.")
    # Machine‑checked answer
    print(f"ANSWER APACHE-18 {json.dumps({'flagged': flagged, 'scores': scores})}")

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    tbl = ezr.Tbl(ezr.csv(csv_path))

    apache_17(tbl)
    apache_18(tbl)

if __name__ == "__main__":
    main()
