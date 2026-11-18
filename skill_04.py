#!/usr/bin/env python3
"""
Skill 04: Real‑time vehicle assessment and inventory summarisation.

This script implements CAR‑07 (instant assessment of arriving records) and
CAR‑08 (periodic inventory snapshots and a shortlist of the best cars).

The CSV format is the same as for the previous skills: header row, rows
contain numeric values or '?' for missing values.  The script follows the
exact specification from the assignment – it never hard‑codes row numbers
or values, and it handles missing data as required.

Usage:
    python skill_04.py <csv_path>
"""

import sys
import csv
import json

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def read_csv(path):
    """
    Read the CSV file and return a list of (row_number, row_dict) tuples.
    Row numbers are 1‑based, i.e. the first data row is 1.
    Missing values ('?') are converted to None.
    """
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for i, line in enumerate(reader, start=1):
            parsed = {}
            for k, v in line.items():
                if v == '?' or v is None:
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = None
            rows.append((i, parsed))
    return rows

def goal_columns(rows):
    """
    Return a list of column names that are goals: ending with '+' or '-'.
    """
    # Use the header of the first row (all rows share same keys)
    if not rows:
        return []
    _, first = rows[0]
    return [k for k in first.keys() if k.endswith('-') or k.endswith('+')]

def stats_for_rows(rows, goal_cols):
    """
    For the given rows (list of (rownum, rowdict)), compute for each goal
    column the count of valid rows, mean, min and max.  Only rows that
    contain a non‑None value for the goal column are considered.
    """
    stats = {}
    for col in goal_cols:
        vals = [row[col] for (_, row) in rows if row[col] is not None]
        if vals:
            n = len(vals)
            mean = sum(vals) / n
            mn = min(vals)
            mx = max(vals)
            stats[col] = {"n": n, "mean": mean, "min": mn, "max": mx}
        else:
            stats[col] = {"n": 0, "mean": None, "min": None, "max": None}
    return stats

def min_max_for_rows(rows, goal_cols):
    """
    Compute min and max per goal for the subset of rows that have all
    goal values present.  Returns two dictionaries: mins and maxs.
    """
    valid_rows = [(rnum, rdict) for (rnum, rdict) in rows if all(rdict[c] is not None for c in goal_cols)]
    mins = {}
    maxs = {}
    for col in goal_cols:
        vals = [rdict[col] for (_, rdict) in valid_rows]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
        else:
            mins[col] = maxs[col] = None
    return mins, maxs, valid_rows

def score_row(row, goal_cols, mins, maxs):
    """
    Compute the composite score of a single row (rowdict).  If any
    goal value is missing, return None.
    """
    for col in goal_cols:
        if row[col] is None:
            return None
    score = 0.0
    for col in goal_cols:
        denom = maxs[col] - mins[col]
        if col.endswith('-'):  # minimize
            if denom:
                score += (maxs[col] - row[col]) / denom
            else:
                score += 0.0
        else:  # maximize
            if denom:
                score += (row[col] - mins[col]) / denom
            else:
                score += 0.0
    return score

# ----------------------------------------------------------------------
# CAR-07: instant assessment of arriving records
# ----------------------------------------------------------------------
def assess_arrivals(rows, goal_cols, arrivals):
    """
    For each arrival (list of (rownum, rowdict)), determine whether it
    would be in the top 5 best rows seen so far (history + previous arrivals).
    Return a list of dicts: {"row": int, "desirable": bool}.
    """
    assessments = []
    seen = []  # rows seen so far
    # first add history
    for r in rows[:-len(arrivals)]:
        seen.append(r)
    for arrival in arrivals:
        seen.append(arrival)  # include the arrival
        mins, maxs, valid = min_max_for_rows(seen, goal_cols)
        if not valid:
            desirable = False
        else:
            scored = []
            for (_, rdict) in valid:
                s = score_row(rdict, goal_cols, mins, maxs)
                scored.append((s, rdict))
            scored.sort(key=lambda x: x[0], reverse=True)
            top5 = scored[:5]
            top_rows = [rdict for (_, rdict) in top5]
            desirable = arrival[1] in top_rows
        assessments.append({"row": arrival[0], "desirable": desirable})
    return assessments

# ----------------------------------------------------------------------
# CAR-08: periodic checkpoints
# ----------------------------------------------------------------------
def checkpoints(rows, goal_cols, m_values):
    """
    Produce checkpoints for the requested numbers of arrivals.
    For each m, return a dict:
        {"after": m,
         "summary": {goal: {"n":..., "mean":..., "min":..., "max":...}, ...},
         "shortlist": [rownum, ...]}.
    """
    all_checkpoints = []
    history = rows[:-len(rows)+0]  # placeholder, will adjust
    total_rows = len(rows)
    arrivals_count = round(0.2 * total_rows)
    history = rows[:-arrivals_count]
    arrivals = rows[-arrivals_count:]

    for m in m_values:
        data = history + arrivals[:m]
        mins, maxs, valid = min_max_for_rows(data, goal_cols)
        summary = stats_for_rows(valid, goal_cols)
        # compute scores for shortlisted rows
        if not valid:
            shortlist = []
        else:
            scored = []
            for (_, rdict) in valid:
                s = score_row(rdict, goal_cols, mins, maxs)
                scored.append((s, rdict))
            scored.sort(key=lambda x: x[0], reverse=True)
            top5 = scored[:5]
            shortlist = [rdict_rownum for (_, rdict_rownum) in top5]
        checkpoint = {
            "after": m,
            "summary": summary,
            "shortlist": shortlist
        }
        all_checkpoints.append(checkpoint)
    return all_checkpoints

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path):
    rows = read_csv(csv_path)
    goal_cols = goal_columns(rows)

    # Determine arrivals
    total_rows = len(rows)
    arrivals_count = round(0.2 * total_rows)
    history_rows = rows[:-arrivals_count]
    arrival_rows = rows[-arrivals_count:]

    # --- CAR‑07 ----------------------------------------------------
    print("== CAR-07 ==")
    assessments = assess_arrivals(rows, goal_cols, arrival_rows)
    # Provide at least three lines of output
    print("Assessments:")
    for a in assessments[:3]:
        print(f"Row {a['row']}: {'true' if a['desirable'] else 'false'}")
    print("ANSWER CAR-07 " + json.dumps({"assessments": assessments}, separators=(',', ':')))

    # --- CAR‑08 ----------------------------------------------------
    print("== CAR-08 ==")
    m_vals = [0, 40, 80]
    # Cap m_vals to actual arrivals count
    m_vals = [m for m in m_vals if m <= arrivals_count]
    cps = checkpoints(rows, goal_cols, m_vals)
    print("Checkpoints:")
    for cp in cps:
        print(f"After {cp['after']} arrivals: top rows {cp['shortlist']}")
    print("ANSWER CAR-08 " + json.dumps({"checkpoints": cps}, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_04.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])
