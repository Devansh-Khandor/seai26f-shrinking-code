#!/usr/bin/env python3
"""
Skill 08: Advanced vehicle comparison (CAR‑15 & CAR‑16).

The script implements the following CAR tasks:

* **CAR‑15** – For a poorly rated vehicle, provide alternative vehicles from the
  historical inventory that are similar in input characteristics but have
  better recorded performance.  The output lists the differences in the
  input columns that explain the improvement.

* **CAR‑16** – For a vehicle that does not meet the preferred performance
  targets, find the closest available alternative that meets more of those
  targets while changing as few input characteristics as possible.

The scoring, thresholds, and data handling follow the same rules as the
other skill modules: rows are numbered starting at 1, missing values are
represented by None, and goal columns are those that end in ‘+’ (to be
maximized) or ‘-’ (to be minimized).  The script is run as:

    python skill_08.py <csv_path>
"""

import csv
import json
import math
import sys
from collections import defaultdict
from statistics import mean

# ----------------------------------------------------------------------
# CSV utilities
# ----------------------------------------------------------------------
def read_csv(path):
    """Read the CSV and return a list of (row_index, row_dict) tuples."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, line in enumerate(reader, start=1):
            parsed = {}
            for k, v in line.items():
                if v == '?' or v is None:
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = None
            rows.append((idx, parsed))
    return rows


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def is_goal(col):
    return col.endswith('+') or col.endswith('-')

def is_input(col):
    """Input columns are those that are not goals and not ending with 'X'."""
    return not is_goal(col) and not col.endswith('X')

def compute_goal_ranges(rows, goal_cols):
    """Return min and max dicts for each goal column."""
    mins = {}
    maxs = {}
    for col in goal_cols:
        vals = [rec[col] for _, rec in rows if rec[col] is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
        else:
            mins[col] = maxs[col] = 0.0
    return mins, maxs

def composite_score(rec, mins, maxs, goal_cols):
    """Return composite score for a single record or None if missing."""
    if any(rec.get(col) is None for col in goal_cols):
        return None
    score = 0.0
    for col in goal_cols:
        denom = maxs[col] - mins[col]
        if denom == 0:
            continue
        if col.endswith('-'):  # minimize
            score += (maxs[col] - rec[col]) / denom
        else:  # maximize
            score += (rec[col] - mins[col]) / denom
    return score

def percentile(values, p):
    """Return the p-th percentile (0-100) of a list of numbers."""
    if not values:
        return None
    values = sorted(values)
    n = len(values)
    # Using nearest-rank method
    k = math.ceil((p / 100) * n)
    return values[k-1] if k-1 < n else values[-1]

def threshold_for_goal(col, values):
    """Return threshold for goal column: 75th percentile for '+', 25th for '-'."""
    if col.endswith('+'):
        return percentile(values, 75)
    else:
        return percentile(values, 25)

# ----------------------------------------------------------------------
# CAR-15 implementation
# ----------------------------------------------------------------------
def car15(rows, goal_cols, input_cols, history_rows):
    """Return dict for CAR-15."""
    # Compute global goal ranges for scoring
    mins, maxs = compute_goal_ranges(rows, goal_cols)

    # Compute scores for all rows
    scores = {}
    for idx, rec in rows:
        s = composite_score(rec, mins, maxs, goal_cols)
        if s is not None:
            scores[idx] = s

    # Pick the worst scored row that has a valid score
    if not scores:
        return {"row": None, "alternatives": []}
    poor_row_idx = min(scores, key=lambda k: scores[k])
    poor_row = next(rec for idx, rec in rows if idx == poor_row_idx)

    # Find alternatives among history that have a better score and similar inputs
    alternatives = []
    poor_vals = {col: poor_row[col] for col in input_cols if poor_row[col] is not None}
    for idx, rec in history_rows:
        if idx == poor_row_idx:
            continue
        # Must have a better composite score
        if idx not in scores or scores[idx] <= scores[poor_row_idx]:
            continue
        # Compute distance on input columns where both have values
        valid = True
        dist2 = 0.0
        changes = {}
        for col in input_cols:
            v1 = poor_row[col]
            v2 = rec[col]
            if v1 is None or v2 is None:
                valid = False
                break
            diff = v1 - v2
            dist2 += diff * diff
            if diff != 0:
                changes[col] = [v1, v2]
        if not valid:
            continue
        alternatives.append((dist2, idx, changes))

    # If no suitable alternatives found in history, relax to all rows
    if not alternatives:
        for idx, rec in rows:
            if idx == poor_row_idx:
                continue
            if idx not in scores or scores[idx] <= scores[poor_row_idx]:
                continue
            valid = True
            dist2 = 0.0
            changes = {}
            for col in input_cols:
                v1 = poor_row[col]
                v2 = rec[col]
                if v1 is None or v2 is None:
                    valid = False
                    break
                diff = v1 - v2
                dist2 += diff * diff
                if diff != 0:
                    changes[col] = [v1, v2]
            if not valid:
                continue
            alternatives.append((dist2, idx, changes))

    # Pick top 3 nearest alternatives
    alternatives.sort(key=lambda x: x[0])
    result_alts = []
    for _, idx, changes in alternatives[:3]:
        result_alts.append({"row": idx, "changes": changes})

    return {"row": poor_row_idx, "alternatives": result_alts}

# ----------------------------------------------------------------------
# CAR-16 implementation
# ----------------------------------------------------------------------
def car16(rows, goal_cols, input_cols, history_rows):
    """Return dict for CAR-16."""
    # Compute thresholds for each goal
    thresholds = {}
    for col in goal_cols:
        vals = [rec[col] for _, rec in rows if rec[col] is not None]
        thresholds[col] = threshold_for_goal(col, vals)

    # Count targets met for each row
    def targets_met(rec):
        count = 0
        for col, thr in thresholds.items():
            val = rec[col]
            if val is None:
                continue
            if col.endswith('+') and val >= thr:
                count += 1
            elif col.endswith('-') and val <= thr:
                count += 1
        return count

    target_counts = {}
    for idx, rec in rows:
        target_counts[idx] = targets_met(rec)

    # Find a row that does not meet all targets and has the lowest composite score among such rows
    # Compute composite scores for all rows
    mins, maxs = compute_goal_ranges(rows, goal_cols)
    scores = {}
    for idx, rec in rows:
        s = composite_score(rec, mins, maxs, goal_cols)
        if s is not None:
            scores[idx] = s

    # Filter rows that don't meet all targets
    not_all = [idx for idx, cnt in target_counts.items() if cnt < len(goal_cols)]
    if not not_all:
        # All rows meet all targets; nothing to do
        return {"row": None, "targets": thresholds, "alternative": None, "changed": []}

    # Choose the worst-scored among those
    row_idx = min(not_all, key=lambda k: scores.get(k, float('inf')))
    row = next(rec for idx, rec in rows if idx == row_idx)

    row_target_count = target_counts[row_idx]

    # Search alternatives that meet more targets
    best_alts = []
    for idx, rec in history_rows:
        if idx == row_idx:
            continue
        cnt = target_counts[idx]
        if cnt <= row_target_count:
            continue
        # Compute number of differing input columns
        diff_cols = [col for col in input_cols
                     if row.get(col) is not None and rec.get(col) is not None and row[col] != rec[col]]
        best_alts.append((len(diff_cols), idx, diff_cols, cnt))

    if not best_alts:
        # If no alternative found in history, consider all rows
        for idx, rec in rows:
            if idx == row_idx:
                continue
            cnt = target_counts[idx]
            if cnt <= row_target_count:
                continue
            diff_cols = [col for col in input_cols
                         if row.get(col) is not None and rec.get(col) is not None and row[col] != rec[col]]
            best_alts.append((len(diff_cols), idx, diff_cols, cnt))

    if not best_alts:
        # Still none, return None
        return {"row": row_idx, "targets": thresholds, "alternative": None, "changed": []}

    # Pick alternative with fewest changes; tie by lowest row number
    best_alts.sort(key=lambda x: (x[0], x[1]))
    _, alt_idx, diff_cols, _ = best_alts[0]

    return {"row": row_idx, "targets": thresholds, "alternative": alt_idx, "changed": diff_cols}

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path):
    rows = read_csv(csv_path)

    if not rows:
        print("No data rows found.")
        return

    # Determine goal and input columns
    _, first = rows[0]
    goal_cols = [k for k in first if is_goal(k)]
    input_cols = [k for k in first if is_input(k)]

    # Determine held-out and arrival rows
    total_rows = len(rows)
    arrivals_count = round(0.2 * total_rows)
    history_rows = rows[:-arrivals_count] if arrivals_count > 0 else rows
    arrival_rows = rows[-arrivals_count:] if arrivals_count > 0 else []

    # CAR-15
    print("== CAR-15 ==")
    res15 = car15(rows, goal_cols, input_cols, history_rows)
    if res15["row"] is None:
        print("No suitable poor vehicle found.")
    else:
        poor_idx = res15["row"]
        poor_score = composite_score(dict(rows[poor_idx-1][1]), *compute_goal_ranges(rows, goal_cols), goal_cols)
        print(f"Poor vehicle (row {poor_idx}) composite score: {poor_score:.4f}")
        if res15["alternatives"]:
            print("Alternative vehicles with better performance:")
            for alt in res15["alternatives"]:
                changes_str = ", ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in alt["changes"].items())
                print(f"  Row {alt['row']}: {changes_str}")
            print("Explanation: The alternatives have similar input values but outperform the poor vehicle on the goal metrics.")
        else:
            print("No better alternatives found in the historical inventory.")
    print("ANSWER CAR-15 " + json.dumps(res15, separators=(',', ':')))

    # CAR-16
    print("\n== CAR-16 ==")
    res16 = car16(rows, goal_cols, input_cols, history_rows)
    if res16["row"] is None:
        print("All vehicles meet all preferred targets.")
    else:
        print(f"Vehicle (row {res16['row']}) does not meet all preferred targets.")
        print("Preferred target thresholds:")
        for k, v in res16["targets"].items():
            print(f"  {k}: {v:.4f}")
        if res16["alternative"] is not None:
            print(f"Closest alternative meeting more targets: row {res16['alternative']}")
            if res16["changed"]:
                changes = ", ".join(res16["changed"])
                print(f"Changed input columns: {changes}")
            else:
                print("No input changes required.")
        else:
            print("No suitable alternative found.")
    print("ANSWER CAR-16 " + json.dumps(res16, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_08.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])
