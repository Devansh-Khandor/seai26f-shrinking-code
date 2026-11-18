#!/usr/bin/env python3
"""
Skill 08: Vehicle recommendation and improvement analysis.

- CAR-15: For a poorly rated vehicle, show alternative vehicles in the historical
          inventory with similar characteristics but better recorded performance,
          and explain which differences could account for the improvement.
- CAR-16: For a vehicle that does not meet preferred performance targets,
          identify the closest available alternative that meets more of those
          targets while changing as few vehicle characteristics as possible.
"""

import csv
import json
import sys
import math
from collections import defaultdict

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def parse_value(v: str):
    if v == '?' or v.strip() == '':
        return None
    try:
        return float(v)
    except ValueError:
        return v  # keep string for categorical columns

def is_goal(col_name: str) -> bool:
    return (col_name.endswith('+') or col_name.endswith('-')) and not col_name.endswith('X')

def is_input(col_name: str) -> bool:
    return not is_goal(col_name) and not col_name.endswith('X')

def is_maximize(col_name: str) -> bool:
    return col_name.endswith('+')

def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows, reader.fieldnames

def compute_minmax(rows, cols):
    mins, maxs = {}, {}
    for col in cols:
        vals = [parse_value(r[col]) for r in rows]
        vals = [v for v in vals if isinstance(v, (int, float)) and v is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
    return mins, maxs

def normalize_goal(row, mins, maxs, goals):
    norm = {}
    for col in goals:
        v = parse_value(row[col])
        if v is None or not isinstance(v, (int, float)):
            return None
        mn = mins[col]
        mx = maxs[col]
        if mx == mn:
            norm[col] = 1.0
        else:
            if is_maximize(col):
                norm[col] = (v - mn) / (mx - mn)
            else:
                norm[col] = (mx - v) / (mx - mn)
    return norm

def score_from_norm(norm_dict):
    return sum(norm_dict.values())

def normalize_inputs(row, mins, maxs, inputs):
    norm = {}
    for col in inputs:
        v = parse_value(row[col])
        if v is None or not isinstance(v, (int, float)):
            return None
        mn = mins[col]
        mx = maxs[col]
        if mx == mn:
            norm[col] = 1.0
        else:
            norm[col] = (v - mn) / (mx - mn)
    return norm

def euclidean_distance(dict1, dict2):
    sq = 0.0
    cnt = 0
    for col, val1 in dict1.items():
        val2 = dict2.get(col)
        if val2 is None:
            continue
        sq += (val1 - val2) ** 2
        cnt += 1
    if cnt == 0:
        return float('inf')
    return math.sqrt(sq)

# ----------------------------------------------------------------------
# CAR-15 implementation
# ----------------------------------------------------------------------
def car_15(rows, goals, inputs, history_rows):
    # Min-max for goals over all rows
    goal_mins, goal_maxs = compute_minmax(rows, goals)

    # Compute scores for rows with complete goals
    scores = {}
    for idx, r in enumerate(rows, start=1):
        norm = normalize_goal(r, goal_mins, goal_maxs, goals)
        if norm is not None:
            scores[idx] = score_from_norm(norm)

    if not scores:
        return None  # no valid rows

    # Worst row
    worst_row_num = min(scores, key=lambda k: scores[k])
    worst_score = scores[worst_row_num]
    worst_row = rows[worst_row_num - 1]

    # Min-max for input columns over historical rows (for similarity)
    inp_mins, inp_maxs = compute_minmax(history_rows, inputs)

    # Normalized inputs for worst row
    worst_norm_inputs = normalize_inputs(worst_row, inp_mins, inp_maxs, inputs)
    if worst_norm_inputs is None:
        worst_norm_inputs = {}

    alternatives = []
    for idx, r in enumerate(history_rows, start=1):
        if idx == worst_row_num:
            continue
        # Need to compare against original indices: history_rows start at 1
        # but actual row number is idx (since history starts at 1)
        # However history_rows corresponds to first N rows of full dataset,
        # so indices are same as original indices.
        # So idx is actual row number.
        # Check that row has complete goals and better score
        norm = normalize_goal(r, goal_mins, goal_maxs, goals)
        if norm is None:
            continue
        score = score_from_norm(norm)
        if score <= worst_score:
            continue
        # Compute distance on normalized inputs
        norm_inputs = normalize_inputs(r, inp_mins, inp_maxs, inputs)
        if norm_inputs is None:
            continue
        dist = euclidean_distance(worst_norm_inputs, norm_inputs)
        alternatives.append((dist, idx, r))

    # Sort by distance ascending
    alternatives.sort(key=lambda x: x[0])
    # Take top 3
    top_alts = alternatives[:3]

    alt_list = []
    for _, alt_idx, alt_row in top_alts:
        changes = {}
        for col in inputs:
            val1 = parse_value(worst_row[col])
            val2 = parse_value(alt_row[col])
            if val1 != val2:
                changes[col] = [val1, val2]
        alt_list.append({
            "row": alt_idx,
            "changes": changes
        })

    return {
        "row": worst_row_num,
        "alternatives": alt_list
    }

# ----------------------------------------------------------------------
# CAR-16 implementation
# ----------------------------------------------------------------------
def car_16(rows, goals, inputs):
    # Compute mean thresholds for each goal
    goal_mins, goal_maxs = compute_minmax(rows, goals)
    thresholds = {}
    for col in goals:
        vals = [parse_value(r[col]) for r in rows]
        vals = [v for v in vals if isinstance(v, (int, float)) and v is not None]
        if not vals:
            continue
        mean_val = sum(vals) / len(vals)
        thresholds[col] = mean_val

    # Find a row that fails at least one threshold
    candidate_row_num = None
    candidate_targets_met = 0
    for idx, r in enumerate(rows, start=1):
        met = 0
        for col, thr in thresholds.items():
            val = parse_value(r[col])
            if val is None:
                continue
            if (is_maximize(col) and val >= thr) or (not is_maximize(col) and val <= thr):
                met += 1
        if met < len(thresholds):  # fails at least one
            candidate_row_num = idx
            candidate_targets_met = met
            break

    if candidate_row_num is None:
        return None  # all rows meet all targets

    candidate_row = rows[candidate_row_num - 1]

    # Compute input min/max for normalization (for distance)
    inp_mins, inp_maxs = compute_minmax(rows, inputs)

    candidate_norm_inputs = normalize_inputs(candidate_row, inp_mins, inp_maxs, inputs)
    if candidate_norm_inputs is None:
        candidate_norm_inputs = {}

    best_alternative = None
    best_diff_count = None
    best_distance = None

    for idx, r in enumerate(rows, start=1):
        if idx == candidate_row_num:
            continue
        # Count targets met
        met = 0
        for col, thr in thresholds.items():
            val = parse_value(r[col])
            if val is None:
                continue
            if (is_maximize(col) and val >= thr) or (not is_maximize(col) and val <= thr):
                met += 1
        if met <= candidate_targets_met:
            continue  # must meet more targets

        # Compute number of differing input columns
        diff_count = 0
        changed_cols = []
        for col in inputs:
            val1 = parse_value(candidate_row[col])
            val2 = parse_value(r[col])
            if val1 != val2:
                diff_count += 1
                changed_cols.append(col)

        # Compute distance for tie-breaking
        norm_inputs = normalize_inputs(r, inp_mins, inp_maxs, inputs)
        if norm_inputs is None:
            dist = float('inf')
        else:
            dist = euclidean_distance(candidate_norm_inputs, norm_inputs)

        # Update best alternative
        if (best_alternative is None or
            diff_count < best_diff_count or
            (diff_count == best_diff_count and dist < best_distance)):
            best_alternative = idx
            best_diff_count = diff_count
            best_distance = dist
            best_changed = changed_cols

    if best_alternative is None:
        return None

    return {
        "row": candidate_row_num,
        "targets": thresholds,
        "alternative": best_alternative,
        "changed": best_changed
    }

# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    rows, fieldnames = read_csv(path)

    goal_cols = [c for c in fieldnames if is_goal(c)]
    input_cols = [c for c in fieldnames if is_input(c)]

    # Determine arrivals
    total_rows = len(rows)
    arrivals_count = round(0.2 * total_rows)
    history_rows = rows[:total_rows - arrivals_count]

    # CAR-15
    print("== CAR-15 ==")
    res15 = car_15(rows, goal_cols, input_cols, history_rows)
    if res15 is None:
        print("No suitable poorly rated vehicle found.")
    else:
        print(f"Poorly rated vehicle row: {res15['row']}")
        print("Similar but better alternatives:")
        for alt in res15["alternatives"]:
            print(f"  Row {alt['row']} changes: {alt['changes']}")
    print("ANSWER CAR-15", json.dumps(res15, ensure_ascii=False))

    # CAR-16
    print("\n== CAR-16 ==")
    res16 = car_16(rows, goal_cols, input_cols)
    if res16 is None:
        print("No vehicle found that fails preferred targets.")
    else:
        print(f"Vehicle that fails preferred targets: row {res16['row']}")
        print(f"Thresholds: {res16['targets']}")
        print(f"Alternative row: {res16['alternative']}")
        print(f"Changed input columns: {res16['changed']}")
    print("ANSWER CAR-16", json.dumps(res16, ensure_ascii=False))

if __name__ == "__main__":
    main()
