#!/usr/bin/env python3
"""
Skill 08: Alternative vehicle selection.

This script implements CAR-15 and CAR-16.
"""

import sys
import csv
import json
import math
from statistics import mean, median, stdev
from typing import List, Dict, Tuple, Optional

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def read_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read CSV and return header and list of row dicts."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    return header, rows

def parse_value(v: str) -> Optional[float]:
    """Parse numeric value, return None for missing."""
    if v in ('', '?'):
        return None
    try:
        return float(v)
    except ValueError:
        return None

def is_goal(col: str) -> bool:
    return col.endswith('+') or col.endswith('-')

def is_input(col: str) -> bool:
    return not is_goal(col) and not col.endswith('X')

# --------------------------------------------------------------------------- #
# Normalisation helpers
# --------------------------------------------------------------------------- #

def compute_minmax(rows: List[Dict[str, str]], cols: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    mins, maxs = {}, {}
    for col in cols:
        vals = [parse_value(r[col]) for r in rows if parse_value(r[col]) is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
        else:
            mins[col] = maxs[col] = 0.0
    return mins, maxs

def normalize(val: float, mn: float, mx: float, is_minimize: bool) -> float:
    if mn == mx:
        return 1.0
    return (mx - val) / (mx - mn) if is_minimize else (val - mn) / (mx - mn)

# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def compute_overall_score(rows: List[Dict[str, str]],
                          goal_cols: List[str],
                          mins: Dict[str, float],
                          maxs: Dict[str, float]) -> List[Optional[float]]:
    scores = []
    for r in rows:
        norm_vals = []
        skip = False
        for col in goal_cols:
            v = parse_value(r[col])
            if v is None:
                skip = True
                break
            n = normalize(v, mins[col], maxs[col], col.endswith('-'))
            norm_vals.append(n)
        if skip or not norm_vals:
            scores.append(None)
        else:
            scores.append(sum(norm_vals) / len(norm_vals))
    return scores

# --------------------------------------------------------------------------- #
# CAR-15: Poorly rated vehicle
# --------------------------------------------------------------------------- #

def car15(header: List[str], rows: List[Dict[str, str]]) -> Dict:
    goal_cols = [c for c in header if is_goal(c)]
    input_cols = [c for c in header if is_input(c)]
    # Min/max for goals (for scoring) and inputs (for distance)
    goal_mins, goal_maxs = compute_minmax(rows, goal_cols)
    inp_mins, inp_maxs = compute_minmax(rows, input_cols)

    scores = compute_overall_score(rows, goal_cols, goal_mins, goal_maxs)

    # Identify poorest row (lowest score)
    poor_idx = None
    poor_score = None
    for i, s in enumerate(scores, start=1):
        if s is None:
            continue
        if poor_score is None or s < poor_score:
            poor_score = s
            poor_idx = i
    if poor_idx is None:
        # All rows missing goals – pick first row
        poor_idx = 1
        poor_score = None

    # Gather candidate rows for alternative (must have score and all inputs present)
    candidates = []
    for i, r in enumerate(rows, start=1):
        if i == poor_idx:
            continue
        # Ensure all goal values present
        if any(parse_value(r[c]) is None for c in goal_cols):
            continue
        # Ensure all input values present
        if any(parse_value(r[c]) is None for c in input_cols):
            continue
        # compute score
        s = compute_overall_score([r], goal_cols, goal_mins, goal_maxs)[0]
        if s is None:
            continue
        candidates.append((i, r, s))

    # Compute distance from poor row
    poor_row = rows[poor_idx - 1]
    def row_norm_inputs(r):
        return [normalize(parse_value(r[c]), inp_mins[c], inp_maxs[c], False) for c in input_cols]

    poor_norm = row_norm_inputs(poor_row)

    def distance_to_poor(other_row):
        other_norm = row_norm_inputs(other_row)
        # Euclidean
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(poor_norm, other_norm)))

    # Filter those with higher score
    better = [(i, r, s) for i, r, s in candidates if s > poor_score]
    if not better:
        # If none better, just take closest regardless of score
        better = candidates

    # Find nearest neighbour
    if better:
        alt_idx, alt_row, alt_score = min(better, key=lambda x: distance_to_poor(x[1]))
    else:
        alt_idx, alt_row, alt_score = None, None, None

    # Compute changes (only input columns that differ)
    changes = {}
    if alt_row is not None:
        for c in input_cols:
            pv = parse_value(poor_row[c])
            av = parse_value(alt_row[c])
            if pv is not None and av is not None and pv != av:
                changes[c] = [pv, av]
    # Build answer
    answer = {
        "row": poor_idx,
        "alternatives": [{
            "row": alt_idx,
            "changes": changes
        }] if alt_idx else []
    }

    # Print results
    print("== CAR-15 ==")
    print(f"Poor vehicle row: {poor_idx}")
    print(f"Alternative vehicle row: {alt_idx if alt_idx else 'none'}")
    print(f"Changes: {changes}")
    print(f"ANSWER CAR-15 {json.dumps(answer)}")

    return answer

# --------------------------------------------------------------------------- #
# CAR-16: Vehicle not meeting preferred targets
# --------------------------------------------------------------------------- #

def compute_percentile(sorted_vals: List[float], pct: float) -> float:
    """Return value at given percentile (0-100)."""
    if not sorted_vals:
        return 0.0
    idx = int(math.floor(pct / 100.0 * len(sorted_vals)))
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return sorted_vals[idx]

def car16(header: List[str], rows: List[Dict[str, str]]) -> Dict:
    goal_cols = [c for c in header if is_goal(c)]
    input_cols = [c for c in header if is_input(c)]

    # Compute thresholds for each goal
    thresholds = {}
    for col in goal_cols:
        vals = [parse_value(r[col]) for r in rows if parse_value(r[col]) is not None]
        if not vals:
            continue
        vals.sort()
        if col.endswith('+'):  # maximize → want high
            thresholds[col] = round(compute_percentile(vals, 80.0), 4)
        else:  # minimize → want low
            thresholds[col] = round(compute_percentile(vals, 20.0), 4)

    # Identify first row that fails any target
    failing_row_idx = None
    for i, r in enumerate(rows, start=1):
        fails = False
        for col in goal_cols:
            v = parse_value(r[col])
            if v is None:
                fails = True
                break
            th = thresholds[col]
            if col.endswith('+') and v < th:
                fails = True
                break
            if col.endswith('-') and v > th:
                fails = True
                break
        if fails:
            failing_row_idx = i
            break
    if failing_row_idx is None:
        # All meet targets – pick first row as default
        failing_row_idx = 1

    # Count targets met for each row
    def targets_met(r):
        cnt = 0
        for col in goal_cols:
            v = parse_value(r[col])
            if v is None:
                continue
            th = thresholds[col]
            if col.endswith('+') and v >= th:
                cnt += 1
            if col.endswith('-') and v <= th:
                cnt += 1
        return cnt

    original_row = rows[failing_row_idx - 1]
    orig_met = targets_met(original_row)

    # Find alternatives that meet more targets
    alternatives = []
    for i, r in enumerate(rows, start=1):
        if i == failing_row_idx:
            continue
        met = targets_met(r)
        if met <= orig_met:
            continue
        # Compute changed input columns
        changed = [c for c in input_cols if parse_value(original_row[c]) != parse_value(r[c])]
        alternatives.append((i, r, met, changed))

    if alternatives:
        # choose minimal number of changed inputs, then higher targets met, then higher score
        # compute overall scores for all rows to rank
        goal_mins, goal_maxs = compute_minmax(rows, goal_cols)
        overall_scores = compute_overall_score(rows, goal_cols, goal_mins, goal_maxs)
        alternatives.sort(key=lambda x: (len(x[3]), -x[2], -(overall_scores[x[0]-1] if overall_scores[x[0]-1] is not None else -1), x[0]))
        best_idx, best_row, best_met, best_changed = alternatives[0]
    else:
        best_idx, best_changed = None, []

    answer = {
        "row": failing_row_idx,
        "targets": thresholds,
        "alternative": best_idx,
        "changed": best_changed
    }

    # Print results
    print("\n== CAR-16 ==")
    print(f"Row not meeting targets: {failing_row_idx}")
    print(f"Targets (thresholds): {thresholds}")
    print(f"Alternative row: {best_idx if best_idx else 'none'}")
    print(f"Changed input columns: {best_changed}")
    print(f"ANSWER CAR-16 {json.dumps(answer)}")

    return answer

# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    car15(header, rows)
    car16(header, rows)

if __name__ == "__main__":
    main()
