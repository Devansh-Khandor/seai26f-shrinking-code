#!/usr/bin/env python3
"""
Skill 06: Recommendation explanation and tradeoff analysis.

This script implements CAR-11 and CAR-12.  It follows the same data
handling conventions as the earlier skills: numeric values are parsed as
floats, missing values are represented by None, and goal columns are
identified by a trailing '+' (maximize) or '-' (minimize).  Columns
ending with 'X' are not used as inputs.

The script prints, for each request, a section header of the form
== <ID> == followed by at least three lines of human‑readable
results, then a machine‑checked answer line as specified in the
problem statement.
"""

import sys
import csv
import json
import math
from typing import List, Dict, Tuple, Any, Optional

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def read_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read CSV file, return header and list of rows (dict of strings)."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    return header, rows

def parse_value(v: str) -> Optional[float]:
    """Parse numeric value; return None for missing or invalid entries."""
    if v in ('', '?'):
        return None
    try:
        return float(v)
    except ValueError:
        return None

def is_goal(col: str) -> bool:
    return col.endswith('+') or col.endswith('-')

# --------------------------------------------------------------------------- #
# Scoring utilities
# --------------------------------------------------------------------------- #

def compute_global_minmax(rows: List[Dict[str, str]], goal_cols: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return dicts of min and max values per goal column across all rows."""
    mins = {}
    maxs = {}
    for col in goal_cols:
        vals = [parse_value(r[col]) for r in rows if parse_value(r[col]) is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
        else:
            mins[col] = maxs[col] = 0.0
    return mins, maxs

def normalize_value(val: float, mn: float, mx: float, is_minimize: bool) -> float:
    if mn == mx:
        return 1.0
    return (mx - val) / (mx - mn) if is_minimize else (val - mn) / (mx - mn)

def score_row(row: Dict[str, str], mins: Dict[str, float], maxs: Dict[str, float], goal_cols: List[str]) -> Optional[float]:
    """Return average normalized score for a row; None if any goal missing."""
    norms = []
    for col in goal_cols:
        v = parse_value(row[col])
        if v is None:
            return None
        n = normalize_value(v, mins[col], maxs[col], col.endswith('-'))
        norms.append(n)
    return sum(norms) / len(norms)

# --------------------------------------------------------------------------- #
# Recommendation logic
# --------------------------------------------------------------------------- #

def get_recommended_and_neighbors(rows: List[Dict[str, str]], mins: Dict[str, float],
                                 maxs: Dict[str, float], goal_cols: List[str],
                                 eval_budget: int) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Evaluate first `eval_budget` rows, find best, then return its 1‑based row number
    and a list of three nearest neighbors (row number and goal values).
    """
    # Evaluate rows
    eval_rows = rows[:eval_budget]
    scores = {}
    for idx, row in enumerate(eval_rows, start=1):
        s = score_row(row, mins, maxs, goal_cols)
        if s is not None:
            scores[idx] = s
    if not scores:
        # No evaluable rows
        return 0, []

    # Best row
    best_row_num = max(scores, key=lambda k: scores[k])
    best_row = rows[best_row_num - 1]

    # Compute distances to all rows (excluding those with missing goals)
    distances = []
    for idx, row in enumerate(rows, start=1):
        if idx == best_row_num:
            continue
        # Check all goals present
        if any(parse_value(row[col]) is None for col in goal_cols):
            continue
        # Compute Euclidean distance on normalized values
        dist_sq = 0.0
        for col in goal_cols:
            v = parse_value(row[col])
            n = normalize_value(v, mins[col], maxs[col], col.endswith('-'))
            vb = parse_value(best_row[col])
            nb = normalize_value(vb, mins[col], maxs[col], col.endswith('-'))
            dist_sq += (n - nb) ** 2
        distances.append((dist_sq, idx, row))

    # Get 3 closest
    distances.sort(key=lambda x: x[0])
    neighbors = []
    for _, nidx, nrow in distances[:3]:
        # Prepare values dict with raw strings
        vals = {c: nrow[c] for c in goal_cols}
        neighbors.append({"row": nidx, "values": vals})
    return best_row_num, neighbors

# --------------------------------------------------------------------------- #
# Tradeoff analysis
# --------------------------------------------------------------------------- #

def find_conflicting_pair(rows: List[Dict[str, str]],
                         mins: Dict[str, float],
                         maxs: Dict[str, float],
                         goal_cols: List[str]) -> Tuple[int, int, float, float]:
    """
    Return indices (1‑based) of two rows: preferred (higher score) and
    a conflicting one (different strengths).  Also return their scores.
    """
    # Compute scores for all rows with complete goals
    all_scores = {}
    for idx, row in enumerate(rows, start=1):
        s = score_row(row, mins, maxs, goal_cols)
        if s is not None:
            all_scores[idx] = s

    if not all_scores:
        return 0, 0, 0.0, 0.0

    # Preferred is the best
    pref_idx = max(all_scores, key=lambda k: all_scores[k])
    pref_score = all_scores[pref_idx]
    pref_row = rows[pref_idx - 1]

    # Find conflicting row
    conflict_idx = None
    for idx, row in enumerate(rows, start=1):
        if idx == pref_idx:
            continue
        # Must have all goals
        if any(parse_value(row[c]) is None for c in goal_cols):
            continue
        # Check if there is at least one goal where this row is better
        better = False
        for c in goal_cols:
            a_val = parse_value(pref_row[c])
            b_val = parse_value(row[c])
            if a_val is None or b_val is None:
                continue
            if c.endswith('-'):  # minimize
                if b_val < a_val:
                    better = True
                    break
            else:  # maximize
                if b_val > a_val:
                    better = True
                    break
        if better:
            conflict_idx = idx
            break

    # If no conflicting row found, use second best
    if conflict_idx is None:
        # Sort by score descending
        sorted_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_items) >= 2:
            conflict_idx = sorted_items[1][0]
        else:
            conflict_idx = pref_idx  # only one row

    conflict_score = all_scores[conflict_idx]
    return pref_idx, conflict_idx, pref_score, conflict_score

def compute_differences(a_row: Dict[str, str], b_row: Dict[str, str], goal_cols: List[str]) -> List[Dict[str, Any]]:
    """Return list of differences for each goal column."""
    diffs = []
    for c in goal_cols:
        a_val = parse_value(a_row[c])
        b_val = parse_value(b_row[c])
        if a_val is None or b_val is None:
            continue
        diffs.append({"feature": c, "a": a_val, "b": b_val})
    return diffs

def compute_tradeoffs(a_row: Dict[str, str], b_row: Dict[str, str], goal_cols: List[str]) -> List[str]:
    """Return list of goal columns where b is better than a."""
    trades = []
    for c in goal_cols:
        a_val = parse_value(a_row[c])
        b_val = parse_value(b_row[c])
        if a_val is None or b_val is None:
            continue
        if c.endswith('-'):  # minimize
            if b_val < a_val:
                trades.append(c)
        else:  # maximize
            if b_val > a_val:
                trades.append(c)
    return trades

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    # Identify goal columns
    goal_cols = [c for c in header if is_goal(c)]

    # Compute global min/max for goals
    mins, maxs = compute_global_minmax(rows, goal_cols)

    # Budget: at most min(30, len(rows)//5)
    budget = min(30, len(rows) // 5)

    # ---------------- CAR-11 ----------------
    print("== CAR-11 ==")
    rec_row_num, neighbors = get_recommended_and_neighbors(rows, mins, maxs, goal_cols, budget)

    if rec_row_num == 0:
        print("No evaluable rows found.")
    else:
        rec_score = score_row(rows[rec_row_num-1], mins, maxs, goal_cols)
        print(f"Recommended row: {rec_row_num}")
        print(f"Score: {rec_score:.4f}")
        print("Neighbors:")
        for n in neighbors:
            vals_str = ", ".join(f"{k}={v}" for k, v in n["values"].items())
            print(f"  Row {n['row']}: {vals_str}")

    # Machine‑checked answer
    answer_car11 = {
        "recommended": rec_row_num,
        "neighbors": neighbors
    }
    print(f"ANSWER CAR-11 {json.dumps(answer_car11)}")

    # ---------------- CAR-12 ----------------
    print("\n== CAR-12 ==")
    a_idx, b_idx, a_score, b_score = find_conflicting_pair(rows, mins, maxs, goal_cols)

    if a_idx == 0 or b_idx == 0:
        print("Insufficient data for comparison.")
    else:
        # Determine preferred
        preferred = a_idx if a_score >= b_score else b_idx
        a_row = rows[a_idx-1]
        b_row = rows[b_idx-1]
        diff_list = compute_differences(a_row, b_row, goal_cols)
        tradeoff_list = compute_tradeoffs(a_row, b_row, goal_cols)

        print(f"Row {a_idx} score: {a_score:.4f}")
        print(f"Row {b_idx} score: {b_score:.4f}")
        print(f"Preferred: Row {preferred}")
        print("Differences:")
        for d in diff_list:
            print(f"  {d['feature']}: a={d['a']:.4f}, b={d['b']:.4f}")
        if tradeoff_list:
            print("Trade‑offs (better in non‑preferred row):")
            print("  " + ", ".join(tradeoff_list))
        else:
            print("No trade‑offs found.")

    # Machine‑checked answer
    answer_car12 = {
        "a": a_idx,
        "b": b_idx,
        "preferred": preferred,
        "scores": {str(a_idx): round(a_score, 4), str(b_idx): round(b_score, 4)},
        "differences": diff_list,
        "tradeoffs": tradeoff_list
    }
    print(f"ANSWER CAR-12 {json.dumps(answer_car12)}")

if __name__ == "__main__":
    main()
