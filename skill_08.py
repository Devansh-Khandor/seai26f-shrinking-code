#!/usr/bin/env python3
"""
skill_08.py: Implements CAR-15 and CAR-16 for used‑car dealership data.

Usage:
    python skill_08.py <csv_path>
"""

import csv
import json
import math
import sys
from typing import List, Tuple, Dict, Any, Optional

# ------------------------------------------------------------------
# CSV reading utilities
# ------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[str], List[List[Any]]]:
    """Read CSV, convert numeric cells, keep '?' as None."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows: List[List[Any]] = []
        for row in reader:
            conv_row: List[Any] = []
            for v in row:
                if v == "?":
                    conv_row.append(None)
                else:
                    try:
                        if "." in v or "e" in v.lower():
                            conv_row.append(float(v))
                        else:
                            conv_row.append(int(v))
                    except ValueError:
                        conv_row.append(v)
            rows.append(conv_row)
    return header, rows

# ------------------------------------------------------------------
# Feature / goal helpers
# ------------------------------------------------------------------
def goal_indices(header: List[str]) -> List[int]:
    """Indices of goal columns (ending with '+' or '-')."""
    return [i for i, h in enumerate(header) if h.endswith(("+", "-"))]

def input_indices(header: List[str]) -> List[int]:
    """Indices of input columns (neither goal nor X)."""
    return [i for i, h in enumerate(header) if not h.endswith(("X", "+", "-"))]

def is_max_goal(header: List[str]) -> Dict[int, bool]:
    return {i: header[i].endswith("+") for i in goal_indices(header)}

# ------------------------------------------------------------------
# Min/Max computation
# ------------------------------------------------------------------
def compute_min_max(rows: List[List[Any]], cols: List[int]) -> Tuple[Dict[int, float], Dict[int, float]]:
    mins, maxs = {}, {}
    for i in cols:
        vals = [r[i] for r in rows if r[i] is not None]
        if vals:
            mins[i] = min(vals)
            maxs[i] = max(vals)
        else:
            mins[i] = maxs[i] = None
    return mins, maxs

# ------------------------------------------------------------------
# Normalization helpers
# ------------------------------------------------------------------
def normalize_goal(value: Any, mn: float, mx: float, is_max: bool) -> float:
    """Return 0-1 normalized value for a goal."""
    if mn is None or mx is None or mx == mn:
        return 0.5
    if is_max:
        return (value - mn) / (mx - mn)
    else:
        return (mx - value) / (mx - mn)

def score_row(row: List[Any], goal_idx: List[int], mins: Dict[int, float],
              maxs: Dict[int, float], is_max: Dict[int, bool]) -> float:
    """Sum of normalized goal values for a row."""
    score = 0.0
    for i in goal_idx:
        v = row[i]
        if v is None:
            continue
        score += normalize_goal(v, mins[i], maxs[i], is_max[i])
    return score

# ------------------------------------------------------------------
# Thresholds for preferred performance
# ------------------------------------------------------------------
def compute_thresholds(rows: List[List[Any]], goal_idx: List[int], is_max: Dict[int, bool]) -> Dict[int, float]:
    """For maximize goals take 75th percentile, for minimize 25th percentile."""
    thresholds = {}
    for i in goal_idx:
        vals = sorted([r[i] for r in rows if r[i] is not None])
        n = len(vals)
        if n == 0:
            thresholds[i] = None
            continue
        if is_max[i]:
            # 75th percentile
            pos = int(math.floor(0.75 * n))
            pos = max(0, min(n - 1, pos))
            thresholds[i] = vals[pos]
        else:
            # 25th percentile
            pos = int(math.floor(0.25 * n))
            pos = max(0, min(n - 1, pos))
            thresholds[i] = vals[pos]
    return thresholds

def meets_threshold(row: List[Any], goal_idx: List[int],
                    thresholds: Dict[int, float], is_max: Dict[int, bool]) -> bool:
    """Return True if row meets all thresholds."""
    for i in goal_idx:
        v = row[i]
        if v is None or thresholds[i] is None:
            return False
        if is_max[i]:
            if v < thresholds[i] - 1e-12:
                return False
        else:
            if v > thresholds[i] + 1e-12:
                return False
    return True

# ------------------------------------------------------------------
# Distance between rows over input features
# ------------------------------------------------------------------
def compute_input_ranges(rows: List[List[Any]], inp_idx: List[int]) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Min and max for numeric inputs, only over rows provided."""
    mins, maxs = {}, {}
    for i in inp_idx:
        vals = [r[i] for r in rows if r[i] is not None and isinstance(r[i], (int, float))]
        if vals:
            mins[i] = min(vals)
            maxs[i] = max(vals)
        else:
            mins[i] = maxs[i] = None
    return mins, maxs

def distance_rows(r1: List[Any], r2: List[Any], inp_idx: List[int],
                  mins: Dict[int, float], maxs: Dict[int, float]) -> float:
    """Euclidean-like distance with categorical handling. Missing treated as max difference."""
    sq = 0.0
    used = 0
    for i in inp_idx:
        v1 = r1[i]
        v2 = r2[i]
        if v1 is None or v2 is None:
            # missing values contribute maximum difference (1)
            sq += 1.0
            used += 1
            continue
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            mn = mins[i]
            mx = maxs[i]
            if mn is None or mx is None or mx == mn:
                diff = 0.0
            else:
                diff = abs((v1 - mn) / (mx - mn) - (v2 - mn) / (mx - mn))
            sq += diff * diff
        else:
            diff = 0.0 if v1 == v2 else 1.0
            sq += diff
        used += 1
    if used == 0:
        return float("inf")
    return math.sqrt(sq)

# ------------------------------------------------------------------
# CAR-15 implementation
# ------------------------------------------------------------------
def car15(header: List[str], rows: List[List[Any]]) -> Dict[str, Any]:
    """Find a poorly rated vehicle and propose similar better alternatives."""
    goal_idx = goal_indices(header)
    inp_idx = input_indices(header)
    is_max = is_max_goal(header)

    mins_goal, maxs_goal = compute_min_max(rows, goal_idx)
    # Score each row
    scores = [score_row(r, goal_idx, mins_goal, maxs_goal, is_max) for r in rows]
    # Poorly rated vehicle: lowest score
    poor_idx = min(range(len(rows)), key=lambda i: scores[i])
    poor_row = rows[poor_idx]

    # Historical rows: all except arrivals (last round(0.2*n))
    n_rows = len(rows)
    arrivals_count = int(round(0.2 * n_rows))
    historical_rows = rows[:n_rows - arrivals_count]
    historical_indices = list(range(n_rows - arrivals_count))  # 0-based

    # Compute input ranges over historical for distance
    mins_inp, maxs_inp = compute_input_ranges(historical_rows, inp_idx)

    # Find alternatives with higher score
    alternatives = []
    for h_idx in historical_indices:
        hist_row = rows[h_idx]
        if scores[h_idx] <= scores[poor_idx]:
            continue
        dist = distance_rows(poor_row, hist_row, inp_idx, mins_inp, maxs_inp)
        alternatives.append((dist, h_idx))

    if not alternatives:
        # If none better, fallback to best historical
        alternatives = [(distance_rows(poor_row, rows[h_idx], inp_idx, mins_inp, maxs_inp), h_idx)
                        for h_idx in historical_indices]
    # Sort by distance
    alternatives.sort(key=lambda t: t[0])
    # Pick the best one
    alt_idx = alternatives[0][1]
    alt_row = rows[alt_idx]

    # Compute changes
    changes = {}
    for i in inp_idx:
        v1 = poor_row[i]
        v2 = alt_row[i]
        if v1 != v2:
            changes[header[i]] = [v1, v2]

    return {
        "row": poor_idx + 1,  # 1-based
        "alternatives": [
            {
                "row": alt_idx + 1,
                "changes": changes
            }
        ]
    }

# ------------------------------------------------------------------
# CAR-16 implementation
# ------------------------------------------------------------------
def car16(header: List[str], rows: List[List[Any]]) -> Dict[str, Any]:
    """Find a vehicle that does not meet preferred targets and suggest an alternative."""
    goal_idx = goal_indices(header)
    inp_idx = input_indices(header)
    is_max = is_max_goal(header)

    mins_goal, maxs_goal = compute_min_max(rows, goal_idx)
    thresholds = compute_thresholds(rows, goal_idx, is_max)

    # Pick a vehicle that fails at least one threshold
    bad_candidates = []
    for i, r in enumerate(rows):
        if not meets_threshold(r, goal_idx, thresholds, is_max):
            bad_candidates.append((i, r))
    if not bad_candidates:
        # If all meet thresholds, just pick worst
        scores = [score_row(r, goal_idx, mins_goal, maxs_goal, is_max) for r in rows]
        bad_idx = max(range(len(rows)), key=lambda i: scores[i])
        bad_row = rows[bad_idx]
    else:
        # pick the one with least number of thresholds met (most failing)
        def num_met(r):
            return sum(1 for i in goal_idx if
                       (r[i] is not None) and
                       (r[i] >= thresholds[i] if is_max[i] else r[i] <= thresholds[i]))
        bad_idx, bad_row = min(bad_candidates, key=lambda t: num_met(t[1]))

    # Determine how many thresholds the bad row meets
    bad_meets = [i for i in goal_idx if
                 (bad_row[i] is not None) and
                 ((bad_row[i] >= thresholds[i]) if is_max[i] else (bad_row[i] <= thresholds[i]))]
    bad_count = len(bad_meets)

    # Search among all other rows for alternative that meets more thresholds
    best_alt_idx: Optional[int] = None
    best_alt_meets: int = -1
    best_changed: int = math.inf
    best_row_obj = None

    for j, r in enumerate(rows):
        if j == bad_idx:
            continue
        meets = [i for i in goal_idx if
                 (r[i] is not None) and
                 ((r[i] >= thresholds[i]) if is_max[i] else (r[i] <= thresholds[i]))]
        count = len(meets)
        if count <= bad_count:
            continue
        # compute number of changed input columns
        changes = 0
        for i in inp_idx:
            if bad_row[i] != r[i]:
                changes += 1
        if count > best_alt_meets or (count == best_alt_meets and changes < best_changed):
            best_alt_idx = j
            best_alt_meets = count
            best_changed = changes
            best_row_obj = r

    if best_alt_idx is None:
        # If no better alternative, pick the best among all with same count
        best_alt_idx = 0
        best_row_obj = rows[0]

    # Build changed list
    changed_cols = [header[i] for i in inp_idx if bad_row[i] != best_row_obj[i]]

    # Prepare thresholds dict for JSON
    thresholds_json = {header[i]: thresholds[i] for i in goal_idx}

    return {
        "row": bad_idx + 1,
        "targets": thresholds_json,
        "alternative": best_alt_idx + 1,
        "changed": changed_cols
    }

# ------------------------------------------------------------------
# Main driver
# ------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    # CAR-15
    print("== CAR-15 ==")
    res15 = car15(header, rows)
    print(f"Poorly rated vehicle: row {res15['row']}")
    for alt in res15['alternatives']:
        print(f"Alternative row {alt['row']}")
        for col, vals in alt['changes'].items():
            print(f"  {col}: {vals[0]} -> {vals[1]}")
    print(f"ANSWER CAR-15 {json.dumps(res15)}")

    # CAR-16
    print("\n== CAR-16 ==")
    res16 = car16(header, rows)
    print(f"Vehicle not meeting targets: row {res16['row']}")
    print("Target thresholds:")
    for col, thr in res16['targets'].items():
        print(f"  {col}: {thr}")
    print(f"Alternative row: {res16['alternative']}")
    print("Changed input columns:")
    for col in res16['changed']:
        print(f"  {col}")
    print(f"ANSWER CAR-16 {json.dumps(res16)}")

if __name__ == "__main__":
    main()
