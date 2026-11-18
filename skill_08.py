#!/usr/bin/env python3
"""
skill_08.py <csv_path>
Implements CAR-15 (poor vehicle alternatives) and CAR-16 (closest better alternative).
"""

import csv
import json
import math
import sys
from typing import Dict, List, Tuple, Optional

# --------------------------------------------------------------------------- #
# Basic helpers
# --------------------------------------------------------------------------- #

def parse_value(v: str) -> Optional[float]:
    """Convert a CSV cell to a float or None."""
    if v == '?' or v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None

def load_rows(csv_path: str) -> List[Tuple[int, Dict[str, Optional[float]]]]:
    """Read CSV and return list of (row_number, data_dict)."""
    rows: List[Tuple[int, Dict[str, Optional[float]]]] = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            parsed: Dict[str, Optional[float]] = {k: parse_value(v) for k, v in row.items()}
            rows.append((i, parsed))
    return rows

# --------------------------------------------------------------------------- #
# Goal / input column classification
# --------------------------------------------------------------------------- #

def goal_columns(headers: List[str]) -> List[str]:
    """Columns that end with '+' or '-'.  These are goals."""
    return [h for h in headers if h.endswith('+') or h.endswith('-')]

def input_columns(headers: List[str]) -> List[str]:
    """Columns that are not goals and do not end with 'X'."""
    return [h for h in headers if not h.endswith('+') and not h.endswith('-') and not h.endswith('X')]

# --------------------------------------------------------------------------- #
# Composite scoring
# --------------------------------------------------------------------------- #

def compute_min_max(rows: List[Tuple[int, Dict[str, Optional[float]]]],
                   goals: List[str]) -> Dict[str, Tuple[float, float]]:
    minmax: Dict[str, Tuple[float, float]] = {}
    for g in goals:
        values = [row[1][g] for row in rows if row[1].get(g) is not None]
        if values:
            minmax[g] = (min(values), max(values))
    return minmax

def composite_score(row: Dict[str, Optional[float]],
                    minmax: Dict[str, Tuple[float, float]],
                    goals: List[str]) -> Optional[float]:
    norms = []
    for g in goals:
        v = row.get(g)
        if v is None or g not in minmax:
            return None
        mn, mx = minmax[g]
        if mn == mx:
            norm = 0.0
        else:
            norm = (v - mn) / (mx - mn) if g.endswith('+') else (mx - v) / (mx - mn)
        norms.append(norm)
    return sum(norms) / len(norms)

# --------------------------------------------------------------------------- #
# Distance in input space
# --------------------------------------------------------------------------- #

def input_distance(a: Dict[str, Optional[float]],
                   b: Dict[str, Optional[float]],
                   inputs: List[str]) -> float:
    """Euclidean distance ignoring missing values."""
    sq = 0.0
    cnt = 0
    for col in inputs:
        va = a.get(col)
        vb = b.get(col)
        if va is None or vb is None:
            continue
        diff = va - vb
        sq += diff * diff
        cnt += 1
    if cnt == 0:
        return float('inf')
    return math.sqrt(sq)

# --------------------------------------------------------------------------- #
# CAR-15 helpers
# --------------------------------------------------------------------------- #

def find_poor_row(rows: List[Tuple[int, Dict[str, Optional[float]]]],
                 minmax: Dict[str, Tuple[float, float]],
                 goals: List[str]) -> Tuple[int, Dict[str, Optional[float]], float]:
    """Return the row number, data, and composite score of the worst performing vehicle."""
    worst: Optional[Tuple[int, float]] = None
    worst_data: Optional[Dict[str, Optional[float]]] = None
    for num, data in rows:
        score = composite_score(data, minmax, goals)
        if score is None:
            continue
        if worst is None or score < worst[1]:
            worst = (num, score)
            worst_data = data
    if worst is None:
        raise RuntimeError("No vehicle with complete goal data found.")
    return worst[0], worst_data, worst[1]

def alternatives_for_poor(rows: List[Tuple[int, Dict[str, Optional[float]]]],
                          minmax: Dict[str, Tuple[float, float]],
                          goals: List[str],
                          inputs: List[str],
                          poor_num: int,
                          poor_data: Dict[str, Optional[float]],
                          poor_score: float) -> List[Dict]:
    """Find alternative vehicles that are similar and better."""
    candidates = []
    for num, data in rows:
        if num == poor_num:
            continue
        score = composite_score(data, minmax, goals)
        if score is None or score <= poor_score:
            continue
        dist = input_distance(poor_data, data, inputs)
        if math.isinf(dist):
            continue
        candidates.append((dist, num, data, score))
    # sort by distance then by score descending
    candidates.sort(key=lambda x: (x[0], -x[3]))
    alternatives = []
    # Pick up to 3 alternatives
    for _, num, data, _ in candidates[:3]:
        changes = {}
        for col in inputs:
            va = poor_data.get(col)
            vb = data.get(col)
            if va is None or vb is None:
                if va != vb:
                    changes[col] = [va, vb]
            elif va != vb:
                changes[col] = [va, vb]
        alternatives.append({"row": num, "changes": changes})
    return alternatives

# --------------------------------------------------------------------------- #
# CAR-16 helpers
# --------------------------------------------------------------------------- #

def compute_goal_thresholds(rows: List[Tuple[int, Dict[str, Optional[float]]]],
                           goals: List[str]) -> Dict[str, float]:
    """Return mean value for each goal column as threshold."""
    thresholds: Dict[str, float] = {}
    for g in goals:
        vals = [row[1][g] for row in rows if row[1].get(g) is not None]
        if vals:
            thresholds[g] = sum(vals) / len(vals)
    return thresholds

def meets_target(row: Dict[str, Optional[float]],
                 thresholds: Dict[str, float]) -> Dict[str, bool]:
    """Return dict of goal -> True/False whether meets threshold."""
    result: Dict[str, bool] = {}
    for g, th in thresholds.items():
        val = row.get(g)
        if val is None:
            result[g] = False
            continue
        if g.endswith('+'):
            result[g] = val >= th
        else:
            result[g] = val <= th
    return result

def find_row_with_missing_target(rows: List[Tuple[int, Dict[str, Optional[float]]]],
                                 thresholds: Dict[str, float]) -> Tuple[int, Dict[str, Optional[float]], Dict[str, bool]]:
    """Return the first row that fails at least one target."""
    for num, data in rows:
        target_res = meets_target(data, thresholds)
        if not all(target_res.values()):
            return num, data, target_res
    # If none found, just take first row
    return rows[0][0], rows[0][1], meets_target(rows[0][1], thresholds)

def best_alternative(rows: List[Tuple[int, Dict[str, Optional[float]]]],
                     thresholds: Dict[str, float],
                     inputs: List[str],
                     base_num: int,
                     base_data: Dict[str, Optional[float]],
                     base_targets: Dict[str, bool]) -> Tuple[int, List[str], int]:
    """Find the alternative that meets more targets and differs in fewest inputs."""
    base_met = sum(1 for v in base_targets.values() if v)
    best_num = None
    best_diff_count = None
    best_diff_cols: List[str] = []
    best_met = -1

    for num, data in rows:
        if num == base_num:
            continue
        if any(data.get(g) is None for g in thresholds):
            continue
        targ_res = meets_target(data, thresholds)
        met_count = sum(1 for v in targ_res.values() if v)
        if met_count <= base_met:
            continue  # need more targets met
        # Count input differences
        diff_cols = [col for col in inputs if base_data.get(col) != data.get(col)]
        diff_count = len(diff_cols)
        # Pick better candidate
        if (met_count > best_met or
            (met_count == best_met and diff_count < best_diff_count) or
            (met_count == best_met and diff_count == best_diff_count and num < best_num)):
            best_met = met_count
            best_num = num
            best_diff_cols = diff_cols
            best_diff_count = diff_count
    return best_num, best_diff_cols, best_met

# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    rows = load_rows(csv_path)
    if not rows:
        print("No data.")
        return

    headers = list(rows[0][1].keys())
    goals = goal_columns(headers)
    inputs = input_columns(headers)

    minmax = compute_min_max(rows, goals)

    # --------------------- CAR-15 --------------------- #
    print("== CAR-15 ==")
    poor_num, poor_data, poor_score = find_poor_row(rows, minmax, goals)
    alternatives = alternatives_for_poor(rows, minmax, goals, inputs, poor_num, poor_data, poor_score)

    # Print human readable
    print(f"Poor vehicle: Row {poor_num} (composite={poor_score:.4f})")
    if alternatives:
        for alt in alternatives:
            print(f"  Alternative Row {alt['row']} with changes: {alt['changes']}")
    else:
        print("  No better alternatives found.")
    # JSON answer
    ans15 = {"row": poor_num, "alternatives": alternatives}
    print(f"ANSWER CAR-15 {json.dumps(ans15)}")

    # --------------------- CAR-16 --------------------- #
    print("\n== CAR-16 ==")
    thresholds = compute_goal_thresholds(rows, goals)
    base_num, base_data, base_targets = find_row_with_missing_target(rows, thresholds)
    alt_num, diff_cols, alt_met = best_alternative(rows, thresholds, inputs, base_num, base_data, base_targets)

    # Human readable
    print(f"Vehicle with missing targets: Row {base_num}")
    print(f"  Thresholds: {thresholds}")
    print(f"  Target status: {base_targets}")
    if alt_num is None:
        print("  No better alternative found.")
        alt_num = None
        diff_cols = []
    else:
        print(f"  Best alternative: Row {alt_num} (met {alt_met} targets)")
        print(f"  Changed inputs: {diff_cols}")

    ans16 = {
        "row": base_num,
        "targets": thresholds,
        "alternative": alt_num,
        "changed": diff_cols
    }
    print(f"ANSWER CAR-16 {json.dumps(ans16)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_08.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])
