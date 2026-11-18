#!/usr/bin/env python3
"""
skill_09.py <csv_path>
Implements CAR-17 (unusual input combinations) and CAR-18 (unexpected goals).
"""

import csv
import json
import math
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def parse_value(v: str) -> Optional[float]:
    if v == '?' or v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None

def load_rows(csv_path: str) -> List[Tuple[int, Dict[str, Optional[float]]]]:
    rows: List[Tuple[int, Dict[str, Optional[float]]]] = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            parsed = {k: parse_value(v) for k, v in row.items()}
            rows.append((i, parsed))
    return rows

def goal_columns(headers: List[str]) -> List[str]:
    return [h for h in headers if h.endswith('+') or h.endswith('-')]

def input_columns(headers: List[str]) -> List[str]:
    return [h for h in headers if not h.endswith('+') and not h.endswith('-') and not h.endswith('X')]

# --------------------------------------------------------------------------- #
# CAR-17: unusual input combinations
# --------------------------------------------------------------------------- #

def compute_input_means(rows: List[Tuple[int, Dict[str, Optional[float]]]],
                       inputs: List[str]) -> Dict[str, float]:
    sums = defaultdict(float)
    counts = defaultdict(int)
    for _, data in rows:
        for col in inputs:
            v = data.get(col)
            if v is not None:
                sums[col] += v
                counts[col] += 1
    means = {col: sums[col] / counts[col] if counts[col] > 0 else 0.0 for col in inputs}
    return means

def distance_to_mean(row: Dict[str, Optional[float]],
                     means: Dict[str, float],
                     inputs: List[str]) -> float:
    sq = 0.0
    cnt = 0
    for col in inputs:
        v = row.get(col)
        if v is not None:
            d = v - means[col]
            sq += d * d
            cnt += 1
    if cnt == 0:
        return float('inf')
    return math.sqrt(sq)

def car_17(rows: List[Tuple[int, Dict[str, Optional[float]]]],
           headers: List[str]) -> Dict:
    inputs = input_columns(headers)
    means = compute_input_means(rows, inputs)

    distances = {}
    for num, data in rows:
        dist = distance_to_mean(data, means, inputs)
        distances[num] = dist

    vals = [d for d in distances.values() if not math.isinf(d)]
    if not vals:
        return {"flagged": [], "scores": {}}

    mean_d = sum(vals) / len(vals)
    var_d = sum((d - mean_d) ** 2 for d in vals) / len(vals)
    std_d = math.sqrt(var_d)

    threshold = mean_d + 2 * std_d  # flag 2 std above mean

    flagged = [num for num, d in distances.items() if d >= threshold]
    flagged.sort()
    flagged = flagged[:40]

    scores = {str(num): distances[num] for num in flagged}
    return {"flagged": flagged, "scores": scores}

# --------------------------------------------------------------------------- #
# CAR-18: unexpected goals relative to similar vehicles
# --------------------------------------------------------------------------- #

def euclidean_distance(a: Dict[str, Optional[float]],
                       b: Dict[str, Optional[float]],
                       inputs: List[str]) -> float:
    sq = 0.0
    cnt = 0
    for col in inputs:
        va = a.get(col)
        vb = b.get(col)
        if va is not None and vb is not None:
            diff = va - vb
            sq += diff * diff
            cnt += 1
    if cnt == 0:
        return float('inf')
    return math.sqrt(sq)

def car_18(rows: List[Tuple[int, Dict[str, Optional[float]]]],
           headers: List[str]) -> Dict:
    inputs = input_columns(headers)
    goals = goal_columns(headers)

    # Build list of rows with available goal values
    rows_with_goal: Dict[str, List[Tuple[int, Dict[str, Optional[float]]]]] = {g: [] for g in goals}
    for num, data in rows:
        for g in goals:
            if data.get(g) is not None:
                rows_with_goal[g].append((num, data))

    flagged_entries = []

    # For each goal separately
    for g in goals:
        all_rows = rows_with_goal[g]
        if len(all_rows) < 3:
            continue

        # Precompute distances from each row to others (k nearest neighbors)
        for num, data in all_rows:
            # Find k nearest neighbors with the same goal present
            neigh = []
            for n2, d2 in all_rows:
                if n2 == num:
                    continue
                dist = euclidean_distance(data, d2, inputs)
                neigh.append((dist, n2, d2))
            neigh.sort(key=lambda x: x[0])
            # Take first 3 neighbors (or fewer if not available)
            k = min(3, len(neigh))
            if k == 0:
                continue
            neigh_vals = [rows_with_goal[g][i][1][g] for i in range(k)]
            expected = sum(neigh_vals) / k
            actual = data[g]
            diff = abs(actual - expected)
            flagged_entries.append({
                "row": num,
                "column": g,
                "actual": actual,
                "expected": expected,
                "diff": diff
            })

    # Determine threshold: mean + 2 std of diff
    diffs = [e["diff"] for e in flagged_entries]
    if not diffs:
        return {"flagged": []}
    mean_diff = sum(diffs) / len(diffs)
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)
    std_diff = math.sqrt(var_diff)
    threshold = mean_diff + 2 * std_diff

    flagged = [e for e in flagged_entries if e["diff"] >= threshold]
    flagged.sort(key=lambda x: -x["diff"])
    flagged = flagged[:40]

    # Remove helper 'diff' field
    for e in flagged:
        del e["diff"]

    return {"flagged": flagged}

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    rows = load_rows(csv_path)
    if not rows:
        print("No data.")
        return
    headers = list(rows[0][1].keys())

    # CAR-17
    print("== CAR-17 ==")
    res17 = car_17(rows, headers)
    print(f"Unusual rows: {res17['flagged']}")
    print(f"Scores: {res17['scores']}")
    print(f"ANSWER CAR-17 {json.dumps(res17)}")

    # CAR-18
    print("\n== CAR-18 ==")
    res18 = car_18(rows, headers)
    print(f"Flagged entries (up to 40):")
    for e in res18["flagged"]:
        print(f"  Row {e['row']} {e['column']}: actual={e['actual']:.4f}, expected={e['expected']:.4f}")
    print(f"ANSWER CAR-18 {json.dumps(res18)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_09.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])
