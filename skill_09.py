#!/usr/bin/env python3
"""
Skill 09: Unusual vehicle combinations (CAR-17) and outlier performance (CAR-18).

The program follows the same data conventions as the previous skills.
It prints two sections, each starting with a line ``== <ID> ==`` and
provides machine‑checked JSON answers.

Author: OpenAI ChatGPT
"""

import sys
import csv
import json
import math
from typing import List, Dict, Tuple, Any, Optional

# --------------------------------------------------------------------------- #
# Parsing utilities
# --------------------------------------------------------------------------- #

def parse_value(v: str) -> Optional[float]:
    """Return float value or None for missing data."""
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
# Statistics helpers
# --------------------------------------------------------------------------- #

def column_stats(rows: List[Dict[str, str]], cols: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute mean and standard deviation for each column."""
    means = {}
    stds  = {}
    for col in cols:
        values = [parse_value(r[col]) for r in rows if parse_value(r[col]) is not None]
        if not values:
            means[col] = stds[col] = 0.0
            continue
        m = sum(values) / len(values)
        means[col] = m
        # population standard deviation
        s2 = sum((x - m) ** 2 for x in values) / len(values)
        stds[col] = math.sqrt(s2)
    return means, stds

# --------------------------------------------------------------------------- #
# Normalised input representation
# --------------------------------------------------------------------------- #

def compute_normalised_inputs(rows: List[Dict[str, str]],
                              input_cols: List[str],
                              means: Dict[str, float],
                              stds: Dict[str, float]) -> List[Dict[str, Any]]:
    """Return list of dicts with 'idx', 'norm', 'score', 'complete' flag."""
    data = []
    for idx, row in enumerate(rows, start=1):
        norm_vals = []
        complete = True
        for col in input_cols:
            val = parse_value(row[col])
            if val is None:
                complete = False
                continue
            sd = stds[col]
            z = 0.0 if sd == 0 else (val - means[col]) / sd
            norm_vals.append(z)
        score = sum(z ** 2 for z in norm_vals)  # Mahalanobis-like, ignoring covariance
        data.append({"idx": idx, "norm": norm_vals, "score": score, "complete": complete})
    return data

# --------------------------------------------------------------------------- #
# CAR-17: Unusual combinations
# --------------------------------------------------------------------------- #

def car17(rows: List[Dict[str, str]],
          input_cols: List[str],
          means: Dict[str, float],
          stds: Dict[str, float]) -> Dict[str, Any]:
    norm_data = compute_normalised_inputs(rows, input_cols, means, stds)

    # Flag top 40 rows with highest scores
    sorted_by_score = sorted(norm_data, key=lambda d: (-d["score"], d["idx"]))
    flagged = [d["idx"] for d in sorted_by_score[:40]]
    scores = {str(d["idx"]): round(d["score"], 4) for d in sorted_by_score[:40]}

    answer = {"flagged": flagged, "scores": scores}
    return answer

# --------------------------------------------------------------------------- #
# CAR-18: Outlier performance
# --------------------------------------------------------------------------- #

def car18(rows: List[Dict[str, str]],
          input_cols: List[str],
          goal_cols: List[str],
          means: Dict[str, float],
          stds: Dict[str, float],
          goal_means: Dict[str, float],
          goal_stds: Dict[str, float]) -> Dict[str, Any]:
    # Build list of rows with complete input data (needed for distance calculations)
    complete_rows = []
    idx_map = {}  # map idx to position in complete_rows
    for i, row in enumerate(rows, start=1):
        complete = True
        for col in input_cols:
            if parse_value(row[col]) is None:
                complete = False
                break
        if complete:
            idx_map[i] = len(complete_rows)
            # normalised input vector
            norm_vec = []
            for col in input_cols:
                val = parse_value(row[col])
                sd = stds[col]
                z = 0.0 if sd == 0 else (val - means[col]) / sd
                norm_vec.append(z)
            complete_rows.append({"idx": i, "norm": norm_vec})

    # Pre‑compute pairwise distances (squared) between complete rows
    n = len(complete_rows)
    dist_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        vi = complete_rows[i]["norm"]
        for j in range(i+1, n):
            vj = complete_rows[j]["norm"]
            d2 = sum((a - b) ** 2 for a, b in zip(vi, vj))
            dist_matrix[i][j] = d2
            dist_matrix[j][i] = d2

    flagged_entries = []

    # Helper to find k nearest neighbors (excluding self)
    def nearest_k(idx_row: int, k: int = 5) -> List[int]:
        dists = [(dist_matrix[idx_row][j], j) for j in range(n) if j != idx_row]
        dists.sort(key=lambda x: x[0])
        return [j for _, j in dists[:k]]

    # For each target column separately
    for target in goal_cols:
        threshold = 2 * goal_stds.get(target, 0.0)  # 2 std dev
        if threshold == 0.0:
            continue  # cannot flag if no variance
        for r in rows:
            actual = parse_value(r[target])
            if actual is None:
                continue
            idx = int(r["_idx"])  # will set below
        # We'll loop over complete rows, as others cannot have neighbors

    # Re‑create mapping of idx to row dict with index field
    for i, r in enumerate(rows, start=1):
        r["_idx"] = i

    for target in goal_cols:
        threshold = 2 * goal_stds.get(target, 0.0)
        if threshold == 0.0:
            continue
        for r in rows:
            actual = parse_value(r[target])
            if actual is None:
                continue
            idx = int(r["_idx"])
            # Need neighbors only if this row has complete inputs
            if idx not in idx_map:
                continue
            i_pos = idx_map[idx]
            neigh_idx = nearest_k(i_pos, k=5)
            neigh_vals = []
            for ni in neigh_idx:
                neigh_actual = parse_value(rows[complete_rows[ni]["idx"] - 1][target])
                if neigh_actual is not None:
                    neigh_vals.append(neigh_actual)
            if not neigh_vals:
                continue
            mean_neigh = sum(neigh_vals) / len(neigh_vals)
            diff = abs(actual - mean_neigh)
            if diff > threshold:
                flagged_entries.append({
                    "row": idx,
                    "column": target,
                    "actual": round(actual, 4),
                    "expected": round(mean_neigh, 4)
                })

    # Keep top 40 entries by absolute difference
    flagged_entries.sort(key=lambda e: (-abs(e["actual"] - e["expected"]), e["row"]))
    flagged_entries = flagged_entries[:40]

    answer = {"flagged": flagged_entries}
    return answer

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]

    # Identify columns
    goal_cols = [c for c in header if is_goal(c)]
    input_cols = [c for c in header if is_input(c)]

    # Compute stats
    input_means, input_stds = column_stats(rows, input_cols)
    goal_means, goal_stds = column_stats(rows, goal_cols)

    # ==================== CAR-17 ====================
    print("== CAR-17 ==")
    # Summary lines
    print(f"Total rows: {len(rows)}")
    print(f"Input columns: {len(input_cols)} ({', '.join(input_cols)})")
    car17_answer = car17(rows, input_cols, input_means, input_stds)
    print(f"Flagged rows: {len(car17_answer['flagged'])}")
    print(f"Example score: {next(iter(car17_answer['scores'].values())) if car17_answer['scores'] else 'none'}")
    # Machine‑checked answer
    print(f"ANSWER CAR-17 {json.dumps(car17_answer)}")

    # ==================== CAR-18 ====================
    print("\n== CAR-18 ==")
    print(f"Total rows: {len(rows)}")
    print(f"Target columns: {', '.join(goal_cols)}")
    car18_answer = car18(rows, input_cols, goal_cols, input_means, input_stds, goal_means, goal_stds)
    print(f"Flagged entries: {len(car18_answer['flagged'])}")
    if car18_answer["flagged"]:
        ex = car18_answer["flagged"][0]
        print(f"Example flag: row={ex['row']}, col={ex['column']}, act={ex['actual']}, exp={ex['expected']}")
    # Machine‑checked answer
    print(f"ANSWER CAR-18 {json.dumps(car18_answer)}")

if __name__ == "__main__":
    main()
