#!/usr/bin/env python3
"""
skill_09.py

NASA-17: Flag historical projects whose combinations of project characteristics
        are unusually different from the rest of our records so that we can
        review them.

NASA-18: Identify project records whose reported effort or defect counts
        differ substantially from what comparable projects would lead us to expect,
        so we can investigate possible measurement or recording issues.

Usage:
    python skill_09.py <csv_path>
"""

import csv
import json
import math
import sys
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def parse_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read CSV and return rows as list of dicts and header list."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = [row for row in reader]
    return rows, header

def is_numeric(val: str) -> bool:
    if val.strip() == "?":
        return False
    try:
        float(val)
        return True
    except ValueError:
        return False

def to_float(val: str) -> float:
    if val.strip() == "?":
        return float("nan")
    try:
        return float(val)
    except ValueError:
        return float("nan")

# ------------------------------------------------------------------
# NASA-17
# ------------------------------------------------------------------
def compute_numeric_stats(rows: List[Dict[str, str]], numeric_cols: List[str]) -> Dict[str, Tuple[float, float]]:
    """Compute mean and std for each numeric column."""
    stats = {}
    for col in numeric_cols:
        vals = [to_float(r[col]) for r in rows if is_numeric(r[col])]
        if not vals:
            stats[col] = (0.0, 0.0)
            continue
        mean = sum(vals) / len(vals)
        if len(vals) == 1:
            std = 0.0
        else:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = math.sqrt(var)
        stats[col] = (mean, std)
    return stats

def compute_unusualness(rows: List[Dict[str, str]], numeric_cols: List[str],
                       stats: Dict[str, Tuple[float, float]]) -> Dict[int, float]:
    """Compute unusualness score for each row."""
    scores = {}
    for idx, r in enumerate(rows, start=1):
        zsum_sq = 0.0
        for col in numeric_cols:
            if not is_numeric(r[col]):
                continue
            val = to_float(r[col])
            mean, std = stats[col]
            if std == 0:
                continue
            z = (val - mean) / std
            zsum_sq += z * z
        if zsum_sq > 0:
            scores[idx] = round(zsum_sq, 4)
    return scores

# ------------------------------------------------------------------
# NASA-18
# ------------------------------------------------------------------
def euclidean_distance(r1: Dict[str, str], r2: Dict[str, str], numeric_cols: List[str]) -> float:
    sq = 0.0
    for col in numeric_cols:
        v1 = r1.get(col, "?")
        v2 = r2.get(col, "?")
        if v1.strip() == "?" or v2.strip() == "?":
            continue
        if not is_numeric(v1) or not is_numeric(v2):
            continue
        d = to_float(v1) - to_float(v2)
        sq += d * d
    return math.sqrt(sq)

def k_nearest_neighbors(rows: List[Dict[str, str]], target_row: Dict[str, str],
                        numeric_cols: List[str], k: int = 3) -> List[Dict[str, str]]:
    distances = []
    for r in rows:
        if r is target_row:
            continue
        d = euclidean_distance(target_row, r, numeric_cols)
        distances.append((d, r))
    distances.sort(key=lambda x: x[0])
    return [r for _, r in distances[:k]]

def compute_expected_targets(rows: List[Dict[str, str]], target_row: Dict[str, str],
                             numeric_cols: List[str]) -> Dict[str, float]:
    """Return expected Effort- and Defects- based on kNN."""
    neighs = k_nearest_neighbors(rows, target_row, numeric_cols, k=3)
    exp = {}
    for col in ["Effort-", "Defects-"]:
        vals = [to_float(r[col]) for r in neighs if not math.isnan(to_float(r[col]))]
        if vals:
            exp[col] = sum(vals) / len(vals)
    return exp

def collect_differences(rows: List[Dict[str, str]], numeric_cols: List[str]) -> Dict[str, List[Tuple[int, float, float]]]:
    """Return list of (row, diff, expected) for each target."""
    diffs = {"Effort-": [], "Defects-": []}
    for idx, r in enumerate(rows, start=1):
        actuals = {}
        for col in ["Effort-", "Defects-"]:
            val = to_float(r[col])
            if not math.isnan(val):
                actuals[col] = val
        if not actuals:
            continue
        expected = compute_expected_targets(rows, r, numeric_cols)
        for col, act in actuals.items():
            exp = expected.get(col, None)
            if exp is None:
                continue
            diff = abs(act - exp)
            diffs[col].append((idx, diff, exp))
    return diffs

def compute_mean_std(vals: List[float]) -> Tuple[float, float]:
    n = len(vals)
    if n == 0:
        return (0.0, 0.0)
    mean = sum(vals) / n
    if n == 1:
        return (mean, 0.0)
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return (mean, math.sqrt(var))

def select_flagged(diffs: Dict[str, List[Tuple[int, float, float]]], limit: int = 10) -> List[Dict[str, object]]:
    """Select up to limit rows that exceed mean + 2*std for either target."""
    flagged = []
    for col, lst in diffs.items():
        if not lst:
            continue
        diffs_vals = [d for _, d, _ in lst]
        mean, std = compute_mean_std(diffs_vals)
        threshold = mean + 2 * std
        for idx, diff, exp in lst:
            if diff > threshold:
                flagged.append({"row": idx, "column": col,
                                "actual": round(to_float(rows[idx-1][col]), 4),
                                "expected": round(exp, 4)})
    # sort by diff descending (we don't have diff here; re-sort by expected? use list above)
    # We already appended in original order; let's sort by diff descending.
    flagged.sort(key=lambda x: abs(x["actual"] - x["expected"]), reverse=True)
    return flagged[:limit]

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(argv: List[str]) -> None:
    if len(argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)

    csv_path = argv[1]
    rows, header = parse_csv(csv_path)

    # Identify goal and X columns
    goal_cols = {c for c in header if c.endswith("+") or c.endswith("-")}
    x_cols = {c for c in header if c.endswith("X")}

    # Feature columns: non-goal, non-X
    feature_cols = [c for c in header if c not in goal_cols and c not in x_cols]

    # Numeric columns among features
    numeric_cols = [c for c in feature_cols if any(is_numeric(row[c]) for row in rows)]

    # ------------------------------------------------------------------
    # NASA-17
    # ------------------------------------------------------------------
    stats = compute_numeric_stats(rows, numeric_cols)
    unusual_scores = compute_unusualness(rows, numeric_cols, stats)
    # Take top 10 rows with highest score
    sorted_scores = sorted(unusual_scores.items(), key=lambda x: x[1], reverse=True)
    flagged_17 = [idx for idx, _ in sorted_scores[:10]]
    scores_dict = {str(idx): unusual_scores[idx] for idx in flagged_17}

    print("== NASA-17 ==")
    # At least three lines of results
    print(f"Flagged rows (top 10): {flagged_17}")
    print(f"Unusualness scores: {scores_dict}")
    print(f"Total flagged: {len(flagged_17)}")
    # Answer line
    answer_17 = {"flagged": flagged_17, "scores": scores_dict}
    print("ANSWER NASA-17 " + json.dumps(answer_17))

    # ------------------------------------------------------------------
    # NASA-18
    # ------------------------------------------------------------------
    diffs = collect_differences(rows, numeric_cols)
    flagged_18 = select_flagged(diffs, limit=10)

    print("\n== NASA-18 ==")
    # At least three lines
    print(f"Flagged entries: {flagged_18}")
    print(f"Number of flagged records: {len(flagged_18)}")
    # Show each flagged entry
    for entry in flagged_18:
        print(f"Row {entry['row']}, column {entry['column']}: actual={entry['actual']} vs expected={entry['expected']}")

    answer_18 = {"flagged": flagged_18}
    print("ANSWER NASA-18 " + json.dumps(answer_18))

if __name__ == "__main__":
    main(sys.argv)
