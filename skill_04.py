#!/usr/bin/env python3
"""
Skill 04: Online assessment of new arrivals and incremental inventory summaries.

Usage:
    python skill_04.py <csv_path>
"""

import sys
import csv
import json
import math
from typing import List, Dict, Tuple

# --------------------------------------------------------------------------- #
# Reuse helpers from skill_01
# --------------------------------------------------------------------------- #

try:
    from skill_01 import parse_csv, is_missing, to_float, compute_stats, normalize
except Exception:
    # Minimal fall‑back implementations
    def parse_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            header = reader.fieldnames if reader.fieldnames else []
        return rows, header

    def is_missing(val: str) -> bool:
        return val == '?' or val is None

    def to_float(val: str) -> float:
        try:
            return float(val)
        except ValueError:
            return math.nan

    def compute_stats(values: List[float]) -> Dict[str, float]:
        n = len(values)
        if n == 0:
            return {"mean": math.nan, "median": math.nan}
        mean_val = sum(values) / n
        sorted_vals = sorted(values)
        mid = n // 2
        if n % 2 == 1:
            med_val = sorted_vals[mid]
        else:
            med_val = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
        return {"mean": mean_val, "median": med_val}

    def normalize(val: float, min_val: float, max_val: float, maximize: bool) -> float:
        if max_val == min_val:
            return 1.0
        if maximize:
            return (val - min_val) / (max_val - min_val)
        return (max_val - val) / (max_val - min_val)

# --------------------------------------------------------------------------- #
# Helper functions for online processing
# --------------------------------------------------------------------------- #

def compute_min_max(rows: List[Dict[str, str]],
                    goal_columns: Dict[str, str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return per column min and max over rows that have a value."""
    col_min = {}
    col_max = {}
    for col in goal_columns:
        vals = [to_float(r[col]) for r in rows if not is_missing(r[col])]
        if vals:
            col_min[col] = min(vals)
            col_max[col] = max(vals)
        else:
            col_min[col] = col_max[col] = 0.0
    return col_min, col_max

def compute_row_score(row: Dict[str, str],
                      col_min: Dict[str, float],
                      col_max: Dict[str, float],
                      goal_columns: Dict[str, str]) -> float:
    """Return composite score for a row; None if any goal column missing."""
    score = 0.0
    for col, dirn in goal_columns.items():
        if is_missing(row[col]):
            return None
        v = to_float(row[col])
        best = normalize(v, col_min[col], col_max[col], maximize=(dirn == '+'))
        score += best
    return score

def column_summary(rows: List[Dict[str, str]],
                   goal_columns: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """Return n, mean, min, max for each goal column over the given rows."""
    summary = {}
    for col in goal_columns:
        vals = [to_float(r[col]) for r in rows if not is_missing(r[col])]
        n = len(vals)
        if n:
            mean = sum(vals) / n
            mn = min(vals)
            mx = max(vals)
        else:
            mean = mn = mx = None
        summary[col] = {"n": n, "mean": mean, "min": mn, "max": mx}
    return summary

# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    rows, header = parse_csv(csv_path)

    # Identify goal columns and direction
    goal_columns = {col: col[-1] for col in header if col.endswith('-') or col.endswith('+')}

    n_rows = len(rows)
    arrivals_count = round(0.2 * n_rows)
    if arrivals_count > n_rows:
        arrivals_count = n_rows

    history_rows = rows[:n_rows - arrivals_count]
    arrival_rows = rows[n_rows - arrivals_count:]

    # ----------------------------------------------------------- #
    # CAR-07: Online assessment of each arrival
    # ----------------------------------------------------------- #
    assessments = []
    # Keep a running list of rows seen so far
    seen_rows = history_rows.copy()
    for i, arrival in enumerate(arrival_rows, start=1):
        # Add the arrival to seen set for scoring
        seen_rows.append(arrival)

        # Recompute min/max and scores over current seen set
        col_min, col_max = compute_min_max(seen_rows, goal_columns)
        scores = [compute_row_score(r, col_min, col_max, goal_columns) for r in seen_rows]
        # Filter out rows that cannot be scored
        valid_scores = [s for s in scores if s is not None]
        # Determine threshold for top 30
        if len(valid_scores) < 30:
            threshold = -float('inf')
        else:
            sorted_scores = sorted(valid_scores, reverse=True)
            threshold = sorted_scores[29]  # 30th highest

        # Score for the current arrival
        arrival_score = compute_row_score(arrival, col_min, col_max, goal_columns)
        desirable = bool(arrival_score is not None and arrival_score >= threshold)
        assessments.append({"row": n_rows - arrivals_count + i, "desirable": desirable})

    # Human readable section
    print("== CAR-07 ==")
    print(f"Total arrivals processed: {len(assessment)}")
    desirable_count = sum(1 for a in assessments if a["desirable"])
    print(f"Desirable arrivals: {desirable_count}")
    print()

    # Machine‑checked answer
    print(f"ANSWER CAR-07 {json.dumps({'assessments': assessments}, separators=(',', ':'))}")

    # ----------------------------------------------------------- #
    # CAR-08: Incremental inventory summary checkpoints
    # ----------------------------------------------------------- #
    checkpoints = []
    # Checkpoints after 0, 40, 80 arrivals (but clip to actual arrivals)
    for m in [0, 40, 80]:
        if m > arrivals_count:
            break
        rows_seen = history_rows + arrival_rows[:m]
        # Summary per goal column
        summary = column_summary(rows_seen, goal_columns)
        # Shortlist of top 5 rows by composite score
        col_min, col_max = compute_min_max(rows_seen, goal_columns)
        # Compute scores for rows that can be scored
        scored_rows = []
        for idx, r in enumerate(rows_seen, start=1):
            s = compute_row_score(r, col_min, col_max, goal_columns)
            if s is not None:
                scored_rows.append((idx, s))
        # Sort by score descending
        scored_rows.sort(key=lambda kv: kv[1], reverse=True)
        shortlist = [idx for idx, _ in scored_rows[:5]]
        checkpoints.append({
            "after": m,
            "summary": summary,
            "shortlist": shortlist
        })

    # Human readable section
    print("== CAR-08 ==")
    print(f"Checkpoints processed: {len(checkpoints)}")
    for cp in checkpoints:
        print(f"  After {cp['after']} arrivals: shortlist={cp['shortlist']}")
    print()

    # Machine‑checked answer
    print(f"ANSWER CAR-08 {json.dumps({'checkpoints': checkpoints}, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
