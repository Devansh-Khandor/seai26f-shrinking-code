#!/usr/bin/env python3
"""
skill_04.py <csv_path>
Implements CAR-07 (arrival assessment) and CAR-08 (incremental summary & shortlist).
"""

import csv
import json
import sys
import math
from typing import List, Dict, Tuple, Optional

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def parse_value(v: str) -> Optional[float]:
    if v == '?' or v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None

def read_csv(csv_path: str) -> Tuple[List[Tuple[int, Dict[str, Optional[float]]]], List[str]]:
    rows: List[Tuple[int, Dict[str, Optional[float]]]] = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for i, row in enumerate(reader, start=1):
            parsed: Dict[str, Optional[float]] = {}
            for k, v in row.items():
                parsed[k] = parse_value(v)
            rows.append((i, parsed))
    return rows, headers

def goal_columns(headers: List[str]) -> List[str]:
    return [h for h in headers if h.endswith('+') or h.endswith('-')]

def compute_min_max(rows: List[Tuple[int, Dict[str, Optional[float]]]], goals: List[str]) -> Dict[str, Tuple[float, float]]:
    minmax: Dict[str, Tuple[float, float]] = {}
    for col in goals:
        values = [row[1][col] for row in rows if row[1][col] is not None]
        if values:
            minmax[col] = (min(values), max(values))
    return minmax

def compute_composite(row: Dict[str, Optional[float]], minmax: Dict[str, Tuple[float, float]], goals: List[str]) -> Optional[float]:
    norm_scores = []
    for col in goals:
        v = row.get(col)
        if v is None or col not in minmax:
            return None
        mn, mx = minmax[col]
        if mn == mx:
            norm = 0.0
        else:
            if col.endswith('+'):
                norm = (v - mn) / (mx - mn)
            else:
                norm = (mx - v) / (mx - mn)
        norm_scores.append(norm)
    return sum(norm_scores) / len(norm_scores)

def compute_summary(rows: List[Tuple[int, Dict[str, Optional[float]]]], goals: List[str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for col in goals:
        vals = [row[1][col] for row in rows if row[1][col] is not None]
        n = len(vals)
        if n == 0:
            mean = min_val = max_val = 0.0
        else:
            mean = sum(vals) / n
            min_val = min(vals)
            max_val = max(vals)
        summary[col] = {"n": n, "mean": mean, "min": min_val, "max": max_val}
    return summary

def top_k(rows: List[Tuple[int, Dict[str, Optional[float]]]], k: int, minmax: Dict[str, Tuple[float, float]], goals: List[str]) -> List[int]:
    scored: List[Tuple[float, int]] = []
    for row_num, data in rows:
        score = compute_composite(data, minmax, goals)
        if score is not None:
            scored.append((score, row_num))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [row for _, row in scored[:k]]

# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    rows_all, headers = read_csv(csv_path)
    N = len(rows_all)
    if N == 0:
        print("No data.")
        return

    goals = goal_columns(headers)

    # Determine arrivals
    arrivals_count = round(0.2 * N)
    if arrivals_count > N:
        arrivals_count = N
    history_rows = rows_all[:N - arrivals_count]
    arrivals_rows = rows_all[N - arrivals_count:]

    # --------------------------- CAR-07 --------------------------- #
    print("== CAR-07 ==")
    assessments = []
    seen_rows = list(history_rows)  # copy

    for arrival in arrivals_rows:
        seen_rows.append(arrival)
        # compute min/max over seen_rows
        minmax = compute_min_max(seen_rows, goals)
        # get top 5 rows
        top5 = top_k(seen_rows, 5, minmax, goals)
        # assess
        desirable = arrival[0] in top5
        assessments.append({"row": arrival[0], "desirable": desirable})
        # print a simple line
        print(f"Row {arrival[0]}: {'desirable' if desirable else 'not desirable'}")

    # Answer line
    print(f"ANSWER CAR-07 {json.dumps({'assessments': assessments})}")

    # --------------------------- CAR-08 --------------------------- #
    print("\n== CAR-08 ==")
    checkpoints = []
    for m in [0, 40, 80]:
        m = min(m, arrivals_count)  # can't exceed actual arrivals
        seen_rows_checkpoint = list(history_rows) + arrivals_rows[:m]
        minmax = compute_min_max(seen_rows_checkpoint, goals)
        summary = compute_summary(seen_rows_checkpoint, goals)
        shortlist = top_k(seen_rows_checkpoint, 5, minmax, goals)
        checkpoints.append({"after": m, "summary": summary, "shortlist": shortlist})
        # print human readable
        print(f"\nAfter {m} arrivals:")
        print("Summary:")
        for col, stats in summary.items():
            print(f"  {col}: n={stats['n']}, mean={stats['mean']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")
        print("Shortlist (row numbers):", shortlist)

    # Answer line
    print(f"ANSWER CAR-08 {json.dumps({'checkpoints': checkpoints})}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_04.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])
