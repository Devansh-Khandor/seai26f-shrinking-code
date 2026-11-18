#!/usr/bin/env python3
import csv
import json
import sys
import math
from typing import List, Dict, Any, Tuple

# ------------------------------------------------------------------
# Data utilities
# ------------------------------------------------------------------
def read_data(path: str) -> List[Dict[str, Any]]:
    """
    Read CSV into list of dicts.
    '?' or '' -> None.
    numeric strings -> float.
    others keep as str.
    """
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if v == '?' or v == '':
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows


# ------------------------------------------------------------------
# Goal columns
# ------------------------------------------------------------------
def goal_columns(rows: List[Dict[str, Any]]) -> List[str]:
    if not rows:
        return []
    return [k for k in rows[0] if k.endswith('-') or k.endswith('+')]


# ------------------------------------------------------------------
# Statistics helpers
# ------------------------------------------------------------------
def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: List[float]) -> float:
    if not values:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


# ------------------------------------------------------------------
# Performance scoring
# ------------------------------------------------------------------
def compute_mins_maxs(rows_subset: List[Dict[str, Any]], goals: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    mins = {}
    maxs = {}
    for col in goals:
        vals = [r[col] for r in rows_subset if r[col] is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
        else:
            mins[col] = maxs[col] = 0.0
    return mins, maxs


def performance_score(row: Dict[str, Any], goals: List[str], mins: Dict[str, float], maxs: Dict[str, float]) -> float:
    score = 0.0
    for col in goals:
        v = row[col]
        if v is None:
            return None
        mn = mins[col]
        mx = maxs[col]
        if mx == mn:
            norm = 0.0
        else:
            if col.endswith('-'):
                norm = (mx - v) / (mx - mn)
            else:
                norm = (v - mn) / (mx - mn)
        score += norm
    return score


# ------------------------------------------------------------------
# CAR-07: assessments for arrivals
# ------------------------------------------------------------------
def car_07(rows: List[Dict[str, Any]], goals: List[str], arrivals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    assessments = []
    seen_rows = []  # list of (row_num, score)

    # initialise with history (all rows except arrivals)
    history = rows[:len(rows)-len(arrivals)]

    for idx, r in enumerate(history, start=1):
        # compute score if possible
        mins, maxs = compute_mins_maxs(history, goals)
        score = performance_score(r, goals, mins, maxs)
        if score is not None:
            seen_rows.append((idx, score))

    for i, r in enumerate(arrivals, start=1):
        row_num = len(rows)-len(arrivals)+i  # 1-based index
        # compute current mins/maxs from seen rows (history + previous arrivals)
        current_subset = rows[:row_num-1]  # all rows before this arrival
        mins, maxs = compute_mins_maxs(current_subset, goals)
        arrival_score = performance_score(r, goals, mins, maxs)

        # build current top-5 list
        current_scores = sorted(seen_rows, key=lambda x: (-x[1], x[0]))
        top5 = current_scores[:5]

        desirable = False
        if arrival_score is not None:
            if len(top5) < 5:
                desirable = True
            else:
                # compare to the 5th best score
                if arrival_score > top5[-1][1]:
                    desirable = True
                elif arrival_score == top5[-1][1] and row_num < top5[-1][0]:
                    desirable = True
        assessments.append({"row": row_num, "desirable": desirable})

        # add this arrival to seen rows if it has a score
        if arrival_score is not None:
            seen_rows.append((row_num, arrival_score))

    return assessments


# ------------------------------------------------------------------
# CAR-08: checkpoints summaries & shortlist
# ------------------------------------------------------------------
def car_08(rows: List[Dict[str, Any]], goals: List[str], arrivals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    checkpoints = []

    arrival_counts = [0, 40, 80]
    total_arrivals = len(arrivals)
    history_len = len(rows) - total_arrivals

    for m in arrival_counts:
        if m > total_arrivals:
            m = total_arrivals
        # subset up to history + first m arrivals
        subset = rows[:history_len + m]
        # summary per goal
        summary = {}
        for col in goals:
            vals = [r[col] for r in subset if r[col] is not None]
            n = len(vals)
            if n > 0:
                summary[col] = {
                    "n": n,
                    "mean": round(mean(vals), 2),
                    "min": round(min(vals), 2),
                    "max": round(max(vals), 2),
                }
            else:
                summary[col] = {"n": 0, "mean": None, "min": None, "max": None}

        # shortlist: best 5 rows in subset
        # compute mins/maxs over subset for scoring
        mins, maxs = compute_mins_maxs(subset, goals)
        scored_rows = []
        for idx, r in enumerate(subset, start=1):
            sc = performance_score(r, goals, mins, maxs)
            if sc is not None:
                scored_rows.append((idx, sc))
        scored_rows.sort(key=lambda x: (-x[1], x[0]))
        shortlist = [idx for idx, _ in scored_rows[:5]]

        checkpoints.append({
            "after": m,
            "summary": summary,
            "shortlist": shortlist
        })

    return checkpoints


# ------------------------------------------------------------------
# Main routine
# ------------------------------------------------------------------
def main(csv_path: str):
    rows = read_data(csv_path)
    goals = goal_columns(rows)

    # arrivals: last round(0.2 * N) rows
    total_rows = len(rows)
    arrival_cnt = round(0.2 * total_rows)
    arrivals = rows[-arrival_cnt:] if arrival_cnt else []

    # CAR-07
    print("== CAR-07 ==")
    assessments = car_07(rows, goals, arrivals)
    for a in assessments:
        print(f"Row {a['row']} desirable: {a['desirable']}")
    answer07 = {"assessments": assessments}
    print(f"ANSWER CAR-07 {json.dumps(answer07, ensure_ascii=False)}")

    # CAR-08
    print("\n== CAR-08 ==")
    checkpoints = car_08(rows, goals, arrivals)
    for cp in checkpoints:
        print(f"After {cp['after']} arrivals")
        for col, stats in cp['summary'].items():
            print(f"  {col}: n={stats['n']} mean={stats['mean']} min={stats['min']} max={stats['max']}")
        print(f"  shortlist: {cp['shortlist']}")
    answer08 = {"checkpoints": checkpoints}
    print(f"ANSWER CAR-08 {json.dumps(answer08, ensure_ascii=False)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
