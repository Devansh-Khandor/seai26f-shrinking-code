#!/usr/bin/env python3
"""
skill_01.py: Summary and top-10 vehicle ranking for auto93 dataset.
Usage: python skill_01.py <csv_path>
"""
import sys
import json
import ezr
from typing import List, Dict, Any

def read_csv(path: str) -> List[List[Any]]:
    """Read CSV and return list of rows (atoms)."""
    with open(path, encoding="utf-8-sig") as f:
        header = next(f).strip().split(",")
        rows = []
        for line in f:
            if line.strip():
                rows.append([ezr.atom(s) for s in line.split(",")])
    return header, rows

def build_table(header: List[str], rows: List[List[Any]]) -> ezr.TBL:
    """Create an ezr table from header and rows."""
    return ezr.Tbl([header] + rows)

def compute_stats(tbl: ezr.TBL) -> Dict[str, Any]:
    """Compute mean, median, min, max, best, worst for each goal column."""
    result = {}
    for at, col in tbl.cols.items():
        if col.goal == 0:  # not a goal
            continue
        values = [row[at] for row in tbl.rows if row[at] != "?"]
        n = len(values)
        if n == 0:
            continue
        sorted_vals = sorted(values)
        mean = sum(values) / n
        median = (
            sorted_vals[n // 2]
            if n % 2 == 1
            else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        )
        mn = min(values)
        mx = max(values)
        best = mn if col.goal == -1 else mx
        worst = mx if col.goal == -1 else mn
        result[col.txt] = {
            "mean": mean,
            "median": median,
            "min": mn,
            "max": mx,
            "best": best,
            "worst": worst,
        }
    return result

def normalize(val: Any, mn: float, mx: float, goal: int) -> float:
    """Normalize a value to [0,1] given goal direction."""
    if val == "?":
        return 0.0
    if mx == mn:
        return 1.0
    if goal == 1:  # maximize
        return (val - mn) / (mx - mn)
    else:          # minimize
        return (mx - val) / (mx - mn)

def compute_top10(tbl: ezr.TBL) -> List[int]:
    """Compute top 10 rows based on combined goal score."""
    goal_cols = [(col.at, col) for col in tbl.cols.values() if col.goal != 0]
    if not goal_cols:
        return []

    # precompute min/max for each goal column
    stats = {}
    for at, col in goal_cols:
        vals = [row[at] for row in tbl.rows if row[at] != "?"]
        if vals:
            stats[at] = {"min": min(vals), "max": max(vals), "goal": col.goal}
        else:
            stats[at] = {"min": 0, "max": 0, "goal": col.goal}

    scores = []
    for idx, row in enumerate(tbl.rows):
        # skip rows with missing goal values
        if any(row[at] == "?" for at, _ in goal_cols):
            continue
        score = 0.0
        for at, col in goal_cols:
            mn = stats[at]["min"]
            mx = stats[at]["max"]
            val = row[at]
            score += normalize(val, mn, mx, col.goal)
        scores.append((score, idx + 1))  # 1-based row number

    # sort by score descending, tie by row number ascending
    scores.sort(key=lambda x: (-x[0], x[1]))
    top10 = [row_no for _, row_no in scores[:10]]
    return top10

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    header, rows = read_csv(path)
    tbl = build_table(header, rows)

    # CAR-01
    stats = compute_stats(tbl)
    print("== CAR-01 ==")
    print(f"Statistics for goals: {stats}")
    print(f"ANSWER CAR-01 {json.dumps(stats, separators=(',', ':'))}")

    # CAR-02
    top_rows = compute_top10(tbl)
    answer = {"top": top_rows}
    print("\n== CAR-02 ==")
    print(f"Top 10 rows by combined score: {top_rows}")
    print(f"ANSWER CAR-02 {json.dumps(answer, separators=(',', ':'))}")

if __name__ == "__main__":
    main()
