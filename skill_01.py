#!/usr/bin/env python3
"""
Entry point for the client's requested skills.
"""

import sys
import json
import math
from typing import List, Tuple, Dict

# The ezr library is available in the same directory.
from ezr import csv  # we only need csv to parse the file

# ---------- Utility helpers ----------
def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")

def median(values: List[float]) -> float:
    n = len(values)
    if n == 0:
        return float("nan")
    s = sorted(values)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0

def best_worst(values: List[float], maximize: bool) -> Tuple[float, float]:
    """Return (best, worst) given direction."""
    if not values:
        return (float("nan"), float("nan"))
    if maximize:
        return (max(values), min(values))
    else:
        return (min(values), max(values))

# ---------- Skill CAR-01 ----------
def car_01_summary(rows: List[List[object]]) -> Dict[str, Dict[str, float]]:
    """
    Compute summary stats for the three goal columns:
    Lbs-, Acc+, Mpg+.
    """
    # Identify indices from header
    header = rows[0]
    idxs = {name: i for i, name in enumerate(header)}
    goal_cols = ["Lbs-", "Acc+", "Mpg+"]
    stats = {}
    for col in goal_cols:
        at = idxs[col]
        values = [float(r[at]) for r in rows[1:] if r[at] != "?"]
        stats[col] = {
            "mean": round(mean(values), 6),
            "median": round(median(values), 6),
            "min": min(values) if values else float("nan"),
            "max": max(values) if values else float("nan"),
            "best": round(best_worst(values, maximize=col[0] != "-")[0], 6),
            "worst": round(best_worst(values, maximize=col[0] != "-")[1], 6),
        }
    return stats

# ---------- Skill CAR-02 ----------
def car_02_top(rows: List[List[object]], top_n: int = 10) -> List[int]:
    """
    Find top rows according to a combined ranking that favors low weight,
    high mpg, and high acceleration.
    Rows with any missing goal value are skipped.
    Returns 1‑based row numbers (excluding header).
    """
    header = rows[0]
    idxs = {name: i for i, name in enumerate(header)}
    goal_cols = ["Lbs-", "Acc+", "Mpg+"]
    # Build list of (row_number, row, rank_sum)
    ranked = []
    for rn, row in enumerate(rows[1:], start=1):
        # skip rows with missing any of the three goals
        if any(row[idxs[g]] == "?" for g in goal_cols):
            continue
        # For each column compute rank: ascending for min, descending for max
        values = [float(row[idxs[g]]) for g in goal_cols]
        # ranks: we use simple ranking within this dataset
        # Create sorted lists to assign rank positions (1 = best)
        # For Lbs- (min), lower better
        # For Acc+, Mpg+ (max), higher better
        # We'll compute ranks by sorting values accordingly
        ranks = []
        # Lbs-
        sorted_lbs = sorted((v, i) for i, v in enumerate(values))
        pos_lbs = {i: rank+1 for rank, (_, i) in enumerate(sorted_lbs)}
        ranks.append(pos_lbs[0])
        # Acc+
        sorted_acc = sorted((v, i) for i, v in enumerate(values), reverse=True)
        pos_acc = {i: rank+1 for rank, (_, i) in enumerate(sorted_acc)}
        ranks.append(pos_acc[1])
        # Mpg+
        sorted_mpg = sorted((v, i) for i, v in enumerate(values), reverse=True)
        pos_mpg = {i: rank+1 for rank, (_, i) in enumerate(sorted_mpg)}
        ranks.append(pos_mpg[2])
        rank_sum = sum(ranks)
        ranked.append((rn, row, rank_sum))
    # Sort by rank_sum ascending (lower is better)
    ranked.sort(key=lambda x: x[2])
    top_rows = [rn for rn, _, _ in ranked[:top_n]]
    return top_rows

# ---------- Main ----------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    # Read CSV into list of rows
    rows = [row for row in csv(path)]
    if not rows:
        print("No data found.")
        sys.exit(1)

    # CAR-01
    print("== CAR-01 ==")
    summary = car_01_summary(rows)
    # Print at least three lines of results
    for col, vals in summary.items():
        print(f"{col}: {vals}")
    print(f"ANSWER CAR-01 {json.dumps(summary)}")

    # CAR-02
    print("\n== CAR-02 ==")
    top10 = car_02_top(rows, top_n=10)
    # Print at least three lines
    print("Top 10 rows:", top10)
    print(f"ANSWER CAR-02 {json.dumps({'top': top10})}")

if __name__ == "__main__":
    main()
