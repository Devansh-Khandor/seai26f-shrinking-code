#!/usr/bin/env python3
"""
Skill 01 – NASA project summary and top performers

This script reads a NASA project CSV file and prints
two analyses:
  1. Summary statistics for the goal columns
     (Kloc+, Effort-, Defects-).
  2. Top 10 projects that best balance size, effort
     and defect counts.

The output format follows the specification:
   * a line "== ID ==" for each request,
   * at least three lines of results,
   * a machine‑checked answer line beginning with
     "ANSWER ID " followed by a JSON object.

All calculations are performed on the data file supplied
on the command line, and no hard‑coded values are used.
"""

import json
import sys
import ezr

# ----------------------------------------------------------------------
# Helper functions for statistics

def _mean(values):
    return sum(values) / len(values)

def _median(values):
    n = len(values)
    s = sorted(values)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0

def _stats(values, best_worst):
    """
    Return dict with mean, median, min, max,
    best, worst.  `best_worst` is a tuple
    (best_value, worst_value) based on direction.
    """
    return {
        "mean": _mean(values),
        "median": _median(values),
        "min": min(values),
        "max": max(values),
        "best": best_worst[0],
        "worst": best_worst[1]
    }

# ----------------------------------------------------------------------
# Main analysis functions

def compute_goal_stats(tbl):
    """Return a mapping of goal column name to its stats."""
    # Goal columns are those ending in '+' or '-'
    goals = [col for col in tbl.cols.values() if col.txt.endswith(('+', '-'))]
    goal_stats = {}
    for col in goals:
        values = [row[col.at] for row in tbl.rows if row[col.at] != "?"]
        if not values:
            continue
        # Determine direction: maximize or minimize
        if col.txt.endswith('+'):
            best, worst = max(values), min(values)
        else:
            best, worst = min(values), max(values)
        goal_stats[col.txt] = _stats(values, (best, worst))
    return goal_stats

def compute_top_projects(tbl, goal_stats, top_n=10):
    """Return the row numbers of the top projects."""
    # Normalisation ranges
    ranges = {name: (stats["min"], stats["max"]) for name, stats in goal_stats.items()}
    # Evaluate each row that has complete data for all goals
    scored = []
    for i, row in enumerate(tbl.rows, start=1):  # 1-based row number
        try:
            # Skip rows with missing goal data
            if any(row[tbl.cols[name].at] == "?" for name in goal_stats):
                continue
            score = 0.0
            for name, (minv, maxv) in ranges.items():
                val = row[tbl.cols[name].at]
                if maxv == minv:
                    norm = 0.0
                else:
                    if name.endswith('+'):  # maximize
                        norm = (val - minv) / (maxv - minv)
                    else:  # minimize
                        norm = (maxv - val) / (maxv - minv)
                score += norm
            scored.append((score, i))
        except Exception:  # defensive: ignore rows that cause errors
            continue
    scored.sort(reverse=True, key=lambda x: x[0])
    return [idx for _, idx in scored[:top_n]]

# ----------------------------------------------------------------------
# Main entry point

def main(csv_path):
    # Load table
    tbl = ezr.Tbl(ezr.csv(csv_path))

    # 1. NASA-01
    goal_stats = compute_goal_stats(tbl)
    print("== NASA-01 ==")
    for name in ["Kloc+", "Effort-", "Defects-"]:
        stats = goal_stats.get(name)
        if stats is None:
            continue
        print(f"{name}: mean={stats['mean']:.2f}, median={stats['median']:.2f}, "
              f"min={stats['min']:.2f}, max={stats['max']:.2f}, "
              f"best={stats['best']:.2f}, worst={stats['worst']:.2f}")

    # Machine‑checked answer line
    ans_01 = {name: {
        "mean": round(stats["mean"], 2),
        "median": round(stats["median"], 2),
        "min": round(stats["min"], 2),
        "max": round(stats["max"], 2),
        "best": round(stats["best"], 2),
        "worst": round(stats["worst"], 2)
    } for name, stats in goal_stats.items()}
    print(f"ANSWER NASA-01 {json.dumps(ans_01, separators=(',', ':'))}")

    # 2. NASA-02
    top_rows = compute_top_projects(tbl, goal_stats, top_n=10)
    print("== NASA-02 ==")
    for r in top_rows:
        print(r)

    ans_02 = {"top": top_rows}
    print(f"ANSWER NASA-02 {json.dumps(ans_02, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python skill_01.py <csv_path>")
    main(sys.argv[1])
