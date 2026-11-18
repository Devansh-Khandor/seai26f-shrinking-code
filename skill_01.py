#!/usr/bin/env python3
"""
skill_01.py: Summary and top projects for NASA data.
Usage: python skill_01.py <path-to-csv>
"""

import json
import sys
from typing import List

import ezr


def read_table(path: str) -> ezr.TBL:
    """Read CSV into an ezr table."""
    rows = list(ezr.csv(path))
    # rows are already lists of coerced values
    return ezr.Tbl(rows)


def stats_for_goal(tbl: ezr.TBL, col: ezr.COL) -> dict:
    """Return mean, median, min, max, best, worst for a goal column."""
    values = [row[col.at] for row in tbl.rows if row[col.at] != "?"]
    n = len(values)
    if n == 0:
        # No data; return NaNs
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "best": None,
            "worst": None,
        }
    mean = sum(values) / n
    sorted_vals = sorted(values)
    median = (
        sorted_vals[n // 2]
        if n % 2 == 1
        else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    )
    mn = min(values)
    mx = max(values)
    best = mx if col.goal > 0 else mn
    worst = mn if col.goal > 0 else mx
    return {
        "mean": mean,
        "median": median,
        "min": mn,
        "max": mx,
        "best": best,
        "worst": worst,
    }


def compute_nasa_01(tbl: ezr.TBL) -> dict:
    """Compute summary statistics for the three goal columns."""
    result = {}
    for name, col in tbl.cols.items():
        if name.endswith(("+", "-")):
            result[name] = stats_for_goal(tbl, col)
    return result


def compute_nasa_02(tbl: ezr.TBL) -> dict:
    """Identify top 10 projects by overall combination of goals."""
    goal_cols = [col for name, col in tbl.cols.items() if name.endswith(("+", "-"))]
    # Compute ydist for each row that has all goal values
    rows_with_goals = [
        (i + 1, r) for i, r in enumerate(tbl.rows) if all(r[col.at] != "?" for col in goal_cols)
    ]
    # If no rows, return empty
    if not rows_with_goals:
        return {"top": []}
    # ydist uses the table's stats, so ensure tbl has correct stats
    rows_with_goals.sort(key=lambda item: ezr.ydist(tbl, item[1]))
    top10 = [num for num, _ in rows_with_goals[:10]]
    return {"top": top10}


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv-file>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    tbl = read_table(csv_path)

    # NASA-01
    print("== NASA-01 ==")
    nasa01 = compute_nasa_01(tbl)
    # Simple human‑readable output (at least three lines)
    for col_name, stats in nasa01.items():
        print(f"{col_name}: mean={stats['mean']:.3g}, median={stats['median']:.3g}, "
              f"min={stats['min']:.3g}, max={stats['max']:.3g}, "
              f"best={stats['best']:.3g}, worst={stats['worst']:.3g}")
    print("ANSWER NASA-01", json.dumps(nasa01))

    # NASA-02
    print("\n== NASA-02 ==")
    nasa02 = compute_nasa_02(tbl)
    # Human‑readable list of top rows
    print("Top rows:", ", ".join(map(str, nasa02["top"])))
    print("ANSWER NASA-02", json.dumps(nasa02))


if __name__ == "__main__":
    main()
