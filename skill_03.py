#!/usr/bin/env python3
"""
skill_03.py: Cylinder and model-year analysis for auto93 dataset.
Usage: python skill_03.py <csv_path>
"""

import json
import sys
import ezr
from typing import List, Dict, Any, Tuple

# ------------------------------------------------------------------
# Utility functions (copy/paste from skill_01 / skill_02)

def read_csv(path: str) -> Tuple[List[str], List[List[Any]]]:
    """Read CSV and return header and rows (atoms)."""
    with open(path, encoding="utf-8-sig") as f:
        header = next(f).strip().split(",")
        rows: List[List[Any]] = []
        for line in f:
            if line.strip():
                rows.append([ezr.atom(s) for s in line.split(",")])
    return header, rows


def build_table(header: List[str], rows: List[List[Any]]) -> ezr.TBL:
    """Create an ezr table from header and rows."""
    return ezr.Tbl([header] + rows)


def median(lst: List[float]) -> float:
    """Return median of a list of numbers."""
    n = len(lst)
    if n == 0:
        return 0.0
    lst_sorted = sorted(lst)
    mid = n // 2
    if n % 2 == 1:
        return lst_sorted[mid]
    return (lst_sorted[mid - 1] + lst_sorted[mid]) / 2.0


def mean(lst: List[float]) -> float:
    """Return mean of a list of numbers."""
    if not lst:
        return 0.0
    return sum(lst) / len(lst)


# ------------------------------------------------------------------
# CAR-05: Cylinder groups

def car_05(tbl: ezr.TBL) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    """Return groups stats and comparisons for CAR-05."""
    # Find index of goal columns
    goal_cols = [col for col in tbl.cols.values() if col.goal != 0]
    mpg_col = next((col for col in goal_cols if col.txt == "Mpg+"), None)
    if not mpg_col:
        raise ValueError("No Mpg+ goal column found")

    # Gather groups by Clndrs
    groups: Dict[Any, List[float]] = {}
    for row in tbl.rows:
        cl = row[tbl.cols["Clndrs"].at]
        val = row[mpg_col.at]
        if val == "?":
            continue
        groups.setdefault(cl, []).append(val)

    # Prepare output dictionary
    groups_out: Dict[str, Dict[str, float]] = {}
    for cl, vals in groups.items():
        groups_out[str(cl)] = {
            "n": len(vals),
            "median": round(median(vals), 10),
            "mean": round(mean(vals), 10),
        }

    # Prepare comparisons
    comparisons: List[Dict[str, Any]] = []
    cl_values = sorted(groups.keys())
    for i in range(len(cl_values)):
        for j in range(i + 1, len(cl_values)):
            a = cl_values[i]
            b = cl_values[j]
            verdict = "different" if not ezr.same(groups[a], groups[b], eps=0.35) else "same"
            comparisons.append({"a": a, "b": b, "verdict": verdict})

    return groups_out, comparisons


def print_car_05(groups: Dict[str, Dict[str, float]], comparisons: List[Dict[str, Any]]):
    print("== CAR-05 ==")
    print("Groups:")
    for cl, stats in groups.items():
        print(f"  Clndrs={cl} n={stats['n']} median={stats['median']} mean={stats['mean']}")
    print("\nComparisons:")
    for cmp in comparisons:
        print(f"  {cmp['a']} vs {cmp['b']}: {cmp['verdict']}")
    ans = {"groups": groups, "comparisons": comparisons}
    print(f"ANSWER CAR-05 {json.dumps(ans)}")


# ------------------------------------------------------------------
# CAR-06: History vs Arrivals

def car_06(tbl: ezr.TBL) -> Tuple[List[Dict[str, Any]], str]:
    """Return groups info and verdict for CAR-06."""
    total_rows = len(tbl.rows)
    arrivals_count = round(0.2 * total_rows)
    if arrivals_count == 0:
        arrivals_count = 1  # ensure at least one arrival
    arrivals_start = total_rows - arrivals_count

    # Determine group ranges by Model value
    model_col = tbl.cols["Model"]
    history_rows = tbl.rows[:arrivals_start]
    arrival_rows = tbl.rows[arrivals_start:]

    def get_range(rows: List[List[Any]]) -> Tuple[int, int]:
        vals = [row[model_col.at] for row in rows if isinstance(row[model_col.at], int) or isinstance(row[model_col.at], float)]
        return int(min(vals)), int(max(vals))

    hist_from, hist_to = get_range(history_rows)
    arr_from, arr_to = get_range(arrival_rows)

    groups = [
        {"from": hist_from, "to": hist_to, "n": len(history_rows)},
        {"from": arr_from, "to": arr_to, "n": len(arrival_rows)},
    ]

    # Compute ydist values for each group (skip rows with missing goal values)
    goal_cols = [col for col in tbl.cols.values() if col.goal != 0]
    def collect_ydist(rows):
        y = []
        for r in rows:
            if any(r[col.at] == "?" for col in goal_cols):
                continue
            y.append(ezr.ydist(tbl, r))
        return y

    y_hist = collect_ydist(history_rows)
    y_arr = collect_ydist(arrival_rows)

    verdict = "different" if not ezr.same(y_hist, y_arr, eps=0.35) else "same"

    return groups, verdict


def print_car_06(groups: List[Dict[str, Any]], verdict: str):
    print("\n== CAR-06 ==")
    for g in groups:
        print(f"  Model {g['from']} to {g['to']}, n={g['n']}")
    print(f"\nVerdict: {verdict}")
    ans = {"groups": groups, "verdict": verdict}
    print(f"ANSWER CAR-06 {json.dumps(ans)}")


# ------------------------------------------------------------------
# Main driver

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    header, rows = read_csv(path)
    tbl = build_table(header, rows)

    groups, comparisons = car_05(tbl)
    print_car_05(groups, comparisons)

    groups06, verdict06 = car_06(tbl)
    print_car_06(groups06, verdict06)


if __name__ == "__main__":
    main()
