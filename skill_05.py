#!/usr/bin/env python3
"""
Skill 05: Apache configuration anomaly and combination analysis

This program implements two analyses for the Apache configuration data:

* **APACHE-09** – When a configuration performs poorly,
  identify which of its enabled or disabled settings are most
  strongly associated with unfavorable performance in our historical
  measurements.

* **APACHE-10** – Identify combinations of server settings that
  frequently appear among poorly performing configurations and
  show how they differ from better‑performing combinations.

The program prints a header line for each analysis, a few human‑readable
lines, and ends each section with a machine‑checked answer line
containing a single‑line JSON object.
"""

import csv
import json
import sys
import math
from typing import List, Dict, Any, Tuple, Set


# ----------------------------------------------------------------------
# CSV utilities
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Read the CSV and return header and list of rows as dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows: List[Dict[str, Any]] = []
        for row in rdr:
            d: Dict[str, Any] = {}
            for h, val in zip(header, row):
                if val == "?":
                    d[h] = None
                else:
                    try:
                        if "." in val or "e" in val.lower():
                            d[h] = float(val)
                        else:
                            d[h] = int(val)
                    except ValueError:
                        d[h] = val
            rows.append(d)
    return header, rows


def is_goal(col: str) -> bool:
    return col.endswith("+") or col.endswith("-")


def is_input(col: str) -> bool:
    return not col.endswith("X") and not is_goal(col)


def mean_std(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    m = sum(values) / n
    var = sum((x - m) ** 2 for x in values) / max(1, n - 1)
    return (m, math.sqrt(var))


# ----------------------------------------------------------------------
# Analysis helpers
# ----------------------------------------------------------------------
def find_poor_rows(rows: List[Dict[str, Any]], goal: str, min_count: int = 5) -> List[int]:
    """Return row numbers (1‑based) whose goal value is unusually high."""
    values = [row[goal] for row in rows if row[goal] is not None]
    if not values:
        return []
    m, sd = mean_std(values)
    threshold = m + 1.5 * sd
    poor = [i + 1 for i, r in enumerate(rows) if r[goal] is not None and r[goal] > threshold]
    # If not enough, take the worst rows
    if len(poor) < min_count:
        sorted_rows = sorted(
            [(r[goal], i + 1) for i, r in enumerate(rows) if r[goal] is not None],
            reverse=True,
        )
        poor = [idx for _, idx in sorted_rows[:min_count]]
    return poor


def find_good_rows(rows: List[Dict[str, Any]], goal: str, min_count: int = 5) -> List[int]:
    """Return row numbers (1‑based) of best performing rows (lowest goal)."""
    values = [(r[goal], i + 1) for i, r in enumerate(rows) if r[goal] is not None]
    if not values:
        return []
    sorted_rows = sorted(values, key=lambda x: x[0])
    good = [idx for _, idx in sorted_rows[:min_count]]
    return good


def compute_ranking(
    rows: List[Dict[str, Any]], goal: str, input_cols: List[str], top_n: int = 3
) -> List[str]:
    """Return column names most strongly associated with poor performance."""
    diffs: List[Tuple[float, str]] = []
    for col in input_cols:
        # Consider only rows with non‑missing values in both fields
        vals1 = [row[goal] for row in rows if row[col] == 1 and row[goal] is not None]
        vals0 = [row[goal] for row in rows if row[col] == 0 and row[goal] is not None]
        if not vals1 or not vals0:
            continue
        m1, _ = mean_std(vals1)
        m0, _ = mean_std(vals0)
        diff = abs(m1 - m0)
        diffs.append((diff, col))
    diffs.sort(key=lambda x: (-x[0], x[1]))
    return [col for _, col in diffs[:top_n]]


def compute_combinations(
    bad_rows: List[int],
    good_rows: List[int],
    rows: List[Dict[str, Any]],
    input_cols: List[str],
) -> List[Dict[str, Any]]:
    """Return list of setting combinations that are characteristic of bad rows."""
    bad_set: Set[int] = set(bad_rows)
    good_set: Set[int] = set(good_rows)
    combos: List[Dict[str, Any]] = []

    # Prepare list of rows for fast lookup
    bad_rows_data = [rows[i - 1] for i in bad_rows]
    good_rows_data = [rows[i - 1] for i in good_rows]

    # Examine all pairs of input columns
    n_cols = len(input_cols)
    for i in range(n_cols):
        col1 = input_cols[i]
        for j in range(i + 1, n_cols):
            col2 = input_cols[j]
            # Look for 0/1 combinations
            for v1 in (0, 1):
                for v2 in (0, 1):
                    # Count matches
                    bad_match = sum(
                        1
                        for r in bad_rows_data
                        if r.get(col1) == v1 and r.get(col2) == v2
                    )
                    good_match = sum(
                        1
                        for r in good_rows_data
                        if r.get(col1) == v1 and r.get(col2) == v2
                    )
                    if not bad_match:
                        continue
                    bad_share = bad_match / len(bad_rows)
                    good_share = good_match / len(good_rows) if good_rows else 0.0
                    if bad_share > good_share:
                        combo = {
                            "settings": {col1: v1, col2: v2},
                            "bad_share": round(bad_share, 3),
                            "good_share": round(good_share, 3),
                        }
                        combos.append(combo)

    # Sort combos: most typical of bad first
    combos.sort(
        key=lambda c: (
            -c["bad_share"],
            -(c["bad_share"] - c["good_share"]),
        )
    )
    return combos


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    header, rows = read_csv(csv_path)

    # Identify columns
    goal_cols = [c for c in header if is_goal(c)]
    if not goal_cols:
        print("No goal columns found.")
        sys.exit(1)
    goal_col = goal_cols[0]  # use first goal column (Performance-)
    input_cols = [c for c in header if is_input(c)]

    # --------------------------------------------------------------
    # APACHE-09
    # --------------------------------------------------------------
    print("== APACHE-09 ==")
    poor_rows = find_poor_rows(rows, goal_col, min_count=5)
    ranking = compute_ranking(rows, goal_col, input_cols, top_n=3)

    print(f"Poor rows (1‑based indices): {poor_rows}")
    print(f"Top associated settings: {ranking}")

    answer_09 = {"poor": poor_rows, "ranking": ranking}
    print(f"ANSWER APACHE-09 {json.dumps(answer_09)}")

    # --------------------------------------------------------------
    # APACHE-10
    # --------------------------------------------------------------
    print("\n== APACHE-10 ==")
    good_rows = find_good_rows(rows, goal_col, min_count=5)
    combos = compute_combinations(poor_rows, good_rows, rows, input_cols)

    print(f"Bad rows: {poor_rows}")
    print(f"Good rows: {good_rows}")
    print(f"Significant combinations (first 5 shown): {combos[:5]}")

    answer_10 = {"bad": poor_rows, "good": good_rows, "combinations": combos}
    print(f"ANSWER APACHE-10 {json.dumps(answer_10)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
