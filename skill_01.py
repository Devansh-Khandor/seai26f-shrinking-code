#!/usr/bin/env python3
"""
skill_01.py - Apache server configuration analysis

Usage:
    python skill_01.py <csv_path>

Outputs:
    == APACHE-01 ==
        Performance-: mean=..., median=..., min=..., max=..., best=..., worst=...
        ... (other summary lines)
    ANSWER APACHE-01 <JSON>

    == APACHE-02 ==
        Top rows: [ ... ]
        Setting shares:
            <setting_name>: top_share=..., rest_share=...
        ...
    ANSWER APACHE-02 <JSON>
"""

import csv
import json
import sys
from statistics import median
from typing import List, Dict, Any


def read_csv(path: str) -> List[Dict[str, Any]]:
    """
    Read CSV and return list of rows (dict header->value).
    Values are parsed to int/float if possible; otherwise kept as string.
    Missing cells are represented by the string '?'.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, Any] = {}
            for k, v in row.items():
                v = v.strip()
                if v == '':
                    v = '?'  # treat empty as missing
                try:
                    parsed[k] = int(v)
                except ValueError:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows


def is_goal(col: str) -> bool:
    return col.endswith('+') or col.endswith('-')


def direction_is_min(col: str) -> bool:
    """True if goal is to minimize (ends with '-')."""
    return col.endswith('-')


def numeric_value(v: Any) -> Any:
    """Return numeric value or None if missing."""
    if v == '?' or v is None:
        return None
    return v


def compute_stats(rows: List[Dict[str, Any]], goal_cols: List[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for col in goal_cols:
        values = [numeric_value(row[col]) for row in rows if numeric_value(row[col]) is not None]
        if not values:
            continue
        values.sort()
        m = sum(values) / len(values)
        med = median(values)
        mn = values[0]
        mx = values[-1]
        best = mn if direction_is_min(col) else mx
        worst = mx if direction_is_min(col) else mn
        result[col] = {
            'mean': round(m, 4),
            'median': round(med, 4),
            'min': round(mn, 4),
            'max': round(mx, 4),
            'best': round(best, 4),
            'worst': round(worst, 4)
        }
    return result


def top_n_rows(rows: List[Dict[str, Any]], goal_col: str) -> List[int]:
    """Return row numbers (1-based) of top favorable configurations."""
    # Build list of (value, index)
    candidates = [(numeric_value(r[goal_col]), i)
                  for i, r in enumerate(rows, start=1)
                  if numeric_value(r[goal_col]) is not None]
    # Sort ascending (better for minimization)
    candidates.sort(key=lambda x: x[0])
    N = len(rows)
    top_n = max(5, min(20, int(round(0.1 * N))))
    top_n = min(top_n, N)
    top_rows = [i for _, i in candidates[:top_n]]
    return top_rows


def compute_shares(rows: List[Dict[str, Any]],
                   top_rows: List[int],
                   goal_cols: List[str]) -> List[Dict[str, Any]]:
    """Compute share of 1s for each non-goal column in top and rest rows."""
    # Map row number to row data
    row_map = {i + 1: r for i, r in enumerate(rows)}
    top_set = set(top_rows)
    rest_rows = [r for idx, r in enumerate(rows, start=1) if idx not in top_set]

    # Determine setting columns (non-goal)
    settings = [c for c in rows[0].keys() if not is_goal(c) and c != 'X']
    shares_list: List[Dict[str, Any]] = []

    def is_one(v):
        return v == 1 or v is True or (isinstance(v, str) and v.lower() == '1')

    top_n = len(top_rows)
    rest_n = len(rest_rows)

    for setting in settings:
        top_ones = sum(1 for i in top_rows if is_one(row_map[i].get(setting)))
        rest_ones = sum(1 for r in rest_rows if is_one(r.get(setting)))
        top_share = round(top_ones / top_n, 4) if top_n else 0.0
        rest_share = round(rest_ones / rest_n, 4) if rest_n else 0.0
        shares_list.append({
            'setting': setting,
            'top_share': top_share,
            'rest_share': rest_share
        })

    # Sort by distinguishing power (difference)
    shares_list.sort(key=lambda d: abs(d['top_share'] - d['rest_share']), reverse=True)
    # Ensure at least 2 entries
    return shares_list[:max(2, len(shares_list))]


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    rows = read_csv(csv_path)
    if not rows:
        print("No data.")
        sys.exit(0)

    # Identify goal columns
    goal_cols = [c for c in rows[0].keys() if is_goal(c)]
    if not goal_cols:
        print("No goal columns found.")
        sys.exit(0)

    # ----------------------------------------------------
    # APACHE-01
    # ----------------------------------------------------
    stats = compute_stats(rows, goal_cols)
    print("== APACHE-01 ==")
    # Print at least three lines of results
    for col in goal_cols:
        s = stats[col]
        print(f"{col}: mean={s['mean']}, median={s['median']}, "
              f"min={s['min']}, max={s['max']}, best={s['best']}, worst={s['worst']}")
    answer01 = {"Performance-": stats.get("Performance-") or {}}
    print("ANSWER APACHE-01", json.dumps(answer01, separators=(',', ':')))

    # ----------------------------------------------------
    # APACHE-02
    # ----------------------------------------------------
    goal_col = goal_cols[0]  # assume single goal column as in dataset
    top_rows = top_n_rows(rows, goal_col)
    shares = compute_shares(rows, top_rows, goal_cols)

    print("\n== APACHE-02 ==")
    print("Top rows:", top_rows)
    print("Setting shares:")
    for d in shares:
        print(f"  {d['setting']}: top_share={d['top_share']}, rest_share={d['rest_share']}")
    answer02 = {"top": top_rows, "settings": shares}
    print("ANSWER APACHE-02", json.dumps(answer02, separators=(',', ':')))


if __name__ == "__main__":
    main()
