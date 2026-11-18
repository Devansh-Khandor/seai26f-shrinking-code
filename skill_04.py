#!/usr/bin/env python3
"""
skill_04.py - Apache performance monitoring and incremental analysis

Usage:
    python skill_04.py <csv_path>
"""

import csv
import json
import math
import sys
from collections import defaultdict
from typing import List, Dict, Any, Tuple


def parse_value(v: str) -> Any:
    """Convert a CSV field to int/float if possible; keep '?' as is."""
    v = v.strip()
    if v == "":
        return "?"
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def read_csv(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Read the CSV and return rows as list of dicts and the header."""
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for line in reader:
            if not line:
                continue
            row: Dict[str, Any] = {}
            for key, val in zip(headers, line):
                row[key] = parse_value(val)
            rows.append(row)
    return rows, headers


def is_goal(col: str) -> bool:
    """True if column is a goal (ends with '+' or '-')"""
    return col.endswith("+") or col.endswith("-")


def goal_direction(col: str) -> str:
    """Return 'min' if goal to minimize, 'max' otherwise."""
    return "min" if col.endswith("-") else "max"


def compute_stats(rows: List[Dict[str, Any]], goal_col: str) -> Dict[str, Any]:
    """Return statistics (n, mean, min, max) for a single goal column."""
    vals = []
    for r in rows:
        v = r.get(goal_col, "?")
        if v != "?":
            try:
                vals.append(float(v))
            except Exception:
                pass
    if not vals:
        return {"n": 0, "mean": None, "min": None, "max": None}
    mean_val = sum(vals) / len(vals)
    return {
        "n": len(vals),
        "mean": mean_val,
        "min": min(vals),
        "max": max(vals),
    }


def shortlist_best(rows: List[Dict[str, Any]],
                   goal_col: str,
                   best_n: int = 5) -> List[int]:
    """Return a list of 1‑based row indices of the best rows seen so far."""
    vals: List[Tuple[float, int]] = []  # (score, row_index)
    for idx, r in enumerate(rows, start=1):
        v = r.get(goal_col, "?")
        if v == "?":
            continue
        try:
            val = float(v)
        except Exception:
            continue
        vals.append((val, idx))
    if not vals:
        return []

    # Sort by best for the goal direction
    direction = goal_direction(goal_col)
    reverse = direction == "max"
    vals.sort(key=lambda x: x[0], reverse=reverse)
    return [idx for _, idx in vals[:best_n]]


def assess_arrivals(rows: List[Dict[str, Any]],
                    goal_col: str,
                    arrivals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return assessment list for each arrival row."""
    assessments: List[Dict[str, Any]] = []
    prior_vals: List[float] = []

    # Load all historical values (rows before arrivals)
    for r in rows:
        v = r.get(goal_col, "?")
        if v == "?":
            continue
        try:
            prior_vals.append(float(v))
        except Exception:
            continue

    direction = goal_direction(goal_col)
    # Determine worst value among prior
    if direction == "min":
        # Worst is max prior
        worst_value = max(prior_vals) if prior_vals else None
    else:
        worst_value = min(prior_vals) if prior_vals else None

    for a in arrivals:
        row_num = a["_row_num"]
        v = a.get(goal_col, "?")
        if v == "?":
            alert = False
        else:
            try:
                val = float(v)
            except Exception:
                alert = False
            else:
                if worst_value is None:
                    alert = False
                else:
                    if direction == "min":
                        alert = val > worst_value
                    else:
                        alert = val < worst_value
        assessments.append({"row": row_num, "alert": alert})
        # Update prior values and worst_value
        if v != "?":
            try:
                val = float(v)
                prior_vals.append(val)
                if direction == "min":
                    if val > worst_value:
                        worst_value = val
                else:
                    if val < worst_value:
                        worst_value = val
            except Exception:
                pass
    return assessments


def checkpoints(rows: List[Dict[str, Any]],
                goal_col: str,
                arrivals: List[Dict[str, Any]],
                checkpoint_counts: List[int]) -> List[Dict[str, Any]]:
    """Return checkpoint data for given m values."""
    result: List[Dict[str, Any]] = []
    history_rows = [r for r in rows if not r.get("_is_arrival", False)]
    for m in checkpoint_counts:
        # Use history + first m arrivals
        used_rows = history_rows + arrivals[:m]
        # Stats
        stats = compute_stats(used_rows, goal_col)
        # Shortlist best 5
        best_rows = shortlist_best(used_rows, goal_col, best_n=5)
        result.append(
            {
                "after": m,
                "summary": {goal_col: stats},
                "shortlist": best_rows,
            }
        )
    return result


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, headers = read_csv(csv_path)

    # Identify goal column (assume at least one)
    goal_cols = [h for h in headers if is_goal(h)]
    if not goal_cols:
        print("No goal column found.")
        sys.exit(1)
    goal_col = goal_cols[0]

    # Number of rows
    nrows = len(rows)

    # Determine arrivals: last round(0.2*nrows) rows
    m_arrivals = int(round(0.2 * nrows))
    # Mark arrivals
    for idx, r in enumerate(rows, start=1):
        if idx > nrows - m_arrivals:
            r["_is_arrival"] = True
            r["_row_num"] = idx
        else:
            r["_is_arrival"] = False
            r["_row_num"] = idx

    arrivals = [r for r in rows if r["_is_arrival"]]
    history_rows = [r for r in rows if not r["_is_arrival"]]

    # ---- APACHE-07 ----
    assessments = assess_arrivals(rows, goal_col, arrivals)
    print("== APACHE-07 ==")
    print(f"Total arrivals: {len(arrivals)}")
    for a in assessments:
        print(f"Row {a['row']}: alert={a['alert']}")
    print(
        f"ANSWER APACHE-07 {json.dumps({'assessments': assessments}, separators=(',', ':'))}"
    )

    # ---- APACHE-08 ----
    # Checkpoints after 0, 19, and 38 arrivals
    checkpoint_counts = [0, 19, 38]
    # Ensure we don't exceed number of arrivals
    checkpoint_counts = [c for c in checkpoint_counts if c <= m_arrivals]
    checkpoints_data = checkpoints(rows, goal_col, arrivals, checkpoint_counts)

    print("\n== APACHE-08 ==")
    for cp in checkpoints_data:
        print(f"After {cp['after']} arrivals:")
        print(f"  {goal_col} stats: n={cp['summary'][goal_col]['n']}, "
              f"mean={cp['summary'][goal_col]['mean']:.4f}, "
              f"min={cp['summary'][goal_col]['min']:.4f}, "
              f"max={cp['summary'][goal_col]['max']:.4f}")
        print(f"  Shortlist top 5 rows: {cp['shortlist']}")
    print(
        f"ANSWER APACHE-08 {json.dumps({'checkpoints': checkpoints_data}, separators=(',', ':'))}"
    )


if __name__ == "__main__":
    main()
