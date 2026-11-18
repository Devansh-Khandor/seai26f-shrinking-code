#!/usr/bin/env python3
"""
skill_07.py

APACHE-13 – Discover a configuration with a low performance-metric value while
evaluating as few candidate configurations as reasonably possible.

APACHE-14 – When I have a limited benchmarking budget, recommend which
configurations should be evaluated first so that we can discover promising
server settings early.

The program prints a section for each request and a machine‑checked answer
line.  It uses only the goal values of rows already evaluated/ordered.
"""

import csv
import json
import sys
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[Tuple[int, Dict[str, str]]], List[str]]:
    """Return a list of (row_number, row_dict) and fieldnames."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for rn, rec in enumerate(reader, start=1):
            rows.append((rn, rec))
    return rows, fieldnames


def find_goal_col(fieldnames: List[str]) -> str:
    """Return the first goal column that ends with '-' (minimize).  If none,
    return the first column that ends with '+'."""
    for c in fieldnames:
        if c.endswith("-"):
            return c
    for c in fieldnames:
        if c.endswith("+"):
            return c
    raise ValueError("No goal column found")


def is_x(col: str) -> bool:
    return col.endswith("X")


def feature_cols(fieldnames: List[str], goal_col: str) -> List[str]:
    """Return all columns that can be used as features."""
    return [
        c
        for c in fieldnames
        if not c.endswith("+")
        and not c.endswith("-")
        and not is_x(c)
    ]


def to_float(v: str):
    if v in (None, "?"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


# ----------------------------------------------------------------------
# Distance over feature columns
# ----------------------------------------------------------------------
def feature_distance(rn1: int, rn2: int, rows_dict: Dict[int, Dict[str, str]],
                     fcols: List[str]) -> float:
    """Euclidean distance over numeric features; 0/1 for categorical."""
    rec1 = rows_dict[rn1]
    rec2 = rows_dict[rn2]
    sq = 0.0
    cnt = 0
    for col in fcols:
        v1 = rec1.get(col)
        v2 = rec2.get(col)
        if v1 in (None, "?") or v2 in (None, "?"):
            continue
        # numeric?
        try:
            f1 = float(v1)
            f2 = float(v2)
            sq += (f1 - f2) ** 2
        except ValueError:
            # categorical
            sq += 0 if v1 == v2 else 1
        cnt += 1
    return sq ** 0.5 if cnt > 0 else float("inf")


# ----------------------------------------------------------------------
# Selection algorithm
# ----------------------------------------------------------------------
def select_rows(rows: List[Tuple[int, Dict[str, str]]],
                budget: int,
                goal_col: str,
                fcols: List[str]) -> List[int]:
    """
    Greedy selection: start with the first row, then repeatedly
    pick the row whose nearest neighbour among already chosen rows
    has the lowest goal value.  Uses only the goal values of chosen rows.
    """
    # Build dictionary for fast lookup
    rows_dict: Dict[int, Dict[str, str]] = {}
    for rn, rec in rows:
        # Keep original string for printing, but store goal as float
        rec2 = dict(rec)
        goal_val = to_float(rec2.get(goal_col))
        rec2[goal_col] = goal_val
        rows_dict[rn] = rec2

    # Keep only rows that have a valid goal value
    valid_rns = [rn for rn, rec in rows if to_float(rec.get(goal_col)) is not None]
    if not valid_rns:
        return []

    # Shuffle not necessary: start with first valid row
    remaining = set(valid_rns)
    selected: List[int] = []

    chosen: List[int] = []

    while len(selected) < budget and remaining:
        if not chosen:
            # Pick the first available row
            rn = min(remaining)
            remaining.remove(rn)
            selected.append(rn)
            chosen.append(rn)
        else:
            # For each remaining row, predict goal via nearest neighbour
            best_rn = None
            best_pred = None
            for rn in remaining:
                # Find nearest neighbour among chosen
                best_dist = float("inf")
                pred_goal = None
                for cn in chosen:
                    d = feature_distance(rn, cn, rows_dict, fcols)
                    if d < best_dist:
                        best_dist = d
                        pred_goal = rows_dict[cn][goal_col]
                if pred_goal is None:
                    continue
                if best_pred is None or pred_goal < best_pred:
                    best_pred = pred_goal
                    best_rn = rn
            if best_rn is None:
                # Fallback: take the first remaining
                rn = min(remaining)
                remaining.remove(rn)
                selected.append(rn)
                chosen.append(rn)
            else:
                remaining.remove(best_rn)
                selected.append(best_rn)
                chosen.append(best_rn)

    return selected


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)
    goal_col = find_goal_col(fieldnames)
    fcols = feature_cols(fieldnames, goal_col)

    # Budget: min(30, number_of_rows // 5)
    total_rows = len(rows)
    budget = min(30, total_rows // 5)
    if budget == 0:
        budget = 1

    # -------- APACHE-13 ----------
    eval_order = select_rows(rows, budget, goal_col, fcols)
    if eval_order:
        # Determine best row among those evaluated
        best_rn = min(
            eval_order, key=lambda rn: rows_dict[goal_col, rn]
        )
    else:
        best_rn = None

    print("== APACHE-13 ==")
    print(f"Evaluated rows: {len(eval_order)}")
    if eval_order:
        best_val = rows_dict[best_rn][goal_col]
        print(f"Best row: {best_rn}")
        print(f"Best performance: {best_val}")
    else:
        print("No rows evaluated.")

    # Build dictionary again for answer extraction
    rows_dict: Dict[int, Dict[str, str]] = {}
    for rn, rec in rows:
        rec2 = dict(rec)
        goal_val = to_float(rec2.get(goal_col))
        rec2[goal_col] = goal_val
        rows_dict[rn] = rec2
    # For best row, compute again
    if eval_order:
        best_rn = min(eval_order, key=lambda rn: rows_dict[rn][goal_col])
        best_val = rows_dict[best_rn][goal_col]
    else:
        best_rn = None
        best_val = None

    print(
        "ANSWER APACHE-13",
        json.dumps({"evaluated": eval_order, "best": best_rn}, separators=(",", ":"))
    )

    # -------- APACHE-14 ----------
    order = select_rows(rows, budget, goal_col, fcols)

    print("\n== APACHE-14 ==")
    print(f"Order length: {len(order)}")
    print(f"First 5 rows: {order[:5]}")
    print(f"Last 5 rows: {order[-5:]}")

    print(
        "ANSWER APACHE-14",
        json.dumps({"order": order}, separators=(",", ":")),
    )


if __name__ == "__main__":
    main()
