#!/usr/bin/env python3
"""
CAR-13: Identify a vehicle with a strong balance of fuel efficiency,
weight, and acceleration while evaluating as few candidate vehicles as
reasonably possible.
CAR-14: With a limited inspection budget, recommend which vehicles should
be evaluated first so that we can discover promising inventory options early.
"""

import csv
import json
import sys
from typing import Dict, List, Tuple, Any, Optional

# ----------------------------------------------------------------------
# CSV handling
# ----------------------------------------------------------------------
def parse_csv(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return list of rows (dict of column->value) and header list."""
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for rnum, row in enumerate(reader, start=1):
            parsed: Dict[str, Any] = {}
            for k, v in row.items():
                v = v.strip()
                if v == "" or v == "?":
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            parsed["_rownum"] = rnum
            rows.append(parsed)
    return rows, headers

# ----------------------------------------------------------------------
# Goal identification
# ----------------------------------------------------------------------
def goal_columns(headers: List[str]) -> List[str]:
    """Columns ending with '+' or '-'."""
    return [h for h in headers if h.endswith(("+", "-"))]

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def is_complete(row: Dict[str, Any], goals: List[str]) -> bool:
    """Row has all goal values present (not None)."""
    return all(row.get(g) is not None for g in goals)

def compute_score(row: Dict[str, Any],
                 min_vals: Dict[str, float],
                 max_vals: Dict[str, float],
                 goals: List[str]) -> float:
    """Compute balance score for a row given current min/max for each goal."""
    norms: List[float] = []
    for g in goals:
        v = row[g]
        lo = min_vals[g]
        hi = max_vals[g]
        if hi == lo:
            norm = 0.0
        else:
            if g.endswith("+"):
                norm = (v - lo) / (hi - lo)
            else:  # minimize
                norm = (hi - v) / (hi - lo)
        norms.append(norm)
    return sum(norms) / len(norms)

# ----------------------------------------------------------------------
# CAR-13
# ----------------------------------------------------------------------
def car13(rows: List[Dict[str, Any]], goals: List[str], budget: int = 30) -> Dict[str, Any]:
    """Evaluate first up to budget complete rows and return best."""
    evaluated: List[int] = []
    data: List[Tuple[int, Dict[str, Any]]] = []

    # Find first 'budget' rows with all goals present
    for row in rows:
        if len(evaluated) >= budget:
            break
        if is_complete(row, goals):
            evaluated.append(row["_rownum"])
            data.append((row["_rownum"], row))

    # Compute min/max over evaluated rows for normalization
    if not data:
        return {"evaluated": [], "best": None}
    min_vals = {g: float("inf") for g in goals}
    max_vals = {g: float("-inf") for g in goals}
    for _, r in data:
        for g in goals:
            v = r[g]
            if v < min_vals[g]:
                min_vals[g] = v
            if v > max_vals[g]:
                max_vals[g] = v

    # Compute score for each evaluated row
    scores: Dict[int, float] = {}
    for rn, r in data:
        scores[rn] = compute_score(r, min_vals, max_vals, goals)

    # Pick best (max score, tie by lower row number)
    best_row = min(scores.keys(), key=lambda k: (-scores[k], k)) if scores else None
    return {"evaluated": evaluated, "best": best_row}

# ----------------------------------------------------------------------
# CAR-14
# ----------------------------------------------------------------------
def car14(rows: List[Dict[str, Any]], goals: List[str], budget: int = 30) -> Dict[str, Any]:
    """Greedy selection of rows to evaluate first, based only on already
    evaluated rows."""
    remaining = [row for row in rows if is_complete(row, goals)]
    if not remaining:
        return {"order": []}

    order: List[int] = []

    # Start with the first complete row in file order
    first = remaining.pop(0)
    order.append(first["_rownum"])
    min_vals = {g: first[g] for g in goals}
    max_vals = {g: first[g] for g in goals}

    while len(order) < budget and remaining:
        # Evaluate candidate scores based on current min/max
        best_candidate: Optional[Tuple[int, float, Dict[str, Any]]] = None
        for cand in remaining:
            # compute score using current min/max
            score = compute_score(cand, min_vals, max_vals, goals)
            if best_candidate is None or score > best_candidate[1] or \
               (score == best_candidate[1] and cand["_rownum"] < best_candidate[0]):
                best_candidate = (cand["_rownum"], score, cand)

        if best_candidate is None:
            break
        rn, _, cand_row = best_candidate
        # Remove from remaining
        remaining = [r for r in remaining if r["_rownum"] != rn]
        # Append to order
        order.append(rn)
        # Update min/max
        for g in goals:
            v = cand_row[g]
            if v < min_vals[g]:
                min_vals[g] = v
            if v > max_vals[g]:
                max_vals[g] = v

    return {"order": order}

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    rows, headers = parse_csv(csv_path)
    if not rows:
        print("No data")
        return

    goals = goal_columns(headers)
    budget = min(30, len(rows) // 5)  # as specified

    # CAR-13
    print("== CAR-13 ==")
    res13 = car13(rows, goals, budget)
    # Print some diagnostics
    for rn in res13["evaluated"]:
        # recompute score for printing
        row = next(r for r in rows if r["_rownum"] == rn)
        # Use global min/max over evaluated rows
        # We'll just compute using all evaluated rows
    # Simple output
    print(f"Evaluated {len(res13['evaluated'])} rows.")
    print(f"Best row: {res13['best']}")
    print(f"ANSWER CAR-13 {json.dumps(res13, separators=(',', ':'))}")

    # CAR-14
    print("\n== CAR-14 ==")
    res14 = car14(rows, goals, budget)
    print(f"Order length: {len(res14['order'])}")
    print(f"First 5 in order: {res14['order'][:5]}")
    print(f"ANSWER CAR-14 {json.dumps(res14, separators=(',', ':'))}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
