#!/usr/bin/env python3
"""
skill_06.py <csv_path>
Implements CAR-11 (recommendation explanation) and CAR-12 (trade‑off explanation).
"""

import csv
import json
import math
import sys
from typing import Dict, List, Tuple, Optional

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def parse_value(v: str) -> Optional[float]:
    """Convert a CSV string to a float or None."""
    if v == '?' or v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None

def load_rows(csv_path: str) -> List[Tuple[int, Dict[str, Optional[float]]]]:
    """Read the CSV and return a list of (row_number, data_dict)."""
    rows: List[Tuple[int, Dict[str, Optional[float]]]] = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            parsed: Dict[str, Optional[float]] = {}
            for k, v in row.items():
                parsed[k] = parse_value(v)
            rows.append((i, parsed))
    return rows

def goal_columns(headers: List[str]) -> List[str]:
    """Return columns ending with '+' or '-'.  Columns ending with 'X' are excluded."""
    return [h for h in headers if h.endswith('+') or h.endswith('-')]

def compute_min_max(rows: List[Tuple[int, Dict[str, Optional[float]]]],
                    goals: List[str]) -> Dict[str, Tuple[float, float]]:
    minmax: Dict[str, Tuple[float, float]] = {}
    for g in goals:
        vals = [row[1][g] for row in rows if row[1][g] is not None]
        if vals:
            minmax[g] = (min(vals), max(vals))
    return minmax

def composite_score(row: Dict[str, Optional[float]],
                    minmax: Dict[str, Tuple[float, float]],
                    goals: List[str]) -> Optional[float]:
    """Return normalized composite score (average of normalized goals)."""
    norm = []
    for g in goals:
        v = row.get(g)
        if v is None or g not in minmax:
            return None
        mn, mx = minmax[g]
        if mn == mx:
            n = 0.0
        else:
            if g.endswith('+'):
                n = (v - mn) / (mx - mn)
            else:
                n = (mx - v) / (mx - mn)
        norm.append(n)
    return sum(norm) / len(norm)

def euclidean_distance(row_a: Dict[str, Optional[float]],
                       row_b: Dict[str, Optional[float]],
                       goals: List[str]) -> float:
    """Return squared Euclidean distance between two rows on goal columns."""
    dist2 = 0.0
    for g in goals:
        a = row_a[g]
        b = row_b[g]
        if a is None or b is None:
            continue
        dist2 += (a - b) ** 2
    return dist2

# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    rows = load_rows(csv_path)
    if not rows:
        print("No data.")
        return

    headers = list(rows[0][1].keys())
    goals = goal_columns(headers)
    minmax = compute_min_max(rows, goals)

    # Budget: evaluate at most N rows
    budget = min(30, len(rows) // 5)

    # Evaluate composite scores for the first 'budget' rows
    evaluated: List[Tuple[int, float, Dict[str, Optional[float]]]] = []
    for row_num, data in rows[:budget]:
        score = composite_score(data, minmax, goals)
        if score is not None:
            evaluated.append((row_num, score, data))

    if not evaluated:
        print("== CAR-11 ==")
        print("No evaluated rows with complete goal data.")
        print("ANSWER CAR-11 {}")
        print()
        print("== CAR-12 ==")
        print("No evaluated rows with complete goal data.")
        print("ANSWER CAR-12 {}")
        return

    # ---- CAR-11 ----
    # Recommended vehicle: highest composite score among evaluated
    recommended = max(evaluated, key=lambda x: x[1])
    rec_num, rec_score, rec_data = recommended

    # Find nearest neighbors among all rows with complete goal data
    all_with_goals = [ (n,d) for n,d in rows if all(d.get(g) is not None for g in goals) ]
    neighbors = []
    for n, d in all_with_goals:
        if n == rec_num:
            continue
        dist = euclidean_distance(rec_data, d, goals)
        neighbors.append((dist, n, d))
    neighbors.sort(key=lambda x: x[0])
    top_neighbors = neighbors[:3]

    # Build JSON output for CAR-11
    car11_output = {"recommended": rec_num, "neighbors": []}
    for _, n, d in top_neighbors:
        values = {g: d[g] for g in goals}
        car11_output["neighbors"].append({"row": n, "values": values})

    print("== CAR-11 ==")
    print(f"Recommended row: {rec_num} (score {rec_score:.4f})")
    print("Top 3 neighbors:")
    for nb in car11_output["neighbors"]:
        print(f"  Row {nb['row']}: {nb['values']}")
    print(f"ANSWER CAR-11 {json.dumps(car11_output)}")

    # ---- CAR-12 ----
    # Pick the two best evaluated rows
    if len(evaluated) < 2:
        print("\n== CAR-12 ==")
        print("Not enough evaluated rows to compare.")
        print("ANSWER CAR-12 {}")
        return

    evaluated_sorted = sorted(evaluated, key=lambda x: x[1], reverse=True)
    a_num, a_score, a_data = evaluated_sorted[0]
    b_num, b_score, b_data = evaluated_sorted[1]

    # Compute differences and tradeoffs
    differences = []
    tradeoffs = []
    for g in goals:
        a_val = a_data[g]
        b_val = b_data[g]
        if a_val != b_val:
            differences.append({"feature": g, "a": a_val, "b": b_val})
        # Determine if b is better
        if a_val is not None and b_val is not None:
            if g.endswith('+') and b_val > a_val:
                tradeoffs.append(g)
            if g.endswith('-') and b_val < a_val:
                tradeoffs.append(g)

    car12_output = {
        "a": a_num,
        "b": b_num,
        "preferred": a_num,
        "scores": {str(a_num): a_score, str(b_num): b_score},
        "differences": differences[:3],  # at least 1, but cap at 3 for brevity
        "tradeoffs": tradeoffs[:3]  # at least 1
    }

    print("\n== CAR-12 ==")
    print(f"Preferred row {a_num} (score {a_score:.4f}) over row {b_num} (score {b_score:.4f})")
    print("Differences:")
    for d in car12_output["differences"]:
        print(f"  {d['feature']}: {a_num}={d['a']}, {b_num}={d['b']}")
    print("Trade‑offs (better in b):", ", ".join(car12_output["tradeoffs"]))
    print(f"ANSWER CAR-12 {json.dumps(car12_output)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_06.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])
