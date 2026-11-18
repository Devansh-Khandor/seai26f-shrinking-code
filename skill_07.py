#!/usr/bin/env python3
"""
skill_07.py <csv_path>
Implements CAR-13 (balanced vehicle identification) and CAR-14 (evaluation order recommendation).
"""

import csv
import json
import sys
from typing import Dict, List, Tuple, Optional

# --------------------------------------------------------------------------- #
# Utility functions
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
    """Return min and max for each goal column."""
    minmax: Dict[str, Tuple[float, float]] = {}
    for g in goals:
        values = [row[1][g] for row in rows if row[1].get(g) is not None]
        if values:
            minmax[g] = (min(values), max(values))
    return minmax

def composite_score(row: Dict[str, Optional[float]],
                    minmax: Dict[str, Tuple[float, float]],
                    goals: List[str]) -> Optional[float]:
    """Return normalized composite score (average of normalized goals)."""
    norms = []
    for g in goals:
        v = row.get(g)
        if v is None or g not in minmax:
            return None
        mn, mx = minmax[g]
        if mn == mx:
            n = 0.0
        else:
            n = (v - mn) / (mx - mn) if g.endswith('+') else (mx - v) / (mx - mn)
        norms.append(n)
    return sum(norms) / len(norms)

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

    # Budget: at most min(30, number of rows // 5)
    budget = min(30, len(rows) // 5)

    # ------------------------------------------------------------------- #
    # CAR-13: balanced vehicle identification
    # ------------------------------------------------------------------- #
    # Compute composite for rows with all goal values
    scored: List[Tuple[int, float]] = []
    for row_num, data in rows:
        score = composite_score(data, minmax, goals)
        if score is not None:
            scored.append((row_num, score))

    # Sort by score descending, tie-breaking by row number ascending
    scored.sort(key=lambda x: (-x[1], x[0]))

    # Pick the first `budget` rows as evaluated
    evaluated = [row_num for row_num, _ in scored[:budget]]
    best = evaluated[0] if evaluated else None

    # Print CAR-13
    print("== CAR-13 ==")
    print(f"Evaluated {len(evaluated)} vehicles (budget={budget}).")
    print(f"Best balanced vehicle: Row {best}")
    print(f"ANSWER CAR-13 {json.dumps({'evaluated': evaluated, 'best': best})}")

    # ------------------------------------------------------------------- #
    # CAR-14: evaluation order recommendation
    # ------------------------------------------------------------------- #
    # For each row with all goal values, compute composite
    scored_all = [(row_num, composite_score(data, minmax, goals)) for row_num, data in rows]
    scored_all = [(r, s) for r, s in scored_all if s is not None]
    scored_all.sort(key=lambda x: (-x[1], x[0]))
    order = [r for r, _ in scored_all[:budget]]

    # Print CAR-14
    print("\n== CAR-14 ==")
    print(f"Recommended evaluation order (first {budget} vehicles):")
    print("  ", ", ".join(map(str, order)))
    print(f"ANSWER CAR-14 {json.dumps({'order': order})}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_07.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])
