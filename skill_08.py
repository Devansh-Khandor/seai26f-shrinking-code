#!/usr/bin/env python3
"""
skill_08.py: CAR-15 & CAR-16 – alternatives for poorly rated and under‑target vehicles.

The program reads the CSV supplied on the command line and performs two queries:

* CAR‑15 – Find the worst‑performing vehicle (by composite score) and list the
  best historical alternatives that are similar in the input features but have a
  higher composite score.  For each alternative we list the input columns that
  differ, with the original value followed by the alternative value.

* CAR‑16 – Find a vehicle that meets fewer performance targets than the
  preferred thresholds (median of each goal).  Then find the closest historical
  alternative that meets more targets while changing as few input columns as
  possible.  The output lists which input columns differ.

Both outputs include a machine‑checked JSON line that follows the format described
in the specification.

All rows are kept – missing values are handled per‑cell – and the notion of
``history`` is the set of rows that are not among the last ``round(0.2 * n)`` rows.
The last rows are the ``arrivals``.  The code works for any CSV that follows the
conventions described in the project description.
"""

import csv
import json
import math
import sys
from collections import defaultdict
from statistics import median
from typing import List, Tuple, Dict, Any, Optional, Set

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def parse_cell(s: str):
    """Coerce a CSV cell to int/float/string, or None for '?'."""
    if s == "?":
        return None
    for fn in (int, float):
        try:
            return fn(s)
        except ValueError:
            continue
    return s

def read_csv(path: str) -> Tuple[List[str], List[Tuple[int, List[Any]]]]:
    """Read the file and return header and rows with 1‑based row numbers."""
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows: List[Tuple[int, List[Any]]] = []
        for i, line in enumerate(reader, 1):
            rows.append((i, [parse_cell(v) for v in line]))
    return header, rows

def is_goal(col: str) -> bool:
    return col.endswith(("+", "-"))

def is_input(col: str) -> bool:
    return not col.endswith("X") and not is_goal(col)

# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def goal_bounds(header: List[str], rows: List[Tuple[int, List[Any]]]
               ) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Return min and max for each goal (numeric only)."""
    bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for idx, name in enumerate(header):
        if not is_goal(name):
            continue
        vals = [row[idx] for _, row in rows if isinstance(row[idx], (int, float))]
        if vals:
            bounds[name] = (min(vals), max(vals))
    return bounds

def goal_medians(header: List[str], rows: List[Tuple[int, List[Any]]]
                ) -> Dict[str, float]:
    """Return median for each goal."""
    med: Dict[str, float] = {}
    for idx, name in enumerate(header):
        if not is_goal(name):
            continue
        vals = [row[idx] for _, row in rows if isinstance(row[idx], (int, float))]
        if vals:
            med[name] = median(vals)
    return med

def input_min_max(header: List[str], rows: List[Tuple[int, List[Any]]]
                 ) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Return min/max for each numeric input column."""
    bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for idx, name in enumerate(header):
        if not is_input(name):
            continue
        vals = [row[idx] for _, row in rows if isinstance(row[idx], (int, float))]
        if vals:
            bounds[name] = (min(vals), max(vals))
    return bounds

def composite_score(row: List[Any], header: List[str],
                    bounds: Dict[str, Tuple[Optional[float], Optional[float]]]
                   ) -> Optional[float]:
    """Return the average normalised goal score (higher is better)."""
    scores: List[float] = []
    for idx, name in enumerate(header):
        if not is_goal(name):
            continue
        val = row[idx]
        if val is None:
            return None
        lo, hi = bounds[name]
        if hi == lo:
            norm = 0.5
        else:
            if name.endswith("-"):          # minimise
                norm = (hi - val) / (hi - lo)
            else:                           # maximise
                norm = (val - lo) / (hi - lo)
        scores.append(norm)
    return sum(scores) / len(scores)

# --------------------------------------------------------------------------- #
# Distance between rows (input columns only)
# --------------------------------------------------------------------------- #

def input_distance(row1: List[Any], row2: List[Any], header: List[str],
                   minmax: Dict[str, Tuple[Optional[float], Optional[float]]]
                  ) -> float:
    """Euclidean-like distance over input columns, ignoring missing values."""
    total = 0.0
    for idx, name in enumerate(header):
        if not is_input(name):
            continue
        v1, v2 = row1[idx], row2[idx]
        if v1 is None or v2 is None:
            continue
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            lo, hi = minmax[name]
            rng = hi - lo if hi != lo else 1.0
            diff = abs(v1 - v2) / rng
            total += diff ** 2
        else:
            total += 0.0 if v1 == v2 else 1.0
    return math.sqrt(total)

# --------------------------------------------------------------------------- #
# CAR‑15
# --------------------------------------------------------------------------- #

def car_15(header: List[str], rows: List[Tuple[int, List[Any]]]) -> str:
    # Determine historical inventory (not arrivals)
    n_total = len(rows)
    n_arrivals = round(0.2 * n_total)
    arrivals_set: Set[int] = {r[0] for r in rows[-n_arrivals:]}
    historical_rows = [r for r in rows if r[0] not in arrivals_set]

    # Bounds for goals
    g_bounds = goal_bounds(header, rows)

    # Compute composite scores for all rows that have all goals
    scored: List[Tuple[int, float, List[Any]]] = []
    for rn, row in rows:
        score = composite_score(row, header, g_bounds)
        if score is not None:
            scored.append((rn, score, row))
    if not scored:
        # no valid rows
        return "ANSWER CAR-15 {}"

    # Poor row: lowest score
    poor_row_num, poor_score, poor_row = min(scored, key=lambda t: t[1])

    # Bounds for inputs (numeric only)
    in_minmax = input_min_max(header, historical_rows)

    # Find alternatives among historical inventory
    alternatives: List[Tuple[int, float, List[Any]]] = []
    for rn, row in historical_rows:
        # need all goals present to compare performance
        score = composite_score(row, header, g_bounds)
        if score is None or score <= poor_score:
            continue
        # distance to poor row
        dist = input_distance(poor_row, row, header, in_minmax)
        alternatives.append((rn, dist, row))

    if not alternatives:
        # fallback: if no better historical row, just return the poor row alone
        best_alternatives = []
    else:
        # pick up to 3 best (closest) alternatives
        alternatives.sort(key=lambda t: t[1])
        best_alternatives = alternatives[:3]

    # Build JSON answer
    alt_list = []
    for rn, dist, alt_row in best_alternatives:
        changes = {}
        for idx, name in enumerate(header):
            if not is_input(name):
                continue
            pv = poor_row[idx]
            av = alt_row[idx]
            if pv != av:
                changes[name] = [pv, av]
        alt_list.append({"row": rn, "changes": changes})

    answer = {"row": poor_row_num, "alternatives": alt_list}
    return f"ANSWER CAR-15 {json.dumps(answer, separators=(',', ':'))}"

# --------------------------------------------------------------------------- #
# CAR‑16
# --------------------------------------------------------------------------- #

def car_16(header: List[str], rows: List[Tuple[int, List[Any]]]) -> str:
    # Determine historical inventory (not arrivals)
    n_total = len(rows)
    n_arrivals = round(0.2 * n_total)
    arrivals_set: Set[int] = {r[0] for r in rows[-n_arrivals:]}
    historical_rows = [r for r in rows if r[0] not in arrivals_set]

    # Goal thresholds (median)
    thresholds = goal_medians(header, rows)

    # Helper to count targets met
    def targets_met(row: List[Any]) -> int:
        cnt = 0
        for idx, name in enumerate(header):
            if not is_goal(name):
                continue
            val = row[idx]
            if val is None:
                continue
            thresh = thresholds[name]
            if name.endswith("+"):
                if val >= thresh:
                    cnt += 1
            else:
                if val <= thresh:
                    cnt += 1
        return cnt

    # Find a row that fails some targets
    scored: List[Tuple[int, int, float, List[Any]]] = []
    for rn, row in rows:
        # must have all goals to evaluate
        if any(row[idx] is None for idx, name in enumerate(header) if is_goal(name)):
            continue
        tm = targets_met(row)
        score = composite_score(row, header, goal_bounds(header, rows))
        scored.append((rn, tm, score, row))
    if not scored:
        return "ANSWER CAR-16 {}"

    # Select worst: fewest targets met, tie: lower composite
    worst = min(scored, key=lambda t: (t[1], t[2]))

    worst_row_num, worst_tm, worst_score, worst_row = worst

    # Now find alternatives in historical that meet strictly more targets
    candidates: List[Tuple[int, int, float, List[Any]]] = []
    for rn, row in historical_rows:
        tm = targets_met(row)
        if tm <= worst_tm:
            continue
        score = composite_score(row, header, goal_bounds(header, rows))
        if score is None:
            continue
        candidates.append((rn, tm, score, row))

    if not candidates:
        # if no better alternative, just return the worst
        answer = {"row": worst_row_num, "targets": thresholds, "alternative": None, "changed": []}
        return f"ANSWER CAR-16 {json.dumps(answer, separators=(',', ':'))}"

    # Compute distance over inputs
    in_minmax = input_min_max(header, historical_rows)

    def dist(row_a, row_b):
        return input_distance(row_a, row_b, header, in_minmax)

    # Find the one with minimal distance, tie by higher target count
    best = min(candidates, key=lambda t: (dist(worst_row, t[3]), -t[1]))

    alt_row_num, alt_tm, alt_score, alt_row = best

    # Determine changed input columns
    changed = []
    for idx, name in enumerate(header):
        if not is_input(name):
            continue
        pv = worst_row[idx]
        av = alt_row[idx]
        if pv != av:
            # we include the column name only, not the values
            changed.append(name)

    answer = {"row": worst_row_num,
              "targets": thresholds,
              "alternative": alt_row_num,
              "changed": changed}
    return f"ANSWER CAR-16 {json.dumps(answer, separators=(',', ':'))}"

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv path>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    header, rows = read_csv(path)

    # CAR-15
    print("== CAR-15 ==")
    # Add some human readable lines
    print("Finding worst vehicle and best historical alternatives...")
    print(car_15(header, rows))

    # CAR-16
    print("\n== CAR-16 ==")
    print("Finding vehicle under target and best alternative...")
    print(car_16(header, rows))

if __name__ == "__main__":
    main()
