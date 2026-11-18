#!/usr/bin/env python3
"""
skill_07.py: CAR-13 & CAR-14 – balance of fuel efficiency, weight, acceleration and
               inspection ordering.

The code reads the CSV file supplied on the command line, computes a
composite score for each vehicle that balances the goals
( Lbs-  – weight to minimise,
  Acc+  – acceleration to maximise,
  Mpg+  – fuel‑efficiency to maximise ).
Only rows that have all goal values present are considered.
The best 30 (or fewer if fewer rows exist) rows are evaluated and reported.
The same order is used for CAR‑14 – the vehicles that should be inspected first.
"""

import csv
import json
import sys
from typing import List, Tuple, Dict, Any, Optional


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def parse_cell(s: str):
    """Coerce a CSV cell into int/float/string, or None for '?'."""
    if s == "?":
        return None
    for fn in (int, float):
        try:
            return fn(s)
        except ValueError:
            continue
    return s


def composite_score(row: List[Optional[float]],
                    goal_indices: List[int],
                    header: List[str],
                    mins: Dict[int, float],
                    maxs: Dict[int, float]) -> Optional[float]:
    """Compute the composite score (average of normalised goals)."""
    scores = []
    for idx in goal_indices:
        val = row[idx]
        if val is None:
            return None
        lo, hi = mins[idx], maxs[idx]
        if hi == lo:
            norm = 0.5
        else:
            if header[idx].endswith("-"):           # minimise
                norm = (hi - val) / (hi - lo)
            else:                                   # maximise
                norm = (val - lo) / (hi - lo)
        scores.append(norm)
    return sum(scores) / len(scores)


# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    # Load data
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows_data: List[Tuple[int, List[Optional[float]]]] = []
        for i, line in enumerate(reader, 1):          # 1‑based row numbers
            rows_data.append((i, [parse_cell(v) for v in line]))

    # Identify goal columns (+ or - suffix)
    goal_indices = [i for i, col in enumerate(header)
                    if col.endswith(("+", "-"))]

    # Compute global min/max for each goal (ignoring missing)
    mins: Dict[int, float] = {}
    maxs: Dict[int, float] = {}
    for idx in goal_indices:
        vals = [row[idx] for _, row in rows_data if row[idx] is not None]
        if vals:
            mins[idx] = min(vals)
            maxs[idx] = max(vals)

    # Build list of (row_num, composite_score) for rows that have all goals
    scored_rows: List[Tuple[int, float]] = []
    for row_num, row in rows_data:
        if all(row[idx] is not None for idx in goal_indices):
            comp = composite_score(row, goal_indices, header, mins, maxs)
            if comp is not None:
                scored_rows.append((row_num, comp))

    # Sort descending by composite score
    scored_rows.sort(key=lambda x: x[1], reverse=True)

    # ----- CAR-13 -----
    budget = min(30, len(scored_rows))
    evaluated = scored_rows[:budget]
    eval_nums = [rn for rn, _ in evaluated]
    best_num = eval_nums[0] if eval_nums else None
    best_score = evaluated[0][1] if evaluated else None

    print("== CAR-13 ==")
    print("Evaluated rows:", ", ".join(map(str, eval_nums)))
    # Show top 5 composite scores for reference
    top_scores = ", ".join(f"{rn}:{c:.4f}" for rn, c in evaluated[:5])
    print("Composite scores (top 5):", top_scores)
    if best_num is not None:
        print(f"Best row: {best_num} (score={best_score:.4f})")
    else:
        print("No rows available for evaluation.")

    # JSON answer
    print(f"ANSWER CAR-13 {json.dumps({'evaluated': eval_nums, 'best': best_num})}")

    # ----- CAR-14 -----
    order = eval_nums

    print("\n== CAR-14 ==")
    print("Evaluation order:", ", ".join(map(str, order)))
    print(f"Number of rows to evaluate: {len(order)}")
    print(f"ANSWER CAR-14 {json.dumps({'order': order})}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
