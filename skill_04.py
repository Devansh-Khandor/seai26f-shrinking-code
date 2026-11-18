#!/usr/bin/env python3
"""
skill_04.py: Incremental assessment and inventory summary for used‑car dealership.
"""

import csv
import json
import sys
from collections import defaultdict
from math import sqrt

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

def parse_cell(s: str):
    """Convert a CSV cell to int, float, or keep string; '?' -> None."""
    if s == "?":
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s

def read_csv(path: str):
    """Return header list and rows list (each row is a list of parsed cells)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [list(map(parse_cell, row)) for row in reader]
    return header, rows

# --------------------------------------------------------------------------- #
# Core logic for CAR‑07 and CAR‑08
# --------------------------------------------------------------------------- #

def is_goal(col: str) -> bool:
    return col.endswith(("+", "-"))

def compute_stats(rows, goal_idxs):
    """Compute n, mean, min, max for each goal column across given rows."""
    stats = {}
    for idx in goal_idxs:
        values = [row[idx] for row in rows if row[idx] is not None]
        n = len(values)
        if n == 0:
            mean = min_val = max_val = None
        else:
            mean = sum(values) / n
            min_val = min(values)
            max_val = max(values)
        stats[idx] = {"n": n, "mean": mean, "min": min_val, "max": max_val}
    return stats

def composite_score(row, goal_idxs, mins, maxs):
    """Return composite score for a row given min/max for each goal."""
    norms = []
    for idx in goal_idxs:
        val = row[idx]
        if val is None:
            return None  # cannot score if missing any goal
        mn, mx = mins[idx], maxs[idx]
        if mn == mx:
            norm = 0.5
        else:
            if header[idx].endswith("-"):   # minimize
                norm = (mx - val) / (mx - mn)
            else:                           # maximize
                norm = (val - mn) / (mx - mn)
        norms.append(norm)
    return sum(norms) / len(norms)

# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    global header
    header, raw_rows = read_csv(csv_path)

    # Map row numbers (1‑based)
    rows = [(i+1, r) for i, r in enumerate(raw_rows)]

    # Identify goal columns
    goal_idxs = [i for i, c in enumerate(header) if is_goal(c)]
    goal_names = [header[i] for i in goal_idxs]

    # Determine history vs arrivals
    n_total = len(rows)
    n_arrivals = round(0.2 * n_total)
    arrivals = rows[-n_arrivals:]            # last n_arrivals rows
    history = rows[:n_total - n_arrivals]     # remaining rows

    # ----------------- CAR-07: Arrival assessments ----------------- #
    assessments = []

    # For each arrival in order
    for i, (row_num, row) in enumerate(arrivals):
        # rows seen so far
        seen_rows = history + arrivals[:i+1]
        # compute min/max per goal using seen_rows
        seen_vals = [r for _, r in seen_rows]
        stats = compute_stats(seen_vals, goal_idxs)
        mins = {idx: stats[idx]["min"] for idx in goal_idxs}
        maxs = {idx: stats[idx]["max"] for idx in goal_idxs}

        # Compute composite scores for all rows with complete goals
        scored = []
        for rn, r in seen_rows:
            score = composite_score(r, goal_idxs, mins, maxs)
            if score is not None:
                scored.append((score, rn))

        # Rank by score descending
        scored.sort(reverse=True, key=lambda x: x[0])

        # Determine if current arrival is in top 5
        top5_ids = [rn for _, rn in scored[:5]]
        desirable = row_num in top5_ids

        assessments.append({"row": row_num, "desirable": desirable})

    # ----------------- CAR-08: Checkpoints ----------------- #
    checkpoints = []
    for m in (0, 40, 80):
        # m arrivals after history
        seen_rows = history + arrivals[:m]
        seen_vals = [r for _, r in seen_rows]
        stats = compute_stats(seen_vals, goal_idxs)
        summary = {}
        for idx in goal_idxs:
            c = header[idx]
            s = stats[idx]
            # skip if all missing
            if s["n"] == 0:
                continue
            summary[c] = {"n": s["n"],
                          "mean": s["mean"],
                          "min": s["min"],
                          "max": s["max"]}

        # Shortlist: 5 best rows by composite score
        mins = {idx: stats[idx]["min"] for idx in goal_idxs}
        maxs = {idx: stats[idx]["max"] for idx in goal_idxs}
        scored = []
        for rn, r in seen_rows:
            score = composite_score(r, goal_idxs, mins, maxs)
            if score is not None:
                scored.append((score, rn))
        scored.sort(reverse=True, key=lambda x: x[0])
        shortlist = [rn for _, rn in scored[:5]]

        checkpoints.append({"after": m,
                            "summary": summary,
                            "shortlist": shortlist})

    # ----------------- Output ----------------- #
    # CAR-07
    print("== CAR-07 ==")
    for a in assessments:
        print(f"Row {a['row']}: desirable={a['desirable']}")
    print("ANSWER CAR-07", json.dumps({"assessments": assessments}))

    # CAR-08
    print("\n== CAR-08 ==")
    for cp in checkpoints:
        print(f"After {cp['after']} arrivals:")
        print("  Summary:")
        for g, s in cp['summary'].items():
            print(f"    {g}: n={s['n']} mean={s['mean']} min={s['min']} max={s['max']}")
        print("  Shortlist:", cp['shortlist'])
    print("ANSWER CAR-08", json.dumps({"checkpoints": checkpoints}))
