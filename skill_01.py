#!/usr/bin/env python3
"""
skill_01.py: Summary and ranking for the used‑car inventory.
"""

import csv
import json
import sys
from collections import defaultdict
from statistics import mean as _mean

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

def parse_cell(s: str):
    """Convert a CSV cell to int, float, or keep string; '?' -> None."""
    if s == "?":
        return None
    # try int first
    try:
        return int(s)
    except ValueError:
        pass
    # try float
    try:
        return float(s)
    except ValueError:
        pass
    return s

def median(values):
    """Median of a list of numbers.  Assumes len > 0."""
    n = len(values)
    vs = sorted(values)
    mid = n // 2
    if n % 2:
        return vs[mid]
    return (vs[mid - 1] + vs[mid]) / 2

# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    # Load data
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [list(map(parse_cell, row)) for row in reader]

    # Determine goal columns (ending with '+' or '-')
    goal_cols = [c for c in header if c.endswith(("+", "-"))]

    # Map column name to its index
    idx = {name: i for i, name in enumerate(header)}

    # Collect values per goal column (skip missing)
    goal_values = {col: [] for col in goal_cols}
    for r in rows:
        for col in goal_cols:
            val = r[idx[col]]
            if val is not None:
                goal_values[col].append(val)

    # Compute CAR-01 statistics
    summary = {}
    for col in goal_cols:
        vals = goal_values[col]
        vals_sorted = sorted(vals)
        n = len(vals)
        col_mean = _mean(vals)
        col_median = median(vals)
        col_min = min(vals)
        col_max = max(vals)
        if col.endswith("+"):
            best = col_max
            worst = col_min
        else:
            best = col_min
            worst = col_max
        summary[col] = {
            "mean": col_mean,
            "median": col_median,
            "min": col_min,
            "max": col_max,
            "best": best,
            "worst": worst,
        }

    # Output CAR-01 section
    print("== CAR-01 ==")
    for col in goal_cols:
        s = summary[col]
        # Format numbers: 2 decimal for mean/median, integer for others
        print(
            f"{col} mean: {s['mean']:.2f} median: {s['median']:.2f} "
            f"min: {s['min']} max: {s['max']} best: {s['best']} worst: {s['worst']}"
        )
    # JSON answer line
    json_out = json.dumps(summary, separators=(",", ":"))
    print(f"ANSWER CAR-01 {json_out}")

    # ----------------------------------------------------------------------- #
    # CAR-02: composite ranking
    # ----------------------------------------------------------------------- #
    # We only use Lbs-, Acc+, Mpg+ as per spec
    # Ensure columns exist
    required = ["Lbs-", "Acc+", "Mpg+"]
    if not all(c in header for c in required):
        print("ERROR: Required goal columns missing", file=sys.stderr)
        sys.exit(1)

    # Compute global min/max for each
    globals_minmax = {}
    for col in required:
        vals = [r[idx[col]] for r in rows if r[idx[col]] is not None]
        globals_minmax[col] = (min(vals), max(vals))

    # Score each row
    scored = []
    for row_num, r in enumerate(rows, start=1):
        # Skip rows with any missing goal
        if any(r[idx[col]] is None for col in required):
            continue
        score = 0.0
        for col in required:
            val = r[idx[col]]
            lo, hi = globals_minmax[col]
            if lo == hi:
                norm = 0.5  # avoid division by zero
            else:
                if col.endswith("-"):
                    norm = (hi - val) / (hi - lo)
                else:
                    norm = (val - lo) / (hi - lo)
            score += norm
        score /= len(required)  # average
        scored.append((score, row_num))

    # Top 10 rows by composite score
    scored.sort(reverse=True)
    top_rows = [num for _, num in scored[:10]]

    # Output CAR-02 section
    print("\n== CAR-02 ==")
    print("Top 10 rows (most attractive combination):")
    print(", ".join(map(str, top_rows)))
    # JSON answer line
    json_top = json.dumps({"top": top_rows}, separators=(",", ":"))
    print(f"ANSWER CAR-02 {json_top}")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_01.py <csv path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
