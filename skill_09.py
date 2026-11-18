#!/usr/bin/env python3
"""
skill_09.py: CAR-17 & CAR-18 – flag unusual rows and potential recording mistakes.
"""

import csv
import json
import math
import sys
from collections import defaultdict

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def parse_cell(s: str):
    """Coerce CSV cell to int/float/string; '?' -> None."""
    if s == "?":
        return None
    for fn in (int, float):
        try:
            return fn(s)
        except ValueError:
            continue
    return s

def read_csv(path):
    """Return header list and list of (row_number, row_values)."""
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for i, line in enumerate(reader, 1):
            rows.append((i, [parse_cell(v) for v in line]))
    return header, rows

def is_goal(col: str):
    return col.endswith(("+", "-"))

def is_input(col: str):
    return not col.endswith("X") and not is_goal(col)

# --------------------------------------------------------------------------- #
# Normalisation bounds for numeric input columns
# --------------------------------------------------------------------------- #

def numeric_bounds(header, rows):
    """Return dict of {col: (min, max)} for numeric inputs."""
    bounds = {}
    for idx, name in enumerate(header):
        if not is_input(name):
            continue
        vals = [r[idx] for _, r in rows if isinstance(r[idx], (int, float))]
        if vals:
            bounds[name] = (min(vals), max(vals))
    return bounds

# --------------------------------------------------------------------------- #
# Distance between two rows over input columns
# --------------------------------------------------------------------------- #

def row_distance(r1, r2, header, bounds):
    """Euclidean-like distance over all input columns, ignoring missing."""
    total = 0.0
    for idx, name in enumerate(header):
        if not is_input(name):
            continue
        v1, v2 = r1[idx], r2[idx]
        if v1 is None or v2 is None:
            continue
        if isinstance(v1, (int, float)):
            lo, hi = bounds.get(name, (0, 1))
            rng = hi - lo if hi != lo else 1.0
            diff = abs(v1 - v2) / rng
            total += diff * diff
        else:
            total += 0.0 if v1 == v2 else 1.0
    return math.sqrt(total)

# --------------------------------------------------------------------------- #
# Nearest neighbour helper
# --------------------------------------------------------------------------- #

def k_nearest(row, rows, header, bounds, k=5):
    """Return list of (distance, row_number, row_values) for k nearest neighbours."""
    dists = []
    rnum, rvals = row
    for rnum2, r2 in rows:
        if rnum2 == rnum:
            continue
        d = row_distance(rvals, r2, header, bounds)
        dists.append((d, rnum2, r2))
    dists.sort(key=lambda x: x[0])
    return dists[:k]

# --------------------------------------------------------------------------- #
# CAR-17: Flag unusual input combinations
# --------------------------------------------------------------------------- #

def car17(header, rows):
    bounds = numeric_bounds(header, rows)
    # Compute mean distance to k nearest neighbours for each row
    mean_dists = {}
    for row in rows:
        neigh = k_nearest(row, rows, header, bounds, k=5)
        if neigh:
            mean_dists[row[0]] = sum(d for d, _, _ in neigh) / len(neigh)
        else:
            mean_dists[row[0]] = float("inf")

    # Statistics over mean distances
    vals = list(mean_dists.values())
    mu = sum(vals) / len(vals)
    sigma = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))

    flagged = []
    scores = {}
    for rnum, dist in mean_dists.items():
        z = (dist - mu) / sigma if sigma else 0
        if z > 2:  # 2 standard deviations above mean
            flagged.append(rnum)
            scores[str(rnum)] = round(z, 4)

    flagged.sort()
    if len(flagged) > 40:
        flagged = flagged[:40]
        scores = {str(r): scores[str(r)] for r in flagged}

    # Print results
    print("== CAR-17 ==")
    if flagged:
        print("Unusual rows (input combinations):", ", ".join(map(str, flagged)))
        print("Unusualness scores:")
        for r in flagged:
            print(f"  Row {r}: score={scores[str(r)]}")
    else:
        print("No unusually input rows detected.")
    print(f"ANSWER CAR-17 {json.dumps({'flagged': flagged, 'scores': scores}, separators=(',', ':'))}")

# --------------------------------------------------------------------------- #
# CAR-18: Flag potential recording mistakes in Mpg+ or Lbs-
# --------------------------------------------------------------------------- #

def car18(header, rows):
    bounds = numeric_bounds(header, rows)
    # Identify goal columns of interest
    goal_cols = [c for c in header if c in ("Mpg+", "Lbs-")]
    goal_idxs = [header.index(c) for c in goal_cols]

    # Precompute neighbours for each row
    neigh_cache = {}
    for row in rows:
        neigh_cache[row[0]] = k_nearest(row, rows, header, bounds, k=5)

    # For each row and goal, compute expected as mean of neighbours' values
    flagged = []
    for row in rows:
        rnum, rvals = row
        for g_idx in goal_idxs:
            val = rvals[g_idx]
            if val is None:
                continue
            neigh_vals = []
            for _, _, nvals in neigh_cache[rnum]:
                nval = nvals[g_idx]
                if nval is not None:
                    neigh_vals.append(nval)
            if not neigh_vals:
                continue
            expected = sum(neigh_vals) / len(neigh_vals)
            diff = abs(val - expected)
            flagged.append({"row": rnum, "column": header[g_idx],
                            "actual": val, "expected": round(expected, 4),
                            "diff": diff})

    # Compute overall mean and std of diffs for each column
    col_diffs = defaultdict(list)
    for f in flagged:
        col_diffs[f["column"]].append(f["diff"])
    col_stats = {}
    for col, diffs in col_diffs.items():
        mu = sum(diffs) / len(diffs)
        sigma = math.sqrt(sum((d - mu) ** 2 for d in diffs) / len(diffs))
        col_stats[col] = (mu, sigma)

    # Filter by threshold
    flagged_final = []
    for f in flagged:
        mu, sigma = col_stats[f["column"]]
        if sigma == 0:
            continue
        z = (f["diff"] - mu) / sigma
        if z > 2:  # more than 2 std dev above mean
            flagged_final.append({"row": f["row"], "column": f["column"],
                                 "actual": f["actual"], "expected": f["expected"]})

    flagged_final.sort(key=lambda x: (x["row"], x["column"]))
    if len(flagged_final) > 40:
        flagged_final = flagged_final[:40]

    # Print results
    print("\n== CAR-18 ==")
    if flagged_final:
        print("Rows with suspected recording mistakes:")
        for f in flagged_final:
            print(f"  Row {f['row']} {f['column']}: actual={f['actual']} expected={f['expected']}")
    else:
        print("No significant discrepancies detected.")
    print(f"ANSWER CAR-18 {json.dumps({'flagged': flagged_final}, separators=(',', ':'))}")

# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv path>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    header, rows = read_csv(path)

    car17(header, rows)
    car18(header, rows)

if __name__ == "__main__":
    main()
