#!/usr/bin/env python3
"""
skill_09.py <csv_path>

Implements NASA-17 and NASA-18.

NASA-17: Flag historical projects whose combinations of project characteristics
are unusually different from the rest of the records.
NASA-18: Flag projects whose reported Effort- or Defects- values differ substantially
from what comparable projects would lead us to expect.
"""

import sys
import csv
import json
import math
from statistics import mean, stdev

# --------------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------------
def parse_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def to_float(val):
    """Convert a cell to float, or None for missing."""
    if val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def is_numeric(col, rows):
    """True if at least one row can be parsed as float for this column."""
    for r in rows:
        if to_float(r[col]) is not None:
            return True
    return False

# --------------------------------------------------------------------
# Input (non‑goal) columns
# --------------------------------------------------------------------
def get_input_columns(rows):
    """Return list of columns that are not goals (+ or -) and not X."""
    header = rows[0].keys()
    return [c for c in header if not c.endswith('+') and not c.endswith('-') and not c.endswith('X')]

# --------------------------------------------------------------------
# Normalisation of numeric columns
# --------------------------------------------------------------------
def compute_norm_stats(rows, numeric_cols):
    """Compute mean and std for each numeric column."""
    stats = {}
    for col in numeric_cols:
        vals = [to_float(r[col]) for r in rows if to_float(r[col]) is not None]
        if vals:
            m = mean(vals)
            if len(vals) > 1:
                s = stdev(vals)
            else:
                s = 0.0
        else:
            m = 0.0
            s = 0.0
        stats[col] = (m, s)
    return stats

def get_norm_value(row, col, stats):
    """Return normalised value for a numeric column."""
    v = to_float(row[col])
    if v is None:
        return None
    m, s = stats[col]
    if s == 0:
        return 0.0
    return (v - m) / s

# --------------------------------------------------------------------
# Distance between two rows using normalised numeric columns
# --------------------------------------------------------------------
def euclidean_distance_norm(r1, r2, numeric_cols, stats):
    sq = 0.0
    common = 0
    for col in numeric_cols:
        v1 = get_norm_value(r1, col, stats)
        v2 = get_norm_value(r2, col, stats)
        if v1 is None or v2 is None:
            continue
        diff = v1 - v2
        sq += diff * diff
        common += 1
    if common == 0:
        return float('inf')
    return math.sqrt(sq)

# --------------------------------------------------------------------
# NASA-17: Unusual characteristic combinations
# --------------------------------------------------------------------
def nasa_17(rows):
    input_cols = get_input_columns(rows)
    numeric_cols = [c for c in input_cols if is_numeric(c, rows)]

    norm_stats = compute_norm_stats(rows, numeric_cols)

    n = len(rows)
    mean_dist = []
    for i, r in enumerate(rows):
        total = 0.0
        count = 0
        for j, q in enumerate(rows):
            if i == j:
                continue
            d = euclidean_distance_norm(r, q, numeric_cols, norm_stats)
            if math.isfinite(d):
                total += d
                count += 1
        avg = total / count if count else 0.0
        mean_dist.append((i+1, avg))

    # Sort by decreasing mean distance; pick up to 10
    mean_dist.sort(key=lambda x: -x[1])
    flagged = [row for row, _ in mean_dist[:10]]
    scores = {str(row): round(score, 6) for row, score in mean_dist[:10]}

    # Print results
    print("== NASA-17 ==")
    print(f"Number of projects: {n}")
    print(f"Flagged rows (up to 10): {flagged}")
    print(f"Scores: {scores}")

    print("ANSWER NASA-17", json.dumps({"flagged": flagged, "scores": scores}, separators=(',', ':')))

# --------------------------------------------------------------------
# NASA-18: Outlier effort/defects based on comparable projects
# --------------------------------------------------------------------
def nasa_18(rows):
    input_cols = get_input_columns(rows)
    numeric_cols = [c for c in input_cols if is_numeric(c, rows)]
    norm_stats = compute_norm_stats(rows, numeric_cols)

    n = len(rows)
    deviations = []  # list of dicts

    for i, r in enumerate(rows):
        # Compute distances to all other rows
        distances = []
        for j, q in enumerate(rows):
            if i == j:
                continue
            d = euclidean_distance_norm(r, q, numeric_cols, norm_stats)
            if math.isfinite(d):
                distances.append((d, j))
        # If fewer than 5 neighbours, use all
        k = min(5, len(distances))
        nearest = sorted(distances, key=lambda x: x[0])[:k]

        # Collect neighbor values for Effort- and Defects-
        for col in ["Effort-", "Defects-"]:
            actual = to_float(r[col])
            if actual is None:
                continue
            neigh_vals = [to_float(rows[j][col]) for _, j in nearest if to_float(rows[j][col]) is not None]
            if not neigh_vals:
                continue
            expected = mean(neigh_vals)
            dev = abs(actual - expected)
            deviations.append({
                "row": i+1,
                "column": col,
                "actual": round(actual, 4),
                "expected": round(expected, 4),
                "deviation": dev
            })

    # Sort by deviation descending and pick top 10
    deviations.sort(key=lambda x: -x["deviation"])
    flagged = []
    seen = set()
    for d in deviations:
        key = (d["row"], d["column"])
        if key in seen:
            continue
        seen.add(key)
        flagged.append({"row": d["row"], "column": d["column"],
                        "actual": d["actual"], "expected": d["expected"]})
        if len(flagged) >= 10:
            break

    # Print results
    print("== NASA-18 ==")
    print(f"Number of projects: {n}")
    print(f"Flagged rows (up to 10): {flagged}")

    # Prepare JSON
    json_flagged = [{"row": f["row"], "column": f["column"],
                     "actual": f["actual"], "expected": f["expected"]}
                    for f in flagged]
    print("ANSWER NASA-18", json.dumps({"flagged": json_flagged}, separators=(',', ':')))

# --------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_09.py <csv_path>\n")
        sys.exit(1)
    path = sys.argv[1]
    rows = parse_csv(path)
    nasa_17(rows)
    print()
    nasa_18(rows)

if __name__ == "__main__":
    main()
