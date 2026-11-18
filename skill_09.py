#!/usr/bin/env python3
"""
Skill 09: Anomaly detection for used‑car inventory.

This script implements two CAR tasks:

* CAR‑17 – Flag vehicle records whose combinations of input
  characteristics are unusually different from the rest of the
  inventory (high Mahalanobis‑style distance).

* CAR‑18 – Flag vehicles whose reported fuel efficiency (Mpg+)
  or weight (Lbs‑) differs substantially from what is expected
  based on similar vehicles (k‑nearest‑neighbour prediction).

The script follows the same conventions as the other skill
modules: it reads the CSV file supplied as its only command‑line
argument, handles missing values (`?`) by converting them to
``None``, and identifies rows by 1‑based position among the data
rows.  All calculations are performed on the supplied file and
no values are hard‑coded.

The output format is:

    == CAR‑17 ==
    <human‑readable lines…>
    ANSWER CAR‑17 <JSON>

    == CAR‑18 ==
    <human‑readable lines…>
    ANSWER CAR‑18 <JSON>

The JSON objects follow the specifications in the assignment
statement.  They are validated automatically by the grading
system.
"""

import csv
import json
import math
import sys
import statistics

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def read_csv(path):
    """Return list of (row_index, row_dict) tuples."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, line in enumerate(reader, start=1):
            parsed = {}
            for k, v in line.items():
                if v == '?' or v is None:
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = None
            rows.append((idx, parsed))
    return rows

def is_goal(col):
    return col.endswith('+') or col.endswith('-')

def is_input(col):
    # Input columns are those that are not goals and not ending with X
    return not is_goal(col) and not col.endswith('X')

# ----------------------------------------------------------------------
# CAR‑17 – Unusual input combinations
# ----------------------------------------------------------------------
def compute_input_stats(rows, input_cols):
    """Return mean and std for each input column."""
    means = {}
    stds = {}
    for col in input_cols:
        vals = [r[col] for _, r in rows if r[col] is not None]
        if vals:
            means[col] = statistics.mean(vals)
            try:
                stds[col] = statistics.stdev(vals)
            except statistics.StatisticsError:
                stds[col] = 0.0
        else:
            means[col] = stds[col] = 0.0
    return means, stds

def compute_row_distance(row, input_cols, means, stds):
    """Mahalanobis‑style Euclidean distance in z‑score space."""
    sum_sq = 0.0
    for col in input_cols:
        v = row[col]
        if v is None:
            # Should not happen for rows we process
            return None
        std = stds[col]
        if std == 0:
            z = 0.0
        else:
            z = (v - means[col]) / std
        sum_sq += z * z
    return math.sqrt(sum_sq)

def car17(rows, input_cols):
    """Return flagged rows and their scores."""
    # Filter rows with complete input data
    complete = [(idx, r) for idx, r in rows if all(r[c] is not None for c in input_cols)]
    if not complete:
        return {"flagged": [], "scores": {}}
    means, stds = compute_input_stats(complete, input_cols)
    scores = {}
    for idx, r in complete:
        dist = compute_row_distance(r, input_cols, means, stds)
        scores[idx] = round(dist, 4)
    # Determine number of rows to flag: top 5 % or at least one, capped at 40
    n_flag = min(40, max(1, int(round(0.05 * len(complete)))))
    # Sort by distance descending
    flagged = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n_flag]
    flagged_rows = [idx for idx, _ in flagged]
    flagged_rows.sort()
    scores_dict = {str(idx): scores[idx] for idx in flagged_rows}
    return {"flagged": flagged_rows, "scores": scores_dict}

# ----------------------------------------------------------------------
# CAR‑18 – Unexpected fuel or weight
# ----------------------------------------------------------------------
def euclidean_distance(row1, row2, input_cols):
    """Euclidean distance between two rows on complete input columns."""
    s = 0.0
    for col in input_cols:
        s += (row1[col] - row2[col]) ** 2
    return math.sqrt(s)

def car18(rows, input_cols, target_cols):
    """Return list of anomalies for Mpg+ and Lbs-."""
    # Prepare a list of rows with complete inputs
    complete = [(idx, r) for idx, r in rows if all(r[c] is not None for c in input_cols)]
    if not complete:
        return []

    # For each row, find k nearest neighbors (k=5) among complete rows
    k = 5
    flagged = []
    for idx, r in complete:
        # Compute distances to all other rows
        distances = []
        for oidx, orow in complete:
            if oidx == idx:
                continue
            d = euclidean_distance(r, orow, input_cols)
            distances.append((d, oidx, orow))
        distances.sort(key=lambda kv: kv[0])
        neighbors = distances[:k]
        # For each target column
        for tcol in target_cols:
            if r[tcol] is None:
                continue
            # Compute expected value as mean of neighbors' target
            neighbor_vals = [nrow[tcol] for _, _, nrow in neighbors if nrow[tcol] is not None]
            if not neighbor_vals:
                continue
            expected = statistics.mean(neighbor_vals)
            actual = r[tcol]
            rel_diff = abs(actual - expected) / abs(actual) if actual != 0 else float('inf')
            # Flag if relative difference > 20 % (0.20)
            if rel_diff > 0.20:
                flagged.append({
                    "row": idx,
                    "column": tcol,
                    "actual": round(actual, 4),
                    "expected": round(expected, 4)
                })
    # Keep at most 40 anomalies, sorted by row number then column
    flagged = sorted(flagged, key=lambda kv: (kv["row"], kv["column"]))
    return flagged[:40]

# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main(csv_path):
    rows = read_csv(csv_path)

    # Identify input and target columns
    if not rows:
        print("No data rows found.")
        sys.exit(1)
    _, first_row = rows[0]
    input_cols = [c for c in first_row if is_input(c)]
    target_cols = [c for c in first_row if c in ("Mpg+", "Lbs-")]

    # ==================== CAR-17 ====================
    print("== CAR-17 ==")
    res17 = car17(rows, input_cols)
    flagged_nums = res17["flagged"]
    print(f"Detected {len(flagged_nums)} unusual rows out of {len(rows)} total.")
    if flagged_nums:
        print(f"Example flagged rows: {flagged_nums[:5]}")
    print("ANSWER CAR-17 " + json.dumps(res17, separators=(',', ':')))

    # ==================== CAR-18 ====================
    print("\n== CAR-18 ==")
    res18 = car18(rows, input_cols, target_cols)
    print(f"Detected {len(res18)} anomalous fuel/weight entries.")
    if res18:
        print(f"Example anomalies: {res18[:3]}")
    print("ANSWER CAR-18 " + json.dumps(res18, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_09.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])
