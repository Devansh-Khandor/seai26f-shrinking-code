#!/usr/bin/env python3
"""
skill_09.py <csv path>

Implements:
    NASA-17: Flag historical projects whose combinations of project characteristics
             are unusually different from the rest of our records so that we can
             review them.

    NASA-18: Identify project records whose reported effort or defect counts differ
             substantially from what comparable projects would lead us to expect, so
             we can investigate possible measurement or recording issues.

The script prints a header for each request, at least three human‑readable lines,
and then the machine‑checked JSON answer.
"""

import csv
import json
import math
import sys

# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------
def is_numeric(value):
    """Return True if value can be parsed as float (not missing)."""
    if value == '?' or value is None:
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False

def to_float(value):
    """Convert a string to float or return None for '?'."""
    return None if value == '?' else float(value)

# ------------------------------------------------------------------
# Encoding
# ------------------------------------------------------------------
def build_mappings(rows, cols):
    """Return dict col->value->int for categorical columns."""
    mappings = {}
    for col in cols:
        # Determine if numeric
        numeric = False
        for r in rows:
            v = r[col]
            if v == '?':
                continue
            try:
                float(v)
                numeric = True
                break
            except ValueError:
                continue
        if numeric:
            continue
        # categorical: build mapping
        mapping = {}
        idx = 0
        for r in rows:
            v = r[col]
            if v == '?':
                continue
            if v not in mapping:
                mapping[v] = idx
                idx += 1
        if mapping:
            mappings[col] = mapping
    return mappings

def encode_rows(rows, cols, mappings):
    """Return list of encoded rows (lists of floats or None)."""
    encoded = []
    for r in rows:
        out = []
        for col in cols:
            v = r[col]
            if v == '?':
                out.append(None)
                continue
            if col in mappings:
                out.append(mappings[col][v])
            else:
                out.append(float(v))
        encoded.append(out)
    return encoded

# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------
def compute_means_stds(encoded_rows):
    """Compute mean and std for each column, ignoring None."""
    n_cols = len(encoded_rows[0])
    sums = [0.0] * n_cols
    counts = [0] * n_cols
    for row in encoded_rows:
        for i, v in enumerate(row):
            if v is not None:
                sums[i] += v
                counts[i] += 1
    means = [s / c if c else 0.0 for s, c in zip(sums, counts)]
    # compute std
    sq_sums = [0.0] * n_cols
    for row in encoded_rows:
        for i, v in enumerate(row):
            if v is not None:
                diff = v - means[i]
                sq_sums[i] += diff * diff
    stds = [math.sqrt(sq_sums[i] / counts[i]) if counts[i] > 0 else 1.0
            for i in range(n_cols)]
    return means, stds

# ------------------------------------------------------------------
# Distance
# ------------------------------------------------------------------
def distance(row_a, row_b):
    """Euclidean distance over common non‑None columns."""
    sq = 0.0
    count = 0
    for a, b in zip(row_a, row_b):
        if a is None or b is None:
            continue
        diff = a - b
        sq += diff * diff
        count += 1
    if count == 0:
        return float('inf')
    return math.sqrt(sq)

# ------------------------------------------------------------------
# NASA-17
# ------------------------------------------------------------------
def nasa17(encoded_rows, row_nums, input_cols):
    means, stds = compute_means_stds(encoded_rows)
    scores = {}
    for idx, row in enumerate(encoded_rows):
        s = 0.0
        for val, mu, sigma in zip(row, means, stds):
            if val is None:
                continue
            if sigma == 0:
                z = 0.0
            else:
                z = (val - mu) / sigma
            s += z * z
        scores[row_nums[idx]] = s
    # Get top 10 by score descending
    flagged = sorted(scores.items(), key=lambda x: -x[1])[:10]
    flagged_nums = [f[0] for f in flagged]
    flagged_scores = {str(f[0]): f[1] for f in flagged}
    return flagged_nums, flagged_scores

# ------------------------------------------------------------------
# NASA-18
# ------------------------------------------------------------------
def nasa18(encoded_rows, rows_dicts, row_nums, goal_cols, input_cols):
    results = []
    for idx, (row, rdict) in enumerate(zip(encoded_rows, rows_dicts), start=1):
        for g in goal_cols:
            act = to_float(rdict.get(g, '?'))
            if act is None:
                continue
            # find k nearest neighbors with this goal available
            neighs = []
            for j, (other_row, other_dict) in enumerate(zip(encoded_rows, rows_dicts), start=1):
                if j == idx:
                    continue
                oth_act = to_float(other_dict.get(g, '?'))
                if oth_act is None:
                    continue
                d = distance(row, other_row)
                if math.isinf(d):
                    continue
                neighs.append((d, oth_act))
            if not neighs:
                continue
            neighs.sort(key=lambda x: x[0])
            k = min(3, len(neighs))
            neigh_vals = [v for _, v in neighs[:k]]
            expected = sum(neigh_vals) / len(neigh_vals)
            diff = abs(act - expected)
            results.append((diff, idx, g, act, expected))
    # Sort by diff descending, pick top 10
    results.sort(key=lambda x: -x[0])
    flagged = []
    for diff, idx, g, act, exp in results[:10]:
        flagged.append({
            "row": idx,
            "column": g,
            "actual": act,
            "expected": exp
        })
    return flagged

# ------------------------------------------------------------------
# Main logic
# ------------------------------------------------------------------
def main(csv_path):
    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    n_rows = len(rows)
    row_nums = list(range(1, n_rows + 1))

    # Identify columns
    goal_cols = [c for c in header if c.endswith('+') or c.endswith('-')]
    input_cols = [c for c in header if not c.endswith('X') and c not in goal_cols]

    # Build mappings and encode rows
    mappings = build_mappings(rows, input_cols)
    encoded_rows = encode_rows(rows, input_cols, mappings)

    # ------------------------------------------------------------------
    # NASA-17
    # ------------------------------------------------------------------
    flagged_nums, flagged_scores = nasa17(encoded_rows, row_nums, input_cols)

    print("== NASA-17 ==")
    print(f"Total rows: {n_rows}")
    print(f"Top unusual projects (row numbers): {flagged_nums}")
    print("Unusualness scores for flagged rows:")
    for rn in flagged_nums:
        print(f"  Row {rn}: score={flagged_scores[str(rn)]:.4f}")
    print(f"ANSWER NASA-17 {json.dumps({'flagged': flagged_nums, 'scores': flagged_scores})}")

    # ------------------------------------------------------------------
    # NASA-18
    # ------------------------------------------------------------------
    flagged_defects = nasa18(encoded_rows, rows, row_nums, goal_cols, input_cols)

    print("\n== NASA-18 ==")
    if flagged_defects:
        print(f"Top 10 rows with most surprising effort/defects:")
        for f in flagged_defects:
            print(f"  Row {f['row']} column {f['column']} actual={f['actual']:.2f} expected={f['expected']:.2f}")
    else:
        print("No surprising effort or defect records found.")
    print(f"ANSWER NASA-18 {json.dumps({'flagged': flagged_defects})}")

# ------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
