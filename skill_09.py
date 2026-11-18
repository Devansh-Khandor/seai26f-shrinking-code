#!/usr/bin/env python3
"""
skill_09.py – NASA-17 & NASA-18

NASA‑17: Flag historical projects whose combinations of project characteristics
are unusually different from the rest of the records.

NASA‑18: Identify project records whose reported effort or defect counts
differ substantially from what comparable projects would lead us to expect.

The program prints, for each request, a header line, at least three lines of
human‑readable output, and a machine‑checked JSON line with the answer.
"""

import csv
import json
import math
import sys
from collections import defaultdict

# ----------------------------------------------------------------------
# Utilities for reading and normalising data
# ----------------------------------------------------------------------
def read_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def to_float(val):
    """Convert a string to float, or return None for missing/invalid."""
    if val is None:
        return None
    val = val.strip()
    if val == "" or val == "?":
        return None
    try:
        return float(val)
    except ValueError:
        return None

def is_goal(col):
    return col.endswith(("+", "-"))

def is_x(col):
    return col.endswith("X")

def input_columns(header):
    """Return list of column names that are neither goals nor X."""
    return [c for c in header if not is_goal(c) and not is_x(c)]

# ----------------------------------------------------------------------
# Normalisation and encoding of input columns
# ----------------------------------------------------------------------
def build_feature_maps(rows, cols):
    """
    Build maps for each column:
      * numeric: list of float values
      * categorical: map from category string to integer code
    Returns:
      - col_type: dict col -> 'numeric' or 'categorical'
      - cat_maps: dict col -> mapping dict (for categorical)
    """
    col_type = {}
    cat_maps = {}
    for col in cols:
        numeric_vals = []
        cat_vals = []
        for r in rows:
            v = r.get(col)
            if v is None:
                continue
            v_str = v.strip()
            if v_str == "?":
                continue
            f = to_float(v_str)
            if f is not None:
                numeric_vals.append(f)
            else:
                cat_vals.append(v_str)
        if cat_vals:
            # encode categories
            mapping = {}
            code = 0
            for cat in cat_vals:
                if cat not in mapping:
                    mapping[cat] = code
                    code += 1
            col_type[col] = "categorical"
            cat_maps[col] = mapping
        else:
            col_type[col] = "numeric"
            cat_maps[col] = None
    return col_type, cat_maps

def encode_row(row, cols, col_type, cat_maps):
    """
    Return a dict col -> encoded value (float) or None if missing.
    Numeric values stay as float; categorical values become int codes.
    """
    encoded = {}
    for col in cols:
        v = row.get(col)
        if v is None:
            encoded[col] = None
            continue
        v_str = v.strip()
        if v_str == "?":
            encoded[col] = None
            continue
        if col_type[col] == "numeric":
            f = to_float(v_str)
            encoded[col] = f
        else:
            # categorical
            mapping = cat_maps[col]
            encoded[col] = mapping.get(v_str, None)
    return encoded

def compute_min_max_for_features(rows, cols, col_type, cat_maps):
    """
    For each column, compute min and max of encoded values over all rows.
    Returns dict col -> (min, max)
    """
    mins = {}
    maxs = {}
    for col in cols:
        vals = [v for v in [row[col] for row in rows] if v is not None]
        if not vals:
            mins[col] = maxs[col] = 0.0
        else:
            mins[col] = min(vals)
            maxs[col] = max(vals)
    return mins, maxs

# ----------------------------------------------------------------------
# Distance and centroid calculations
# ----------------------------------------------------------------------
def normalize_value(v, mn, mx):
    if mn == mx:
        return 0.0
    return (v - mn) / (mx - mn)

def compute_centroid(rows, cols, mins, maxs):
    """
    Compute centroid (mean of normalized values) for each feature.
    Returns dict col -> centroid (float)
    """
    sums = defaultdict(float)
    counts = defaultdict(int)
    for r in rows:
        for col in cols:
            v = r.get(col)
            if v is None:
                continue
            v_str = v.strip()
            if v_str == "?":
                continue
            # numeric value
            f = to_float(v_str)
            if f is None:
                continue
            norm = normalize_value(f, mins[col], maxs[col])
            sums[col] += norm
            counts[col] += 1
    centroid = {}
    for col in cols:
        if counts[col] > 0:
            centroid[col] = sums[col] / counts[col]
        else:
            centroid[col] = 0.0
    return centroid

def euclidean_distance(row_vec, other_vec, cols):
    """Compute Euclidean distance using only columns where both have values."""
    sq = 0.0
    valid = 0
    for col in cols:
        v1 = row_vec.get(col)
        v2 = other_vec.get(col)
        if v1 is None or v2 is None:
            continue
        sq += (v1 - v2) ** 2
        valid += 1
    if valid == 0:
        # No overlap, treat as infinite distance
        return float("inf")
    return math.sqrt(sq)

# ----------------------------------------------------------------------
# NASA-17: flag unusual projects
# ----------------------------------------------------------------------
def nasa17(rows, cols, mins, maxs, centroid):
    """
    Compute unusualness score for each row as Euclidean distance from centroid.
    Return list of up to 10 flagged row numbers and a dict of scores.
    """
    scored = []
    for idx, r in enumerate(rows, start=1):
        vec = {}
        for col in cols:
            v = r.get(col)
            if v is None:
                vec[col] = None
                continue
            v_str = v.strip()
            if v_str == "?":
                vec[col] = None
                continue
            f = to_float(v_str)
            if f is None:
                vec[col] = None
                continue
            norm = normalize_value(f, mins[col], maxs[col])
            vec[col] = norm
        # compute distance to centroid using available features
        sq = 0.0
        valid = 0
        for col in cols:
            val = vec.get(col)
            if val is None:
                continue
            cent = centroid[col]
            sq += (val - cent) ** 2
            valid += 1
        if valid == 0:
            dist = 0.0
        else:
            dist = math.sqrt(sq)
        scored.append((idx, dist))
    scored.sort(key=lambda x: x[1], reverse=True)
    flagged = [i for i, _ in scored[:10]]
    scores = {str(i): round(v, 4) for i, v in scored[:10]}
    return flagged, scores

# ----------------------------------------------------------------------
# NASA-18: flag anomalous effort/defects
# ----------------------------------------------------------------------
def nasa18(rows, cols, col_type, cat_maps, mins, maxs):
    """
    For each row, find nearest neighbours (based on input columns) and
    compare actual Effort- and Defects- with neighbor means.
    Flag if abs(actual - mean) > 2 * std.
    Return up to 10 flagged entries.
    """
    # Build encoded vectors for all rows
    encoded_rows = [encode_row(r, cols, col_type, cat_maps) for r in rows]
    flagged_entries = []
    # Pre‑compute list of indices for quick access
    nrows = len(rows)
    for idx, r in enumerate(rows, start=1):
        # Skip rows with missing Effort- or Defects-
        eff_actual = to_float(r.get("Effort-"))
        def_actual = to_float(r.get("Defects-"))
        # Skip if both missing
        if eff_actual is None and def_actual is None:
            continue

        # Compute distances to other rows
        distances = []
        vec = encoded_rows[idx-1]
        for j, other_vec in enumerate(encoded_rows):
            if j == idx-1:
                continue
            d = euclidean_distance(vec, other_vec, cols)
            distances.append((d, j))
        # Sort by distance
        distances.sort(key=lambda x: x[0])
        # Take up to K nearest neighbors that have at least one of the target values
        K = 5
        neigh_idxs = []
        for d, j in distances:
            if len(neigh_idxs) >= K:
                break
            neigh_idxs.append(j)
        # Gather neighbor values for Effort- and Defects-
        neigh_eff = [to_float(rows[j].get("Effort-")) for j in neigh_idxs]
        neigh_eff = [v for v in neigh_eff if v is not None]
        neigh_def = [to_float(rows[j].get("Defects-")) for j in neigh_idxs]
        neigh_def = [v for v in neigh_def if v is not None]

        def check_and_flag(col_name, actual, neigh_vals):
            if actual is None or not neigh_vals:
                return
            mean = sum(neigh_vals) / len(neigh_vals)
            if len(neigh_vals) > 1:
                std = math.sqrt(sum((x - mean) ** 2 for x in neigh_vals) / len(neigh_vals))
            else:
                std = 0.0
            if std == 0:
                return
            diff = abs(actual - mean)
            if diff > 2 * std:
                flagged_entries.append({
                    "row": idx,
                    "column": col_name,
                    "actual": actual,
                    "expected": round(mean, 4)
                })

        check_and_flag("Effort-", eff_actual, neigh_eff)
        check_and_flag("Defects-", def_actual, neigh_def)

    # Keep up to 10 entries, sorted by absolute deviation (descending)
    flagged_entries.sort(key=lambda x: abs(x["actual"] - x["expected"]), reverse=True)
    return flagged_entries[:10]

# ----------------------------------------------------------------------
# Main orchestration
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_csv(csv_path)
    if not rows:
        print("No data rows found.")
        sys.exit(1)

    header = list(rows[0].keys())
    cols = input_columns(header)

    # Build feature maps and encoded data
    col_type, cat_maps = build_feature_maps(rows, cols)
    encoded_rows = [encode_row(r, cols, col_type, cat_maps) for r in rows]

    # Compute min/max for normalization
    mins, maxs = compute_min_max_for_features(encoded_rows, cols, col_type, cat_maps)

    # NASA-17
    print("== NASA-17 ==")
    centroid = compute_centroid(encoded_rows, cols, mins, maxs)
    flagged17, scores17 = nasa17(rows, cols, mins, maxs, centroid)
    if flagged17:
        print(f"Flagged rows: {flagged17}")
        print("Unusualness scores (row:score):")
        for row in flagged17:
            print(f"  Row {row}: {scores17[str(row)]}")
    else:
        print("No unusual rows found.")
    print("ANSWER NASA-17", json.dumps({"flagged": flagged17, "scores": scores17}, separators=(',', ':')))

    # NASA-18
    print("\n== NASA-18 ==")
    flagged18 = nasa18(rows, cols, col_type, cat_maps, mins, maxs)
    if flagged18:
        print("Flagged anomalous records:")
        for e in flagged18:
            print(f"  Row {e['row']} – {e['column']}: actual={e['actual']}, expected={e['expected']}")
    else:
        print("No anomalous records found.")
    print("ANSWER NASA-18", json.dumps({"flagged": flagged18}, separators=(',', ':')))

if __name__ == "__main__":
    main()
