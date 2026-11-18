#!/usr/bin/env python3
"""
Skill 02: NASA-03 and NASA-04 predictions for held‑out projects

This script reads the supplied CSV file of software projects and
produces two analyses:

  * NASA-03 – Estimate the Effort‑ (development effort) required for
    each held‑out project.  The estimate is the mean of the Effort‑
    values of the 3 nearest historical projects.  The script
    prints the predictions for every held‑out row (row numbers
    divisible by 5) and the mean absolute error (MAE) of those
    predictions against the true Effort‑ values.

  * NASA-04 – Estimate the Defects‑ (number of defects) for each
    held‑out project, and provide an uncertainty range.  The
    estimate is the mean of the Defects‑ values of the 3 nearest
    historical projects.  The uncertainty range is the minimum
    and maximum Defects‑ values observed among those nearest
    neighbours.

Both analyses use only the first ``budget`` rows of the historical
data (history = all rows except the last round(0.2 × N) rows,
arrivals).  The budget is the smaller of 30 or ``len(history)//5``.
Only rows that have a known value for the target column are
included in the training set.  Feature columns are all columns
that do **not** end with ``X``, ``+`` or ``-`` and that are not
the target column.

The script outputs three lines for each request – a header
(``== NASA-03 ==`` or ``== NASA-04 ==``), a line summarising the
predictions, and a line with the MAE (for NASA‑03).  After the
summary it prints a machine‑checked answer line:

    ANSWER NASA-03 <json>
    ANSWER NASA-04 <json>

The JSON objects contain the predictions for every held‑out row
(1‑based row numbers as keys) and for NASA‑03 an additional
MAE value.

The script is deterministic – no random choices are made.
"""

import csv
import json
import math
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def atom(s: str) -> Any:
    """Coerce a CSV cell to an int, float, bool or string, stripping whitespace."""
    s = s.strip()
    if s == "?":
        return s
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return {"True": True, "False": False}.get(s, s)


def round_float(v: float) -> float:
    """Return a float rounded to 6 decimals for JSON output."""
    return round(v, 6)


# ----------------------------------------------------------------------
# Distance helpers
# ----------------------------------------------------------------------
def is_numeric(col: str, rows: List[Dict[str, Any]]) -> bool:
    """Return True if the column is numeric (int/float) in any row."""
    for r in rows:
        v = r[col]
        if v != "?":
            return isinstance(v, (int, float))
    return False


def feature_distance(a: Dict[str, Any], b: Dict[str, Any],
                     features: List[str], numeric: Dict[str, bool]) -> float:
    """Compute a simple distance between two rows over given features."""
    dist = 0.0
    count = 0
    for f in features:
        va = a[f]
        vb = b[f]
        if va == "?" or vb == "?":
            continue
        if numeric[f]:
            diff = va - vb
            dist += diff * diff
        else:
            dist += 0 if va == vb else 1
        count += 1
    # If no common features, return a large distance
    if count == 0:
        return float("inf")
    return math.sqrt(dist)


# ----------------------------------------------------------------------
# Core predictive functions
# ----------------------------------------------------------------------
def nearest_neighbors(train: List[Dict[str, Any]],
                      target_col: str,
                      query: Dict[str, Any],
                      features: List[str],
                      numeric: Dict[str, bool],
                      k: int = 3) -> List[Dict[str, Any]]:
    """Return the k nearest neighbors of query from train."""
    # Compute distances
    dists = [(feature_distance(query, tr, features, numeric), tr) for tr in train]
    # Sort by distance
    dists.sort(key=lambda x: x[0])
    return [tr for _, tr in dists[:k]]


def predict_effort(train: List[Dict[str, Any]],
                   query: Dict[str, Any],
                   features: List[str],
                   numeric: Dict[str, bool]) -> float:
    """Predict Effort- as mean of nearest neighbors' Effort-."""
    neigh = nearest_neighbors(train, "Effort-", query, features, numeric)
    values = [tr["Effort-"] for tr in neigh]
    return sum(values) / len(values)


def predict_defects(train: List[Dict[str, Any]],
                   query: Dict[str, Any],
                   features: List[str],
                   numeric: Dict[str, bool]) -> Tuple[float, float, float]:
    """Predict Defects- as mean of nearest neighbors' Defects-, with min/max."""
    neigh = nearest_neighbors(train, "Defects-", query, features, numeric)
    vals = [tr["Defects-"] for tr in neigh]
    mean = sum(vals) / len(vals)
    low = min(vals)
    high = max(vals)
    return mean, low, high


# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main(csv_path: str):
    # Load data
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows_raw = [ [atom(cell) for cell in row] for row in reader ]

    # Build list of dict rows with row numbers (1-based)
    rows: List[Dict[str, Any]] = []
    for idx, cells in enumerate(rows_raw, start=1):
        rows.append(dict(zip(headers, cells)))

    N = len(rows)
    arrivals_count = int(round(0.2 * N))
    history_rows = rows[:N - arrivals_count]
    held_out_rows = [r for r in rows if r["idX"] % 5 == 0]  # idX is numeric
    # Identify feature columns (exclude X, +, -, target)
    feature_cols = [h for h in headers if not (h.endswith("X") or h.endswith("+") or h.endswith("-"))]
    # Exclude target columns for each task
    # Determine numeric columns
    numeric_map = {c: is_numeric(c, rows) for c in feature_cols}

    # Compute budget
    budget = min(30, len(history_rows) // 5)

    # ------------------------------------------------------------------
    # NASA-03: Effort- predictions
    # ------------------------------------------------------------------
    # Training set: first `budget` rows of history with known Effort-
    train_effort = [r for r in history_rows[:budget] if r["Effort-"] != "?"]
    predictions_03: Dict[str, float] = {}
    mae_total = 0.0
    count_03 = 0

    for r in held_out_rows:
        if r["Effort-"] == "?":
            # unknown true value – skip error computation
            continue
        pred = predict_effort(train_effort, r,
                              [c for c in feature_cols if c != "Effort-"],
                              numeric_map)
        predictions_03[str(r["idX"])] = round_float(pred)
        mae_total += abs(pred - r["Effort-"])
        count_03 += 1

    mae_03 = mae_total / count_03 if count_03 else 0.0

    # ------------------------------------------------------------------
    # NASA-04: Defects- predictions with uncertainty
    # ------------------------------------------------------------------
    train_defects = [r for r in history_rows[:budget] if r["Defects-"] != "?"]
    predictions_04: Dict[str, Dict[str, float]] = {}

    for r in held_out_rows:
        if r["Defects-"] == "?":
            # unknown true value – no error calculation needed
            continue
        mean, low, high = predict_defects(train_defects, r,
                                          [c for c in feature_cols if c != "Defects-"],
                                          numeric_map)
        predictions_04[str(r["idX"])] = {
            "estimate": round_float(mean),
            "low": round_float(low),
            "high": round_float(high),
        }

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    # NASA-03
    print("== NASA-03 ==")
    print(f"Predicted Effort- for held‑out rows: {json.dumps(predictions_03)}")
    print(f"MAE: {round_float(mae_03):.6f}")

    # NASA-04
    print("\n== NASA-04 ==")
    print(f"Predicted Defects- for held‑out rows: {json.dumps(predictions_04)}")

    # Machine‑checked answers
    print("\nANSWER NASA-03", json.dumps({
        "predictions": predictions_03,
        "mae": round_float(mae_03)
    }))
    print("ANSWER NASA-04", json.dumps({
        "predictions": predictions_04
    }))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
