#!/usr/bin/env python3
"""
Skill 04: NASA-07 and NASA-08

This script implements two NASA requests:

  * NASA-07 – For each new proposal (arrival) predict Effort- and Defects- using
    nearest‑neighbor regression based on historical data and previously seen
    arrivals.  An arrival is flagged as an alert when either prediction is
    greater than two standard deviations above the mean of that goal
    in the current training set.

  * NASA-08 – After every checkpoint (0, 9 and all arrivals), produce a
    summary of the goal columns (n, mean, min, max) and the 5 best rows
    seen so far.  “Best” means large Kloc+, small Effort-, small Defects-.
"""

import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def atom(s: str) -> Any:
    """Coerce a CSV cell to a Python value."""
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
    """Round to 6 decimal places for JSON output."""
    return round(v, 6)


# ----------------------------------------------------------------------
# Distance helpers (same as skill_02)
# ----------------------------------------------------------------------
def is_numeric(col: str, rows: List[List[Any]]) -> bool:
    """Return True if the column is numeric in any row."""
    for r in rows:
        v = r[col]
        if v != "?":
            return isinstance(v, (int, float))
    return False


def feature_distance(a: List[Any], b: List[Any], features: List[int],
                     numeric: Dict[int, bool]) -> float:
    """Compute a simple distance between two rows over given features."""
    dist = 0.0
    count = 0
    for fi in features:
        va = a[fi]
        vb = b[fi]
        if va == "?" or vb == "?":
            continue
        if numeric[fi]:
            diff = va - vb
            dist += diff * diff
        else:
            dist += 0 if va == vb else 1
        count += 1
    if count == 0:
        return float("inf")
    return math.sqrt(dist)


# ----------------------------------------------------------------------
# Prediction helpers
# ----------------------------------------------------------------------
def nearest_neighbors(train: List[List[Any]], target_at: int, query: List[Any],
                      features: List[int], numeric: Dict[int, bool],
                      k: int = 3) -> List[List[Any]]:
    dists = [(feature_distance(query, tr, features, numeric), tr) for tr in train]
    dists.sort(key=lambda x: x[0])
    return [tr for _, tr in dists[:k]]


def predict_value(train: List[List[Any]], target_at: int, query: List[Any],
                  features: List[int], numeric: Dict[int, bool]) -> float:
    neigh = nearest_neighbors(train, target_at, query, features, numeric, k=3)
    # If no neighbors, fall back to mean of target in train
    if not neigh:
        vals = [row[target_at] for row in train if row[target_at] != "?"]
        if not vals:
            return 0.0
        return sum(vals) / len(vals)
    vals = [tr[target_at] for tr in neigh if tr[target_at] != "?"]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def stats_of_values(vals: List[float]) -> Tuple[float, float]:
    """Return mean and std dev (population) of vals."""
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / n
    return mean, math.sqrt(var)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        raw_rows = [row for row in reader]

    # Convert to Python values and keep row indices (1‑based)
    rows: List[List[Any]] = []
    for row in raw_rows:
        rows.append([atom(v) for v in row])

    total_rows = len(rows)
    # Determine arrivals
    arrivals_count = int(round(0.2 * total_rows))
    history_rows = rows[:-arrivals_count]
    arrivals_rows = rows[-arrivals_count:]

    # Map from row index to data row
    idx_to_row = {i + 1: rows[i] for i in range(total_rows)}

    # Identify goal columns (+/-)
    goal_cols = [name for name in headers if name.endswith("+") or name.endswith("-")]
    goal_at = {name: headers.index(name) for name in goal_cols}

    # Determine numeric columns
    numeric: Dict[int, bool] = {}
    for i, col_name in enumerate(headers):
        numeric[i] = is_numeric(col_name, rows)

    # Features are all columns except X, +, - and target columns
    def is_feature(col_name: str, target: str | None = None) -> bool:
        if col_name.endswith("X") or col_name.endswith("+") or col_name.endswith("-"):
            return False
        if target and col_name == target:
            return False
        return True

    # Pre‑compute indices of feature columns
    feature_at = {name: headers.index(name) for name in headers
                  if is_feature(name)}

    # Pre‑compute indices of numeric features
    numeric_feature_at = {i: numeric[i] for i in feature_at.values()}

    # ------------------------------------------------------------------
    # NASA-07
    # ------------------------------------------------------------------
    training_set: List[List[Any]] = history_rows.copy()
    assessments: List[Dict[str, Any]] = []

    # Pre‑compute training target values for thresholds
    def training_stats(target_name: str) -> Tuple[float, float]:
        at = goal_at[target_name]
        vals = [row[at] for row in training_set if row[at] != "?"]
        return stats_of_values(vals)

    for arrival_row in arrivals_rows:
        # Predict
        eff_at = goal_at["Effort-"]
        def_at = goal_at["Defects-"]
        pred_eff = predict_value(training_set, eff_at, arrival_row,
                                 list(feature_at.values()), numeric_feature_at)
        pred_def = predict_value(training_set, def_at, arrival_row,
                                 list(feature_at.values()), numeric_feature_at)

        # Compute thresholds from current training set
        eff_mean, eff_sd = training_stats("Effort-")
        def_mean, def_sd = training_stats("Defects-")
        eff_thr = eff_mean + 2 * eff_sd if eff_sd > 0 else eff_mean
        def_thr = def_mean + 2 * def_sd if def_sd > 0 else def_mean

        alert = (pred_eff > eff_thr) or (pred_def > def_thr)

        # Find row number
        row_no = next(i for i, r in idx_to_row.items() if r is arrival_row)

        assessments.append({
            "row": row_no,
            "predicted_effort": round_float(pred_eff),
            "predicted_defects": round_float(pred_def),
            "alert": alert
        })

        # Add this row to training set for subsequent predictions
        training_set.append(arrival_row)

    # ------------------------------------------------------------------
    # NASA-08
    # ------------------------------------------------------------------
    checkpoints: List[Dict[str, Any]] = []

    # Helper to compute summary for a set of rows
    def summary_of(rows_subset: List[List[Any]]) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        for g_name in goal_cols:
            at = goal_at[g_name]
            vals = [row[at] for row in rows_subset if row[at] != "?"]
            n = len(vals)
            if n == 0:
                mean_val = min_val = max_val = 0.0
            else:
                mean_val = sum(vals) / n
                min_val = min(vals)
                max_val = max(vals)
            res[g_name] = {
                "n": n,
                "mean": round_float(mean_val),
                "min": round_float(min_val),
                "max": round_float(max_val)
            }
        return res

    # Helper to produce shortlist of top 5 rows
    def shortlist_of(rows_subset: List[List[Any]]) -> List[int]:
        # Only rows with all three goal values known
        valid = [r for r in rows_subset if all(r[goal_at[g]] not in ("?",) for g in ["Kloc+", "Effort-", "Defects-"])]
        # Sort by (Kloc+ desc, Effort- asc, Defects- asc)
        valid.sort(key=lambda r: (
            -r[goal_at["Kloc+"]],
            r[goal_at["Effort-"]],
            r[goal_at["Defects-"]]
        ))
        return [next(i for i, rr in idx_to_row.items() if rr is r) for r in valid[:5]]

    # Checkpoints after 0, 9, and all arrivals
    for m in [0, 9, arrivals_count]:
        rows_up_to = history_rows + arrivals_rows[:m]
        checkpoints.append({
            "after": m,
            "summary": summary_of(rows_up_to),
            "shortlist": shortlist_of(rows_up_to)
        })

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    # NASA-07
    print("== NASA-07 ==")
    print(f"Total arrivals: {arrivals_count}")
    print("Assessments:")
    print(json.dumps({"assessments": assessments}, separators=(',', ': ')))

    # NASA-08
    print("\n== NASA-08 ==")
    print("Checkpoints:")
    print(json.dumps({"checkpoints": checkpoints}, separators=(',', ': ')))

    # Machine‑checked answers
    print(f"\nANSWER NASA-07 {json.dumps({'assessments': assessments}, separators=(',', ': '))}")
    print(f"ANSWER NASA-08 {json.dumps({'checkpoints': checkpoints}, separators=(',', ': '))}")


if __name__ == "__main__":
    main()
