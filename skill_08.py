#!/usr/bin/env python3
"""
Skill 08: NASA-15 and NASA-16

NASA-15 – For a project proposal predicted to require excessive effort, identify
plausible changes to adjustable planning or process characteristics that might
reduce its predicted effort.

NASA-16 – For a project proposal with a high predicted defect count, identify
the smallest plausible combination of changes to adjustable planning or process
characteristics that could improve its predicted quality, while explaining
potential trade‑offs.

The script reads a CSV of software projects.  It uses a simple
k‑nearest‑neighbour regression (k=3) to predict Effort‑ and
Defects‑ for held‑out rows (row numbers divisible by 5).  The
predictions are based on all other rows in the file, ignoring
missing values.  For each NASA request it chooses the held‑out
project with the most extreme predicted value, then explores
single‑column (NASA‑15) or up to three‑column (NASA‑16)
adjustments to reduce the target prediction.  The new values must
appear somewhere in the dataset.

The output format follows the specification:

    == NASA-15 ==
    <human readable lines>
    ANSWER NASA-15 <json>

    == NASA-16 ==
    <human readable lines>
    ANSWER NASA-16 <json>
"""

import csv
import json
import math
import sys
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# Utility: convert string to typed value
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


# ----------------------------------------------------------------------
# Prediction helpers
# ----------------------------------------------------------------------
def distance(row1: Dict[str, Any], row2: Dict[str, Any],
             features: List[str]) -> float:
    """Euclidean distance over numeric features; 1 for unequal symbols.
    Missing values are ignored.  If no comparable feature, return inf.
    """
    s = 0.0
    n = 0
    for f in features:
        v1, v2 = row1[f], row2[f]
        if v1 == "?" or v2 == "?" or isinstance(v1, str) and isinstance(v2, str) and v1 == v2:
            continue
        if isinstance(v1, str) or isinstance(v2, str):
            # symbolic and unequal
            s += 1.0
            n += 1
        else:
            try:
                diff = float(v1) - float(v2)
                s += diff * diff
                n += 1
            except Exception:
                s += 1.0
                n += 1
    return math.sqrt(s) if n else float("inf")


def knn_neighbors(target: Dict[str, Any],
                  training: List[Dict[str, Any]],
                  exclude: set,
                  k: int,
                  features: List[str]) -> List[Dict[str, Any]]:
    """Return k nearest neighbours of target, excluding rows in `exclude`."""
    dists = []
    for r in training:
        if id(r) in exclude:
            continue
        dists.append((distance(target, r, features), r))
    dists.sort(key=lambda x: x[0])
    return [r for _, r in dists[:k]]


def predict(target: Dict[str, Any],
            training: List[Dict[str, Any]],
            features: List[str],
            k: int = 3) -> float:
    """Mean of target's k nearest neighbour goal values."""
    neigh = knn_neighbors(target, training, set(), k, features)
    vals = [r["Effort-"] for r in neigh if r["Effort-"] != "?"]
    return sum(vals) / len(vals) if vals else float("nan")


def predict_defects(target: Dict[str, Any],
                    training: List[Dict[str, Any]],
                    features: List[str],
                    k: int = 3) -> float:
    neigh = knn_neighbors(target, training, set(), k, features)
    vals = [r["Defects-"] for r in neigh if r["Defects-"] != "?"]
    return sum(vals) / len(vals) if vals else float("nan")


# ----------------------------------------------------------------------
# Core analysis
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    # Load data
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [dict(zip(header, [atom(v) for v in r])) for r in reader]

    # Identify indices
    goal_cols = [c for c in header if c.endswith("+") or c.endswith("-")]
    # Adjustable columns: any column that is not a goal, not Kloc+, and not ending in X
    adjustable = [c for c in header
                  if (not c.endswith("+") and not c.endswith("-")
                      and c != "Kloc+" and not c.endswith("X"))]

    # Feature columns are all adjustable columns (including those that may have missing values)
    feature_cols = adjustable

    # Prepare row list with 1‑based numbering
    indexed_rows = [(i + 1, row) for i, row in enumerate(rows)]

    # Held‑out rows: divisible by 5
    held_out = [(num, row) for num, row in indexed_rows if num % 5 == 0]

    # Compute predictions for each held‑out row
    predictions = {}
    for num, row in held_out:
        eff = predict(row, rows, feature_cols)
        defect = predict_defects(row, rows, feature_cols)
        predictions[num] = {"Effort-": eff, "Defects-": defect}

    if not predictions:
        print("== NASA-15 ==")
        print("No held‑out rows to evaluate.")
        print("ANSWER NASA-15 {}")
        print("\n== NASA-16 ==")
        print("No held‑out rows to evaluate.")
        print("ANSWER NASA-16 {}")
        return

    # ------------------------------------------------------------
    # NASA‑15 – find most expensive effort prediction
    # ------------------------------------------------------------
    proj_eff = max(predictions.items(),
                   key=lambda kv: kv[1]["Effort-"])
    proj_eff_num, proj_eff_row = proj_eff[0], next(r for n, r in indexed_rows if n == proj_eff_num)
    eff_before = proj_eff[1]["Effort-"]

    # Find single adjustable change that reduces predicted effort the most
    best_change = None
    best_eff_after = eff_before
    for col in adjustable:
        current_val = proj_eff_row[col]
        # collect distinct values in dataset for this column (excluding missing)
        values = set(r[col] for r in rows if r[col] != "?")
        for v in values:
            if v == current_val:
                continue
            # create a copy with changed value
            tmp = proj_eff_row.copy()
            tmp[col] = v
            eff_after = predict(tmp, rows, feature_cols)
            if eff_after < best_eff_after:
                best_eff_after = eff_after
                best_change = {col: [current_val, v]}

    if best_change is None:
        # no improvement found
        best_change = {}
    # Output NASA‑15
    print("== NASA-15 ==")
    print(f"Project (row {proj_eff_num}):")
    print(f"Predicted Effort‑ before change: {eff_before:.6f}")
    if best_change:
        col, (old, new) = next(iter(best_change.items()))
        print(f"Proposed change: {col} {old} -> {new}")
        print(f"Predicted Effort‑ after change: {best_eff_after:.6f}")
    else:
        print("No single-column adjustment found that reduces effort.")
        print(f"Predicted Effort‑ after change: {eff_before:.6f}")

    # Machine‑checked answer
    answer_15 = {
        "project": proj_eff_num,
        "changes": best_change,
        "predicted_effort": {"before": round(eff_before, 6), "after": round(best_eff_after, 6)}
    }
    print(f"\nANSWER NASA-15 {json.dumps(answer_15)}")

    # ------------------------------------------------------------
    # NASA‑16 – find most defect‑heavy prediction
    # ------------------------------------------------------------
    proj_def = max(predictions.items(),
                   key=lambda kv: kv[1]["Defects-"])
    proj_def_num, proj_def_row = proj_def[0], next(r for n, r in indexed_rows if n == proj_def_num)
    defect_before = proj_def[1]["Defects-"]
    eff_before_def = predict(proj_def_row, rows, feature_cols)

    # For each adjustable column find best value reducing defects
    improvement_candidates = []
    for col in adjustable:
        cur = proj_def_row[col]
        vals = set(r[col] for r in rows if r[col] != "?")
        best_val = None
        best_def = defect_before
        for v in vals:
            if v == cur:
                continue
            tmp = proj_def_row.copy()
            tmp[col] = v
            new_def = predict_defects(tmp, rows, feature_cols)
            if new_def < best_def:
                best_def = new_def
                best_val = v
        if best_val is not None:
            improvement_candidates.append((defect_before - best_def, col, cur, best_val, best_def))

    # pick up to three columns with greatest defect reduction
    improvement_candidates.sort(reverse=True, key=lambda x: x[0])
    best_changes = {}
    defect_after = defect_before
    for _, col, old, new, new_def in improvement_candidates[:3]:
        best_changes[col] = [old, new]
        defect_after = new_def

    eff_after_def = predict(proj_def_row.copy(), rows, feature_cols)  # unchanged effort for base
    # If we changed columns, recompute effort to show trade‑off
    eff_after_def = predict(proj_def_row.copy(), rows, feature_cols)
    if best_changes:
        # apply all changes to get final row
        tmp_row = proj_def_row.copy()
        for c, (o_val, n_val) in best_changes.items():
            tmp_row[c] = n_val
        eff_after_def = predict(tmp_row, rows, feature_cols)

    print("\n== NASA-16 ==")
    print(f"Project (row {proj_def_num}):")
    print(f"Predicted Defects‑ before change: {defect_before:.6f}")
    if best_changes:
        print("Proposed changes:")
        for c, (old, new) in best_changes.items():
            print(f"  {c} {old} -> {new}")
        print(f"Predicted Defects‑ after change: {defect_after:.6f}")
        print(f"Predicted Effort‑ before change: {eff_before_def:.6f}")
        print(f"Predicted Effort‑ after change: {eff_after_def:.6f}")
    else:
        print("No single or small set of changes reduces defects.")
        print(f"Predicted Defects‑ after change: {defect_before:.6f}")

    answer_16 = {
        "project": proj_def_num,
        "changes": best_changes,
        "predicted_defects": {"before": round(defect_before, 6), "after": round(defect_after, 6)},
        "predicted_effort": {"before": round(eff_before_def, 6), "after": round(eff_after_def, 6)}
    }
    print(f"\nANSWER NASA-16 {json.dumps(answer_16)}")


if __name__ == "__main__":
    main()
