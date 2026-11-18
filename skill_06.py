#!/usr/bin/env python3
"""
Skill 6: APACHE-11 and APACHE-12

This script analyses an Apache benchmark CSV file and produces two
recommendation‑type reports:

* APACHE-11 – Recommends a configuration and lists a few similar
  configurations for comparison.

* APACHE-12 – Explains a difference between two configurations,
  showing the settings that differ, the model‑predicted scores,
  and any trade‑offs in the goal columns.

The program prints a human‑readable section for each request and
a machine‑checked JSON line.  The JSON objects are formatted exactly
as required by the evaluation harness.
"""

import csv
import sys
import json
import math
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Read the CSV and return rows and fieldnames."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for i, row in enumerate(reader, start=1):
            cleaned = {}
            for k, v in row.items():
                if v == "?":
                    cleaned[k] = None
                else:
                    try:
                        cleaned[k] = float(v)
                    except ValueError:
                        cleaned[k] = v
            rows.append({"row": i, "values": cleaned})
    return rows, fieldnames


def goal_columns(fieldnames: List[str]) -> List[str]:
    """Return list of columns that end with '+' or '-'."""
    return [c for c in fieldnames if c.endswith(("+", "-"))]


def setting_columns(fieldnames: List[str], goals: List[str]) -> List[str]:
    """Return columns that are inputs for modeling."""
    return [c for c in fieldnames if c not in goals and not c.endswith("X")]


def build_dataset(rows: List[Dict[str, Any]],
                  features: List[str],
                  goals: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return X, y arrays for rows that have a non‑missing goal."""
    X_list, y_list = [], []
    for r in rows:
        y_vals = [r["values"].get(g) for g in goals]
        if any(v is None for v in y_vals):
            continue
        y_list.append([v for v in y_vals])  # keep as list to support multi‑goal
        X_row = []
        for f in features:
            v = r["values"].get(f)
            X_row.append(v if v is not None else np.nan)
        X_list.append(X_row)
    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    return X, y


def predict_rows(rows: List[Dict[str, Any]],
                 model,
                 features: List[str]) -> Dict[int, float]:
    """Return dict mapping row number to predicted value (single goal)."""
    preds = {}
    for r in rows:
        X_row = []
        for f in features:
            v = r["values"].get(f)
            X_row.append(v if v is not None else np.nan)
        X = np.array(X_row, dtype=float).reshape(1, -1)
        pred = model.predict(X)[0]
        preds[r["row"]] = float(pred)
    return preds


def median_imputer(rows: List[Dict[str, Any]], features: List[str]) -> Dict[str, float]:
    """Return median value for each feature across rows (ignoring None)."""
    medians = {}
    for f in features:
        vals = [r["values"].get(f) for r in rows if r["values"].get(f) is not None]
        if vals:
            medians[f] = float(np.median(vals))
        else:
            medians[f] = 0.0
    return medians


def euclidean_distance(row_a: Dict[str, Any], row_b: Dict[str, Any],
                       features: List[str], medians: Dict[str, float]) -> float:
    """Compute Euclidean distance on feature columns, using median for missing."""
    vec_a = []
    vec_b = []
    for f in features:
        va = row_a["values"].get(f)
        vb = row_b["values"].get(f)
        if va is None:
            va = medians[f]
        if vb is None:
            vb = medians[f]
        vec_a.append(va)
        vec_b.append(vb)
    return float(np.linalg.norm(np.array(vec_a) - np.array(vec_b)))


# ----------------------------------------------------------------------
# Main logic for APACHE-11 and APACHE-12
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_06.py <csv_path>")
        sys.exit(1)

    path = sys.argv[1]
    rows, fieldnames = read_csv(path)

    goals = goal_columns(fieldnames)
    if not goals:
        print("No goal column found.")
        sys.exit(1)
    if len(goals) > 1:
        # For simplicity, we only handle one goal column
        goal_col = goals[0]
    else:
        goal_col = goals[0]

    settings = setting_columns(fieldnames, goals)

    # Build regression model (median imputation + linear regression)
    X, y = build_dataset(rows, settings, goals)
    model = make_pipeline(SimpleImputer(strategy="median"), LinearRegression())
    model.fit(X, y[:, 0])

    # Predict all rows
    predictions = predict_rows(rows, model, settings)

    # ------------------------------------------------------------------
    # APACHE-11 – Recommend configuration
    # ------------------------------------------------------------------
    print("== APACHE-11 ==")

    # Best row by actual goal value
    best_row = min(rows, key=lambda r: r["values"].get(goal_col, math.inf))

    # Compute neighbors (k=3)
    medians = median_imputer(rows, settings)
    distances = []
    for r in rows:
        if r["row"] == best_row["row"]:
            continue
        d = euclidean_distance(best_row, r, settings, medians)
        distances.append((d, r))
    distances.sort(key=lambda x: x[0])
    neighbors = [d[1] for d in distances[:3]]

    # Prepare output
    print(f"Recommended row: {best_row['row']} (goal={best_row['values'][goal_col]})")
    print(f"Nearest neighbors: {[n['row'] for n in neighbors]}")
    # Human‑readable details
    print(f"Recommended configuration values: {best_row['values']}")
    print(f"First neighbor values: {neighbors[0]['values']}")

    answer_11 = {
        "recommended": best_row["row"],
        "neighbors": [
            {
                "row": n["row"],
                "values": n["values"]
            } for n in neighbors
        ]
    }
    print(f"ANSWER APACHE-11 {json.dumps(answer_11, separators=(',', ':'))}")

    # ------------------------------------------------------------------
    # APACHE-12 – Explain difference between two configurations
    # ------------------------------------------------------------------
    print("\n== APACHE-12 ==")

    # Find pair with maximum absolute difference in predicted values
    max_pair = None
    max_diff = -1.0
    for i, r1 in enumerate(rows):
        for r2 in rows[i + 1:]:
            # Skip if either missing actual goal
            if r1["values"].get(goal_col) is None or r2["values"].get(goal_col) is None:
                continue
            diff = abs(predictions[r1["row"]] - predictions[r2["row"]])
            if diff > max_diff:
                max_diff = diff
                max_pair = (r1, r2)

    if max_pair is None:
        print("Could not find a valid pair for APACHE-12.")
        sys.exit(1)

    a, b = max_pair
    pred_a = predictions[a["row"]]
    pred_b = predictions[b["row"]]

    # Determine preferred row
    if goal_col.endswith("-"):  # minimisation
        preferred_row = a if pred_a <= pred_b else b
    else:  # maximisation
        preferred_row = a if pred_a >= pred_b else b

    # Differences in settings
    diffs = []
    for s in settings:
        va = a["values"].get(s)
        vb = b["values"].get(s)
        if va is None or vb is None:
            continue
        if va != vb:
            diffs.append({"feature": s, "a": va, "b": vb})
    if not diffs:
        diffs.append({"feature": "none", "a": "none", "b": "none"})

    # Trade‑offs: goal columns where the non‑preferred row has better actual goal
    non_pref = b if preferred_row is a else a
    tradeoffs = []
    for g in goals:
        va = a["values"].get(g)
        vb = b["values"].get(g)
        if va is None or vb is None:
            continue
        if goal_col.endswith("-"):  # minimisation
            better = vb < va
        else:
            better = vb > va
        if better:
            tradeoffs.append(g)

    # Human‑readable output
    print(f"Selected rows: a={a['row']} (pred={pred_a:.3f}), b={b['row']} (pred={pred_b:.3f})")
    print(f"Preferred row: {preferred_row['row']}")
    print(f"Differences in settings: {diffs[:2]}")
    print(f"Trade‑offs (better for non‑preferred): {tradeoffs}")

    answer_12 = {
        "a": a["row"],
        "b": b["row"],
        "preferred": preferred_row["row"],
        "scores": {
            f"a": round(pred_a, 3),
            f"b": round(pred_b, 3)
        },
        "differences": diffs,
        "tradeoffs": tradeoffs
    }
    print(f"ANSWER APACHE-12 {json.dumps(answer_12, separators=(',', ':'))}")


if __name__ == "__main__":
    main()
