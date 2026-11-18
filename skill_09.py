#!/usr/bin/env python3
"""
Skill 9: APACHE-17 and APACHE-18

The program analyses an Apache benchmark CSV file to identify:
  * APACHE-17 – rows whose measured performance deviates unusually from
    the prediction based on their settings.
  * APACHE-18 – rows whose combination of settings is unusually far from
    any previously seen configuration, regardless of performance.

The output consists of human‑readable sections followed by a machine‑checked JSON
line for each request, as required by the evaluation harness.
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
# CSV utilities
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Read CSV and return rows (1‑based) and fieldnames."""
    rows: List[Dict[str, Any]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for i, row in enumerate(reader, start=1):
            cleaned: Dict[str, Any] = {}
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


def get_goal_col(fieldnames: List[str]) -> str:
    for c in fieldnames:
        if c.endswith(("+", "-")):
            return c
    return None


def get_setting_cols(fieldnames: List[str], goal_col: str) -> List[str]:
    return [c for c in fieldnames if c != goal_col and not c.endswith("X")]


def median_dict(rows: List[Dict[str, Any]], cols: List[str]) -> Dict[str, float]:
    medians: Dict[str, float] = {}
    for col in cols:
        vals = [r["values"].get(col) for r in rows if r["values"].get(col) is not None]
        medians[col] = float(np.median(vals)) if vals else 0.0
    return medians


# ----------------------------------------------------------------------
# Modeling utilities
# ----------------------------------------------------------------------
def build_regression(rows: List[Dict[str, Any]],
                     feature_cols: List[str],
                     goal_col: str):
    X_list, y_list = [], []
    for r in rows:
        val = r["values"].get(goal_col)
        if val is None:
            continue
        x_row = []
        for c in feature_cols:
            v = r["values"].get(c)
            x_row.append(v if v is not None else np.nan)
        X_list.append(x_row)
        y_list.append(val)
    if not X_list:
        # No training data: return a dummy model predicting 0
        dummy = make_pipeline(SimpleImputer(strategy="median"),
                              LinearRegression())
        dummy.fit(np.array([[0] * len(feature_cols)]), np.array([0]))
        return dummy
    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    model = make_pipeline(SimpleImputer(strategy="median"),
                          LinearRegression())
    model.fit(X, y)
    return model


def predict_all(rows: List[Dict[str, Any]],
                model,
                feature_cols: List[str]) -> Dict[int, float]:
    preds: Dict[int, float] = {}
    for r in rows:
        x_row = []
        for c in feature_cols:
            v = r["values"].get(c)
            x_row.append(v if v is not None else np.nan)
        X = np.array(x_row, dtype=float).reshape(1, -1)
        preds[r["row"]] = float(model.predict(X)[0])
    return preds


# ----------------------------------------------------------------------
# Distance utilities
# ----------------------------------------------------------------------
def nearest_neighbor_distance(row: Dict[str, Any],
                              history: List[Dict[str, Any]],
                              feature_cols: List[str],
                              medians: Dict[str, float]) -> float:
    """Return Euclidean distance to nearest row in history."""
    vec_row = []
    for c in feature_cols:
        v = row["values"].get(c)
        vec_row.append(v if v is not None else medians[c])
    vec_row = np.array(vec_row, dtype=float)
    min_dist = float("inf")
    for h in history:
        if h["row"] == row["row"]:
            continue
        vec_hist = []
        for c in feature_cols:
            v = h["values"].get(c)
            vec_hist.append(v if v is not None else medians[c])
        vec_hist = np.array(vec_hist, dtype=float)
        dist = float(np.linalg.norm(vec_row - vec_hist))
        if dist < min_dist:
            min_dist = dist
    return min_dist


# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)

    path = sys.argv[1]
    rows, fieldnames = read_csv(path)
    n_rows = len(rows)

    goal_col = get_goal_col(fieldnames)
    if goal_col is None:
        print("No goal column found.")
        sys.exit(1)

    setting_cols = get_setting_cols(fieldnames, goal_col)

    # Determine history / arrivals
    arr_size = int(round(0.2 * n_rows))
    history_rows = rows[:-arr_size] if arr_size < n_rows else rows
    arrival_rows = rows[-arr_size:] if arr_size > 0 else []

    # ------------------------------------------------------------------
    # APACHE-17
    # ------------------------------------------------------------------
    print("== APACHE-17 ==")
    reg_model = build_regression(history_rows, setting_cols, goal_col)
    all_preds = predict_all(rows, reg_model, setting_cols)

    # Residuals on training set
    train_residuals = []
    for r in history_rows:
        actual = r["values"].get(goal_col)
        if actual is None:
            continue
        pred = all_preds.get(r["row"])
        train_residuals.append(abs(actual - pred))
    if train_residuals:
        mean_res = np.mean(train_residuals)
        std_res = np.std(train_residuals, ddof=0)
    else:
        mean_res, std_res = 0.0, 0.0
    threshold_res = mean_res + 2 * std_res if std_res > 0 else 0.0

    flagged_17 = []
    for r in rows:
        actual = r["values"].get(goal_col)
        if actual is None:
            continue
        pred = all_preds.get(r["row"])
        if pred is None:
            continue
        resid = abs(actual - pred)
        if resid > threshold_res:
            flagged_17.append({
                "row": r["row"],
                "column": goal_col,
                "actual": float(actual),
                "expected": float(pred)
            })
        if len(flagged_17) >= 20:
            break

    # Human‑readable output
    print(f"Total flagged rows: {len(flagged_17)}")
    if flagged_17:
        first = flagged_17[0]
        dev = abs(first["actual"] - first["expected"])
        print(f"Example: row {first['row']} deviation={dev:.3f}")
    else:
        print("No flagged rows found.")
    answer_17 = {"flagged": flagged_17}
    print(f"ANSWER APACHE-17 {json.dumps(answer_17, separators=(',', ':'))}")

    # ------------------------------------------------------------------
    # APACHE-18
    # ------------------------------------------------------------------
    print("\n== APACHE-18 ==")
    # Median values for missing imputation
    medians = median_dict(rows, setting_cols)

    # Compute distances to nearest neighbor in history for all rows
    distances = []
    for r in rows:
        dist = nearest_neighbor_distance(r, history_rows, setting_cols, medians)
        distances.append(dist)

    # Statistics
    if distances:
        mean_dist = np.mean(distances)
        std_dist = np.std(distances, ddof=0)
    else:
        mean_dist, std_dist = 0.0, 0.0
    threshold_dist = mean_dist + 2 * std_dist if std_dist > 0 else 0.0

    flagged_18_rows = []
    scores: Dict[str, float] = {}
    for r, dist in zip(rows, distances):
        if dist > threshold_dist:
            flagged_18_rows.append(r["row"])
            scores[str(r["row"])] = float(round(dist, 3))
        if len(flagged_18_rows) >= 20:
            break

    # Human‑readable output
    print(f"Total flagged rows: {len(flagged_18_rows)}")
    if flagged_18_rows:
        print(f"Example flagged row: {flagged_18_rows[0]}, distance={scores[str(flagged_18_rows[0])]}")
    else:
        print("No flagged rows found.")
    answer_18 = {"flagged": flagged_18_rows, "scores": scores}
    print(f"ANSWER APACHE-18 {json.dumps(answer_18, separators=(',', ':'))}")


if __name__ == "__main__":
    main()
