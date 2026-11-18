#!/usr/bin/env python3
"""
Skill 8: APACHE-15 and APACHE-16

This script performs two analyses on an Apache benchmark CSV:

* APACHE-15 – For a poorly performing configuration, recommend a small change (1 or 2 setting modifications) that could reduce the predicted performance.
* APACHE-16 – For a poorly performing configuration, find the smallest number of setting changes that could bring it closer to the best observed performance.

Both analyses are based solely on the data in the CSV file provided on the command line.
"""

import csv
import json
import sys
import math
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


# ----------------------------------------------------------------------
# CSV utilities
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows = []
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


# ----------------------------------------------------------------------
# Modeling helpers
# ----------------------------------------------------------------------
def train_model(rows: List[Dict[str, Any]], features: List[str], goal_col: str):
    X_list, y_list = [], []
    for r in rows:
        val = r["values"].get(goal_col)
        if val is None:
            continue
        x_row = []
        for f in features:
            v = r["values"].get(f)
            x_row.append(v if v is not None else np.nan)
        X_list.append(x_row)
        y_list.append(val)
    if not X_list:
        # no training data; fallback dummy
        dummy = make_pipeline(SimpleImputer(strategy="median"),
                              LinearRegression())
        dummy.fit(np.array([[0] * len(features)]), np.array([0]))
        return dummy
    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    model = make_pipeline(SimpleImputer(strategy="median"),
                          LinearRegression())
    model.fit(X, y)
    return model


def predict_row(r: Dict[str, Any], model, features: List[str]) -> float:
    x_row = []
    for f in features:
        v = r["values"].get(f)
        x_row.append(v if v is not None else np.nan)
    X = np.array(x_row, dtype=float).reshape(1, -1)
    return float(model.predict(X)[0])


def predict_rows(rows: List[Dict[str, Any]], model, features: List[str]) -> Dict[int, float]:
    return {r["row"]: predict_row(r, model, features) for r in rows}


# ----------------------------------------------------------------------
# Analysis helpers
# ----------------------------------------------------------------------
def diff_settings(row1: Dict[str, Any], row2: Dict[str, Any], setting_cols: List[str]) -> List[Tuple[str, Any, Any]]:
    diffs = []
    for s in setting_cols:
        v1 = row1["values"].get(s)
        v2 = row2["values"].get(s)
        if v1 is None or v2 is None:
            continue
        if v1 != v2:
            diffs.append((s, v1, v2))
    return diffs


def best_row(rows: List[Dict[str, Any]], goal_col: str) -> Dict[str, Any]:
    """Return the row with worst performance (max for min goal, min for max goal)."""
    if goal_col.endswith("-"):
        # minimization target
        return max(rows, key=lambda r: r["values"].get(goal_col, math.inf))
    else:
        return min(rows, key=lambda r: r["values"].get(goal_col, -math.inf))


def best_row_better_than(rows: List[Dict[str, Any]], goal_col: str, compare: str) -> List[Dict[str, Any]]:
    """Return rows whose goal value is better than 'compare' for the goal."""
    if goal_col.endswith("-"):
        # minimization: better if lower
        return [r for r in rows if r["values"].get(goal_col, math.inf) < compare]
    else:
        return [r for r in rows if r["values"].get(goal_col, -math.inf) > compare]


# ----------------------------------------------------------------------
# Main logic for APACHE-15 and APACHE-16
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)

    path = sys.argv[1]
    rows, fieldnames = read_csv(path)

    goal_col = get_goal_col(fieldnames)
    if not goal_col:
        print("No goal column found.")
        sys.exit(1)

    setting_cols = get_setting_cols(fieldnames, goal_col)

    # Train a regression model on all rows with recorded goal values
    model = train_model(rows, setting_cols, goal_col)

    # ------------------------------------------------------------------
    # APACHE-15
    # ------------------------------------------------------------------
    print("== APACHE-15 ==")

    poor_row = best_row(rows, goal_col)
    poor_value = poor_row["values"][goal_col]

    # Find alternatives with 1 or 2 setting differences
    alternatives = []
    for r in rows:
        if r["row"] == poor_row["row"]:
            continue
        if r["values"].get(goal_col) is None:
            continue  # we only consider rows with recorded goal
        diffs = diff_settings(poor_row, r, setting_cols)
        if 1 <= len(diffs) <= 2:
            alternatives.append((r, diffs))

    best_alt = None
    best_improvement = -math.inf  # lower better
    for alt_row, diffs in alternatives:
        alt_value = alt_row["values"][goal_col]
        improvement = poor_value - alt_value if goal_col.endswith("-") else alt_value - poor_value
        if improvement > best_improvement:
            best_improvement = improvement
            best_alt = (alt_row, diffs)

    if best_alt is None:
        # Fallback: choose any row with best improvement among all
        for alt_row in rows:
            if alt_row["row"] == poor_row["row"] or alt_row["values"].get(goal_col) is None:
                continue
            alt_value = alt_row["values"][goal_col]
            improvement = poor_value - alt_value if goal_col.endswith("-") else alt_value - poor_value
            if improvement > best_improvement:
                best_improvement = improvement
                best_alt = (alt_row, diff_settings(poor_row, alt_row, setting_cols))

    if best_alt is None:
        print("No alternative configuration found.")
        sys.exit(1)

    alt_row, alt_diffs = best_alt
    changes = {s: [from_val, to_val] for s, from_val, to_val in alt_diffs}
    performance_after = alt_row["values"][goal_col]
    # round values for readability
    print(f"Poor row: {poor_row['row']} (Performance-={round(poor_value,3)})")
    print(f"Suggested alternative row: {alt_row['row']} (Performance-={round(performance_after,3)})")
    print(f"Changes: {changes}")

    answer_15 = {
        "row": poor_row["row"],
        "alternative": alt_row["row"],
        "changes": changes,
        "performance": {
            "before": round(poor_value, 3),
            "after": round(performance_after, 3)
        }
    }
    print(f"ANSWER APACHE-15 {json.dumps(answer_15, separators=(',', ':'))}")

    # ------------------------------------------------------------------
    # APACHE-16
    # ------------------------------------------------------------------
    print("\n== APACHE-16 ==")

    # Best observed performance (min for minimization, max for maximization)
    recorded_goals = [r["values"][goal_col] for r in rows if r["values"][goal_col] is not None]
    if not recorded_goals:
        print("No recorded goal values.")
        sys.exit(1)
    target_value = min(recorded_goals) if goal_col.endswith("-") else max(recorded_goals)

    # Ensure target is better than poor row
    if (goal_col.endswith("-") and target_value >= poor_value) or (not goal_col.endswith("-") and target_value <= poor_value):
        target_value = target_value  # should already be better

    # Find alternative rows that are closer to the target than the poor row
    poor_dist = abs(poor_value - target_value)
    candidates = []
    for r in rows:
        if r["values"].get(goal_col) is None or r["row"] == poor_row["row"]:
            continue
        alt_val = r["values"][goal_col]
        alt_dist = abs(alt_val - target_value)
        if alt_dist < poor_dist:
            diff = diff_settings(poor_row, r, setting_cols)
            candidates.append((len(diff), diff, r))

    if not candidates:
        print("No alternative configuration closer to target found.")
        sys.exit(1)

    # Pick one with fewest differing settings
    candidates.sort(key=lambda x: (x[0], abs(x[2]["values"][goal_col] - target_value)))
    best_candidate = candidates[0]
    changed_settings = [s for s, _, _ in best_candidate[1]]
    alt_row = best_candidate[2]

    print(f"Poor row: {poor_row['row']} (Performance-={round(poor_value,3)})")
    print(f"Target performance: {round(target_value,3)}")
    print(f"Alternative row: {alt_row['row']} (Performance-={round(alt_row['values'][goal_col],3)})")
    print(f"Changed settings: {changed_settings}")

    answer_16 = {
        "row": poor_row["row"],
        "target": round(target_value, 3),
        "alternative": alt_row["row"],
        "changed": changed_settings
    }
    print(f"ANSWER APACHE-16 {json.dumps(answer_16, separators=(',', ':'))}")


if __name__ == "__main__":
    main()
