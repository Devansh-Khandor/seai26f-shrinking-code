#!/usr/bin/env python3
"""
Skill 7: APACHE-13 and APACHE-14

* APACHE-13 – Discover a low‑performance configuration while evaluating as few rows as possible.
  We evaluate the first (up to 30) rows that have a non‑missing goal value.
  The best (lowest for minimisation, highest for maximisation) among the evaluated rows is reported.

* APACHE-14 – Recommend an order of up to 30 rows to evaluate first so that promising settings
  are discovered early.  We iteratively build the order:
      1. Initially pick the smallest row number that has a non‑missing goal.
      2. After each chosen row, train a linear‑regression model on all rows evaluated so far
         (median imputation for missing features).
      3. Predict the goal for every remaining row and pick the one with the best predicted
         performance (lowest for minimisation, highest for maximisation).
  This uses only goal values from rows already evaluated.

The program prints a human‑readable section for each request followed by a machine‑checked
JSON line.  Numbers are plain floats or integers; no NaN/Infinity.  The order of rows is
deterministic given the CSV file.

Author: ChatGPT
"""

import csv
import json
import math
import sys
from typing import Dict, List, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[Dict[str, any]], List[str]]:
    """Read CSV and return rows (with 1‑based row number) and field names."""
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


def get_goal_col(fieldnames: List[str]) -> str:
    for c in fieldnames:
        if c.endswith(("+", "-")):
            return c
    return None


def get_setting_cols(fieldnames: List[str], goal_col: str) -> List[str]:
    return [c for c in fieldnames if c != goal_col and not c.endswith("X")]


def is_minimisation(col: str) -> bool:
    return col.endswith("-")


def best_row_index(rows: List[Dict[str, any]], goal_col: str) -> int:
    """Return the row number with the best goal (min or max)."""
    if is_minimisation(goal_col):
        best = min(rows, key=lambda r: r["values"].get(goal_col, math.inf))
    else:
        best = max(rows, key=lambda r: r["values"].get(goal_col, -math.inf))
    return best["row"]


def train_model(evaluated: List[Dict[str, any]],
                setting_cols: List[str],
                goal_col: str) -> any:
    """Return a fitted pipeline (SimpleImputer + LinearRegression)."""
    X_list, y_list = [], []
    for r in evaluated:
        y_val = r["values"].get(goal_col)
        if y_val is None:
            continue
        x_row = []
        for c in setting_cols:
            v = r["values"].get(c)
            x_row.append(v if v is not None else np.nan)
        X_list.append(x_row)
        y_list.append(y_val)
    if not X_list:
        # No data yet: return a dummy model that predicts 0
        dummy = make_pipeline(SimpleImputer(strategy="median"), LinearRegression())
        dummy.fit(np.array([[0] * len(setting_cols)]), np.array([0]))
        return dummy
    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    model = make_pipeline(SimpleImputer(strategy="median"), LinearRegression())
    model.fit(X, y)
    return model


def predict_row(r: Dict[str, any],
                model: any,
                setting_cols: List[str]) -> float:
    """Return model prediction for a single row."""
    x_row = []
    for c in setting_cols:
        v = r["values"].get(c)
        x_row.append(v if v is not None else np.nan)
    X = np.array(x_row, dtype=float).reshape(1, -1)
    return float(model.predict(X)[0])


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    path = sys.argv[1]
    rows, fieldnames = read_csv(path)

    goal_col = get_goal_col(fieldnames)
    if not goal_col:
        print("No goal column found.")
        sys.exit(1)

    setting_cols = get_setting_cols(fieldnames, goal_col)
    total_rows = len(rows)
    budget = min(30, total_rows // 5)

    # ------------------------------------------------------------------
    # APACHE-13
    # ------------------------------------------------------------------
    print("== APACHE-13 ==")
    evaluated_13 = []
    evaluated_rows_13 = []
    for r in rows:
        if len(evaluated_13) >= 30:
            break
        g = r["values"].get(goal_col)
        if g is None:
            continue
        evaluated_13.append(r["row"])
        evaluated_rows_13.append(r)
    if evaluated_rows_13:
        best_13 = best_row_index(evaluated_rows_13, goal_col)
    else:
        best_13 = None

    print(f"Evaluated rows: {evaluated_13}")
    print(f"Best row found: {best_13}")
    answer_13 = {"evaluated": evaluated_13, "best": best_13}
    print(f"ANSWER APACHE-13 {json.dumps(answer_13, separators=(',', ':'))}")

    # ------------------------------------------------------------------
    # APACHE-14
    # ------------------------------------------------------------------
    print("\n== APACHE-14 ==")
    # Determine rows with known goal (can be evaluated)
    evaluable_rows = [r for r in rows if r["values"].get(goal_col) is not None]
    remaining_set = {r["row"] for r in evaluable_rows}
    evaluated_rows_14: List[Dict[str, any]] = []
    order_14: List[int] = []

    # First choice: smallest row number
    if remaining_set:
        first = min(remaining_set)
        order_14.append(first)
        remaining_set.remove(first)
        evaluated_rows_14.append(next(r for r in evaluable_rows if r["row"] == first))

    # Build order iteratively
    while len(order_14) < budget and remaining_set:
        # Train model on already evaluated rows
        model = train_model(evaluated_rows_14, setting_cols, goal_col)

        # Predict for all remaining rows
        preds = {}
        for r in evaluable_rows:
            if r["row"] in remaining_set:
                preds[r["row"]] = predict_row(r, model, setting_cols)

        # Choose next row with best predicted performance
        if is_minimisation(goal_col):
            next_row = min(preds.items(), key=lambda kv: kv[1])[0]
        else:
            next_row = max(preds.items(), key=lambda kv: kv[1])[0]

        order_14.append(next_row)
        remaining_set.remove(next_row)
        evaluated_rows_14.append(next(r for r in evaluable_rows if r["row"] == next_row))

    print(f"Suggested evaluation order: {order_14}")
    answer_14 = {"order": order_14}
    print(f"ANSWER APACHE-14 {json.dumps(answer_14, separators=(',', ':'))}")


if __name__ == "__main__":
    main()
