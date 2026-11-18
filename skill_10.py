#!/usr/bin/env python3
"""
Skill 10: APACHE-19 and APACHE-20

This script implements the final two analyses required by the client:

APACHE-19 – Compares a simple greedy selection strategy that uses a
regression model to guide the choice of configurations with a fixed
budget against random selection.  For each of 20 independent repeats the
best configuration found by each strategy is reported; the overall
advantage of the greedy method is summarized.

APACHE-20 – Assesses the predictive power of the same regression
model against a naïve baseline that uses the mean of all historical
records.  It reports the MAE of the model, the MAE of the baseline,
the predictions for all held‑out rows (rows whose number is divisible by
5), and a verdict whether the model is better, worse, or no clear
difference.

The program expects the path to the CSV file as its only command‑line
argument and prints one human‑readable section per request followed
by the corresponding machine‑checked answer line.
"""

import csv
import json
import math
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# ----------------------------------------------------------------------
# CSV handling
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[Dict[str, any]], List[str]]:
    """Read CSV and return rows (1‑based) and fieldnames."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for i, row in enumerate(reader, start=1):
            cleaned: Dict[str, any] = {}
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


def is_minimisation(col: str) -> bool:
    return col.endswith("-")


def get_setting_cols(fieldnames: List[str], goal_col: str) -> List[str]:
    return [c for c in fieldnames if c != goal_col and not c.endswith("X")]

# ----------------------------------------------------------------------
# Regression helpers
# ----------------------------------------------------------------------
def train_regression(rows: List[Dict[str, any]],
                     setting_cols: List[str],
                     goal_col: str):
    X_list, y_list = [], []
    for r in rows:
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
        # Return dummy model predicting 0
        dummy = make_pipeline(SimpleImputer(strategy="median"),
                              LinearRegression())
        dummy.fit(np.array([[0] * len(setting_cols)]), np.array([0]))
        return dummy
    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    model = make_pipeline(SimpleImputer(strategy="median"),
                          LinearRegression())
    model.fit(X, y)
    return model


def predict_row(r: Dict[str, any], model, setting_cols: List[str]) -> float:
    x_row = []
    for c in setting_cols:
        v = r["values"].get(c)
        x_row.append(v if v is not None else np.nan)
    X = np.array(x_row, dtype=float).reshape(1, -1)
    return float(model.predict(X)[0])

def predict_all(rows: List[Dict[str, any]],
                model,
                setting_cols: List[str]) -> Dict[int, float]:
    return {r["row"]: predict_row(r, model, setting_cols) for r in rows}

# ----------------------------------------------------------------------
# APACHE-19 helpers
# ----------------------------------------------------------------------
def greedy_selection(rows: List[Dict[str, any]],
                     goal_col: str,
                     setting_cols: List[str],
                     budget: int,
                     seed: int) -> Tuple[List[int], int]:
    """
    Greedy strategy that selects rows sequentially.
    Returns the list of selected row numbers and the best row number among them.
    """
    random.seed(seed)
    # Rows that have a known goal value
    available = [r for r in rows if r["values"].get(goal_col) is not None]
    if not available:
        return [], None

    # Initial selection: first row with known goal
    evaluated = [available[0]]
    remaining = {r["row"]: r for r in available[1:]}
    best_row_num = evaluated[0]["row"]

    while len(evaluated) < budget and remaining:
        # Train model on evaluated rows
        model = train_regression(evaluated, setting_cols, goal_col)

        # Predict remaining rows
        preds = {}
        for row_num, r in remaining.items():
            preds[row_num] = predict_row(r, model, setting_cols)

        # Choose next row based on predicted performance
        if is_minimisation(goal_col):
            next_row_num = min(preds, key=preds.get)
        else:
            next_row_num = max(preds, key=preds.get)

        # Update
        next_row = remaining.pop(next_row_num)
        evaluated.append(next_row)

        # Track best among evaluated
        cur_val = next_row["values"][goal_col]
        if (is_minimisation(goal_col) and cur_val < best_row_num_val) or \
           (not is_minimisation(goal_col) and cur_val > best_row_num_val):
            best_row_num = next_row["row"]
            best_row_num_val = cur_val
        else:
            # keep previous best
            best_row_num_val = best_row_num_val

    # Determine best row among evaluated
    best_val = None
    best_idx = None
    for r in evaluated:
        val = r["values"][goal_col]
        if best_val is None or \
           (is_minimisation(goal_col) and val < best_val) or \
           (not is_minimisation(goal_col) and val > best_val):
            best_val = val
            best_idx = r["row"]
    return [r["row"] for r in evaluated], best_idx


def random_selection(rows: List[Dict[str, any]],
                     goal_col: str,
                     setting_cols: List[str],
                     budget: int,
                     seed: int) -> Tuple[List[int], int]:
    random.seed(seed)
    available = [r for r in rows if r["values"].get(goal_col) is not None]
    if not available:
        return [], None
    selected = random.sample(available, min(budget, len(available)))
    # Determine best row among selected
    best_val = None
    best_idx = None
    for r in selected:
        val = r["values"][goal_col]
        if best_val is None or \
           (is_minimisation(goal_col) and val < best_val) or \
           (not is_minimisation(goal_col) and val > best_val):
            best_val = val
            best_idx = r["row"]
    return [r["row"] for r in selected], best_idx


# ----------------------------------------------------------------------
# APACHE-20 helpers
# ----------------------------------------------------------------------
def evaluate_apache_20(rows: List[Dict[str, any]],
                       goal_col: str,
                       setting_cols: List[str]):
    # Held‑out rows: divisible by 5
    held_out = [r for r in rows if r["row"] % 5 == 0 and r["values"][goal_col] is not None]
    history = [r for r in rows if r["row"] % 5 != 0 and r["values"][goal_col] is not None]

    # Train model on history
    model = train_regression(history, setting_cols, goal_col)

    # Predictions for all rows
    preds = predict_all(rows, model, setting_cols)

    # Compute MAE of model on held‑out
    errors = [abs(preds[r["row"]] - r["values"][goal_col]) for r in held_out]
    mae_model = sum(errors) / len(errors) if errors else 0.0

    # Baseline mean of non‑held‑out values
    baseline = np.mean([r["values"][goal_col] for r in history]) if history else 0.0
    mae_baseline = sum(abs(r["values"][goal_col] - baseline) for r in held_out) / len(held_out) if held_out else 0.0

    # Verdict
    if mae_model < mae_baseline - 1e-6:
        verdict = "better"
    elif mae_model > mae_baseline + 1e-6:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    # Prepare predictions mapping (row number as string)
    predictions = {str(r["row"]): round(preds[r["row"]], 3) for r in held_out}

    return predictions, round(baseline, 3), round(mae_model, 3), round(mae_baseline, 3), verdict

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)

    path = sys.argv[1]
    rows, fieldnames = read_csv(path)
    goal_col = get_goal_col(fieldnames)
    if goal_col is None:
        print("No goal column found.")
        sys.exit(1)
    setting_cols = get_setting_cols(fieldnames, goal_col)
    budget = min(30, len(rows) // 5)

    # -------- APACHE-19 ----------
    print("== APACHE-19 ==")
    repeats = 20
    ours_best = []
    random_best = []

    for i in range(repeats):
        seed = 42 + i  # deterministic per repeat
        _, ours_row = greedy_selection(rows, goal_col, setting_cols, budget, seed)
        _, random_row = random_selection(rows, goal_col, setting_cols, budget, seed)
        if ours_row is not None:
            ours_best.append(ours_row)
        if random_row is not None:
            random_best.append(random_row)

    # Determine overall wins
    ours_wins = 0
    random_wins = 0
    for o, r in zip(ours_best, random_best):
        # Compare the performance values of these rows
        o_val = next(r_["values"][goal_col] for r_ in rows if r_["row"] == o)
        r_val = next(r_["values"][goal_col] for r_ in rows if r_["row"] == r)
        if (is_minimisation(goal_col) and o_val < r_val) or \
           (not is_minimisation(goal_col) and o_val > r_val):
            ours_wins += 1
        elif (is_minimisation(goal_col) and o_val > r_val) or \
             (not is_minimisation(goal_col) and o_val < r_val):
            random_wins += 1
        # ties do not count

    if ours_wins > random_wins:
        verdict = "ours better"
    elif random_wins > ours_wins:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    # Human readable summary
    print(f"Budget per run: {budget}")
    print(f"Repeats performed: {repeats}")
    print(f"Best rows (ours): {ours_best[:5]}{'...' if len(ours_best)>5 else ''}")
    print(f"Best rows (random): {random_best[:5]}{'...' if len(random_best)>5 else ''}")
    print(f"Result: {verdict}")

    answer_19 = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best,
        "random": random_best,
        "verdict": verdict
    }
    print(f"ANSWER APACHE-19 {json.dumps(answer_19, separators=(',', ':'))}")

    # -------- APACHE-20 ----------
    print("\n== APACHE-20 ==")
    predictions, baseline, mae_model, mae_baseline, verdict20 = evaluate_apache_20(rows, goal_col, setting_cols)

    # Human readable summary
    print(f"Baseline mean (non‑held‑out): {baseline}")
    print(f"MAE of model: {mae_model}")
    print(f"MAE of baseline: {mae_baseline}")
    print(f"Verdict: {verdict20}")
    print(f"Predictions for held‑out rows (first 5 shown):")
    first_five = dict(list(predictions.items())[:5])
    print(first_five)

    answer_20 = {
        "predictions": predictions,
        "baseline": baseline,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict20
    }
    print(f"ANSWER APACHE-20 {json.dumps(answer_20, separators=(',', ':'))}")


if __name__ == "__main__":
    main()
