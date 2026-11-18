#!/usr/bin/env python3
"""
Skill 10: Evaluation comparison and performance estimation.

This module implements CAR-19 and CAR-20.  It is self‑contained and uses only the
standard library and scikit‑learn (LinearRegression).

Author: OpenAI ChatGPT
"""

import sys
import csv
import json
import random
import math
from typing import List, Dict, Tuple, Optional

# --------------------------------------------------------------------------- #
# Basic parsing helpers
# --------------------------------------------------------------------------- #

def read_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read CSV file and return header and rows as list of dicts."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    return header, rows

def parse_value(v: str) -> Optional[float]:
    """Parse numeric value or return None for missing data."""
    if v in ('', '?'):
        return None
    try:
        return float(v)
    except ValueError:
        return None

def is_goal(col: str) -> bool:
    return col.endswith('+') or col.endswith('-')

def is_input(col: str) -> bool:
    return not is_goal(col) and not col.endswith('X')

# --------------------------------------------------------------------------- #
# Normalisation utilities
# --------------------------------------------------------------------------- #

def compute_minmax(rows: List[Dict[str, str]], cols: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return dicts of min and max for each column."""
    mins, maxs = {}, {}
    for col in cols:
        vals = [parse_value(r[col]) for r in rows if parse_value(r[col]) is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
        else:
            mins[col] = maxs[col] = 0.0
    return mins, maxs

def normalize(v: float, mn: float, mx: float, minimize: bool) -> float:
    """Return normalised value in [0,1].  If mn==mx, return 1."""
    if mn == mx:
        return 1.0
    return (mx - v) / (mx - mn) if minimize else (v - mn) / (mx - mn)

# --------------------------------------------------------------------------- #
# CAR-19 helpers
# --------------------------------------------------------------------------- #

def overall_goal_score(row: Dict[str, str], mins: Dict[str, float], maxs: Dict[str, float], goal_cols: List[str]) -> Optional[float]:
    """Average normalised score over all goal columns.  None if any goal missing."""
    scores = []
    for col in goal_cols:
        v = parse_value(row[col])
        if v is None:
            return None
        mn, mx = mins[col], maxs[col]
        scores.append(normalize(v, mn, mx, col.endswith('-')))
    return sum(scores) / len(scores)

def input_score(row: Dict[str, str], mins: Dict[str, float], maxs: Dict[str, float], input_cols: List[str]) -> Optional[float]:
    """Average normalised input score.  Assume higher is better."""
    scores = []
    for col in input_cols:
        v = parse_value(row[col])
        if v is None:
            return None
        mn, mx = mins[col], maxs[col]
        # For inputs, we treat higher as better (i.e., minimise negative)
        scores.append(normalize(v, mn, mx, False))
    return sum(scores) / len(scores)

def select_best_with_budget(rows: List[Dict[str, str]],
                            budget: int,
                            mins: Dict[str, float],
                            maxs: Dict[str, float],
                            goal_cols: List[str],
                            input_cols: List[str]) -> int:
    """
    Evaluate up to 'budget' rows using a heuristic based on input scores,
    then return the index (1‑based) of the best evaluated row by goal score.
    """
    # Rank rows by input_score
    ranked = []
    for idx, row in enumerate(rows, start=1):
        s = input_score(row, mins, maxs, input_cols)
        if s is not None:
            ranked.append((s, idx, row))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    # Evaluate first 'budget' rows from ranked list
    evaluated = []
    for s, idx, row in ranked[:budget]:
        gscore = overall_goal_score(row, mins, maxs, goal_cols)
        if gscore is not None:
            evaluated.append((gscore, idx))
    if not evaluated:
        # fallback: pick the first evaluated row that had a goal score
        for _, idx, row in ranked:
            gscore = overall_goal_score(row, mins, maxs, goal_cols)
            if gscore is not None:
                return idx
        # no evaluable rows
        return 0
    # choose best
    evaluated.sort(key=lambda x: (-x[0], x[1]))
    return evaluated[0][1]

def random_best(rows: List[Dict[str, str]],
                budget: int,
                mins: Dict[str, float],
                maxs: Dict[str, float],
                goal_cols: List[str]) -> int:
    """Select 'budget' rows uniformly at random and return best by goal score."""
    indices = list(range(1, len(rows)+1))
    chosen = random.sample(indices, min(budget, len(rows)))
    best = None
    best_idx = 0
    for idx in chosen:
        row = rows[idx-1]
        gscore = overall_goal_score(row, mins, maxs, goal_cols)
        if gscore is None:
            continue
        if best is None or gscore > best or (gscore == best and idx < best_idx):
            best = gscore
            best_idx = idx
    return best_idx

# --------------------------------------------------------------------------- #
# CAR-20 helpers
# --------------------------------------------------------------------------- #

from sklearn.linear_model import LinearRegression

def fit_linear_model(rows: List[Dict[str, str]],
                     feature_cols: List[str],
                     target_col: str) -> Tuple[LinearRegression, List[int]]:
    """Fit LinearRegression on rows with complete data.  Return model and training row indices."""
    X, y, train_idx = [], [], []
    for idx, row in enumerate(rows, start=1):
        # parse target
        tgt = parse_value(row[target_col])
        if tgt is None:
            continue
        # parse features
        feat = []
        complete = True
        for fc in feature_cols:
            fv = parse_value(row[fc])
            if fv is None:
                complete = False
                break
            feat.append(fv)
        if not complete:
            continue
        X.append(feat)
        y.append(tgt)
        train_idx.append(idx)
    if not X:
        raise ValueError("No training data available")
    model = LinearRegression()
    model.fit(X, y)
    return model, train_idx

def predict_model(model: LinearRegression,
                  rows: List[Dict[str, str]],
                  feature_cols: List[str]) -> List[Optional[float]]:
    """Predict target for each row.  Return list of predictions; None if inputs missing."""
    preds = []
    for row in rows:
        feat = []
        complete = True
        for fc in feature_cols:
            fv = parse_value(row[fc])
            if fv is None:
                complete = False
                break
            feat.append(fv)
        if not complete:
            preds.append(None)
        else:
            preds.append(float(model.predict([feat])[0]))
    return preds

# --------------------------------------------------------------------------- #
# Main processing
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    # Identify columns
    goal_cols = [c for c in header if is_goal(c)]
    input_cols = [c for c in header if is_input(c)]
    mpg_col = 'Mpg+'
    if mpg_col not in header:
        print("Error: Mpg+ column not found.")
        sys.exit(1)

    # Precompute min/max for goals and inputs
    goal_mins, goal_maxs = compute_minmax(rows, goal_cols)
    input_mins, input_maxs = compute_minmax(rows, input_cols)

    # --------------------------------------------------
    # CAR-19
    # --------------------------------------------------
    print("== CAR-19 ==")
    random.seed(42)  # reproducible
    repeats = 20
    budget = min(30, len(rows)//5)
    ours_best_indices = []
    random_best_indices = []
    for _ in range(repeats):
        ours = select_best_with_budget(rows, budget, goal_mins, goal_maxs, goal_cols, input_cols)
        rnd = random_best(rows, budget, goal_mins, goal_maxs, goal_cols)
        ours_best_indices.append(ours)
        random_best_indices.append(rnd)

    # Determine verdict
    ours_wins = sum(1 for o, r in zip(ours_best_indices, random_best_indices) if o and (overall_goal_score(rows[o-1], goal_mins, goal_maxs, goal_cols) >
                                                                 overall_goal_score(rows[r-1], goal_mins, goal_maxs, goal_cols) if r else True))
    random_wins = sum(1 for o, r in zip(ours_best_indices, random_best_indices) if r and (overall_goal_score(rows[r-1], goal_mins, goal_maxs, goal_cols) >
                                                                 overall_goal_score(rows[o-1], goal_mins, goal_maxs, goal_cols) if o else True))
    if ours_wins > random_wins:
        verdict = "ours better"
    elif random_wins > ours_wins:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    # Output
    print(f"Budget: {budget}")
    print(f"Repeats: {repeats}")
    print(f"Best rows (ours): {ours_best_indices}")
    print(f"Best rows (random): {random_best_indices}")
    print(f"Verdict: {verdict}")
    answer_car19 = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best_indices,
        "random": random_best_indices,
        "verdict": verdict
    }
    print(f"ANSWER CAR-19 {json.dumps(answer_car19)}")

    # --------------------------------------------------
    # CAR-20
    # --------------------------------------------------
    print("\n== CAR-20 ==")
    # Held‑out rows: numbers divisible by 5
    held_out_indices = [i+1 for i in range(len(rows)) if (i+1) % 5 == 0]
    held_out_rows = [rows[i-1] for i in held_out_indices]
    training_rows = [row for idx, row in enumerate(rows, start=1) if idx not in held_out_indices]

    # Baseline: mean Mpg+ of training rows
    baseline_vals = [parse_value(r[mpg_col]) for r in training_rows if parse_value(r[mpg_col]) is not None]
    baseline = sum(baseline_vals)/len(baseline_vals) if baseline_vals else 0.0

    # Train linear model
    feature_cols = [c for c in header if is_input(c)]
    try:
        model, _ = fit_linear_model(rows, feature_cols, mpg_col)
    except ValueError:
        print("Error: No training data available for linear model.")
        sys.exit(1)

    # Predict on held-out rows
    preds = predict_model(model, held_out_rows, feature_cols)
    # Round predictions
    pred_dict = {}
    mae_model = 0.0
    mae_base = 0.0
    count = 0
    for idx, row, pred in zip(held_out_indices, held_out_rows, preds):
        actual = parse_value(row[mpg_col])
        if actual is None:
            continue  # skip if actual missing, shouldn't happen
        if pred is None:
            # use baseline if model can't predict
            pred = baseline
        pred_dict[str(idx)] = round(pred, 4)
        mae_model += abs(pred - actual)
        mae_base += abs(baseline - actual)
        count += 1
    mae_model /= count if count else 0.0
    mae_base /= count if count else 0.0

    # Verdict
    if mae_model < mae_base:
        verdict20 = "better"
    elif mae_model > mae_base:
        verdict20 = "worse"
    else:
        verdict20 = "no clear difference"

    # Output
    print(f"Predictions: {pred_dict}")
    print(f"Baseline mean: {round(baseline,4)}")
    print(f"MAE model: {round(mae_model,4)}")
    print(f"MAE baseline: {round(mae_base,4)}")
    print(f"Verdict: {verdict20}")
    answer_car20 = {
        "predictions": pred_dict,
        "baseline": round(baseline,4),
        "mae_model": round(mae_model,4),
        "mae_baseline": round(mae_base,4),
        "verdict": verdict20
    }
    print(f"ANSWER CAR-20 {json.dumps(answer_car20)}")

if __name__ == "__main__":
    main()
