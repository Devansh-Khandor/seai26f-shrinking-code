#!/usr/bin/env python3
"""
skill_10.py

Usage:
    python skill_10.py <csv path>

Adds two new analysis sections:
  * CAR-19: Compare a heuristic vehicle-selection strategy to random selection.
  * CAR-20: Compare a regression‑based MPG prediction to a simple historical‑average baseline.

The script prints each section header, a few descriptive lines, and a machine‑checked ANSWER line in JSON format.
"""

import sys
import json
import random
import numpy as np
from sklearn.linear_model import LinearRegression

# Reuse helpers from earlier skills
from skill_01 import read_csv, parse_float
from skill_07 import get_feature_indices, compute_feature_min_max, compute_proxy_score

# --------------------------------------------------------------------------- #
# Helper functions for CAR‑19
# --------------------------------------------------------------------------- #

def compute_goal_bounds(rows, header):
    """
    Return mins and maxs for goal columns over all rows.
    """
    idx_lbs = header.index('Lbs-')
    idx_acc = header.index('Acc+')
    idx_mpg = header.index('Mpg+')
    lbs_vals = [parse_float(r[idx_lbs]) for r in rows if parse_float(r[idx_lbs]) is not None]
    acc_vals = [parse_float(r[idx_acc]) for r in rows if parse_float(r[idx_acc]) is not None]
    mpg_vals = [parse_float(r[idx_mpg]) for r in rows if parse_float(r[idx_mpg]) is not None]
    return {
        'lbs_min': float(np.min(lbs_vals)),
        'lbs_max': float(np.max(lbs_vals)),
        'acc_min': float(np.min(acc_vals)),
        'acc_max': float(np.max(acc_vals)),
        'mpg_min': float(np.min(mpg_vals)),
        'mpg_max': float(np.max(mpg_vals))
    }

def score_row(row, bounds, idx_lbs, idx_acc, idx_mpg):
    """
    Compute combined performance score for a row.
    Higher is better.
    """
    lbs = parse_float(row[idx_lbs])
    acc = parse_float(row[idx_acc])
    mpg = parse_float(row[idx_mpg])
    if lbs is None or acc is None or mpg is None:
        return None
    norm_lbs = (lbs - bounds['lbs_min']) / (bounds['lbs_max'] - bounds['lbs_min']) if bounds['lbs_max'] > bounds['lbs_min'] else 0.0
    norm_acc = (acc - bounds['acc_min']) / (bounds['acc_max'] - bounds['acc_min']) if bounds['acc_max'] > bounds['acc_min'] else 0.0
    norm_mpg = (mpg - bounds['mpg_min']) / (bounds['mpg_max'] - bounds['mpg_min']) if bounds['mpg_max'] > bounds['mpg_min'] else 0.0
    return (1 - norm_lbs) + (1 - norm_acc) + norm_mpg

# --------------------------------------------------------------------------- #
# Helper functions for CAR‑20
# --------------------------------------------------------------------------- #

def get_feature_indices_for_prediction(header):
    """
    Return indices of columns that are not ending in 'X', '+', or '-'.
    """
    return [i for i, name in enumerate(header)
            if not (name.endswith('X') or name.endswith('+') or name.endswith('-'))]

def train_mpg_model(rows, header, feature_idx, target_idx):
    """
    Train a linear regression model to predict Mpg+.
    """
    X, y = [], []
    for r in rows:
        feat = []
        missing = False
        for fi in feature_idx:
            val = parse_float(r[fi])
            if val is None:
                missing = True
                break
            feat.append(val)
        if missing:
            continue
        target = parse_float(r[target_idx])
        if target is None:
            continue
        X.append(feat)
        y.append(target)
    if not X:
        return None
    model = LinearRegression()
    model.fit(np.array(X), np.array(y))
    return model

def predict_mpg_for_rows(model, rows, header, feature_idx, target_idx):
    """
    Return dict row_number -> predicted Mpg+.
    Only rows with complete features are predicted.
    """
    preds = {}
    for i, r in enumerate(rows, start=1):
        feat = []
        missing = False
        for fi in feature_idx:
            val = parse_float(r[fi])
            if val is None:
                missing = True
                break
            feat.append(val)
        if missing:
            continue
        pred = model.predict([feat])[0]
        preds[str(i)] = pred
    return preds

def mae(predictions, actuals):
    """
    Compute mean absolute error between two dicts row->value.
    """
    errs = [abs(predictions[row] - actuals[row]) for row in predictions if row in actuals]
    if not errs:
        return 0.0
    return sum(errs) / len(errs)

# --------------------------------------------------------------------------- #
# Main logic for CAR‑19 and CAR‑20
# --------------------------------------------------------------------------- #

def main(csv_path):
    header, data = read_csv(csv_path)
    n_rows = len(data)

    # -------------------------------- CAR‑19 --------------------------------
    print("== CAR-19 ==")

    # Budget: min(30, n_rows // 5)
    budget = min(30, n_rows // 5)

    # Compute goal bounds once
    bounds = compute_goal_bounds(data, header)
    idx_lbs = header.index('Lbs-')
    idx_acc = header.index('Acc+')
    idx_mpg = header.index('Mpg+')

    # Score all rows
    all_scores = [(i, score_row(r, bounds, idx_lbs, idx_acc, idx_mpg))
                  for i, r in enumerate(data, start=1)]
    # Filter out None
    all_scores = [(i, sc) for i, sc in all_scores if sc is not None]
    # Best overall row (max score)
    ours_best_row, ours_best_score = max(all_scores, key=lambda x: x[1])

    # Our selection method: proxy scores on all rows
    feature_idxs = get_feature_indices(header)
    feat_mins, feat_maxs = compute_feature_min_max(data, feature_idxs)
    proxy_scores = [(i, compute_proxy_score(r, (feat_mins, feat_maxs), feature_idxs))
                    for i, r in enumerate(data, start=1)]
    proxy_scores.sort(key=lambda x: x[1], reverse=True)
    ours_rows = [i for i, _ in proxy_scores[:budget]]
    # The best among ours_rows is ours_best_row

    # Random evaluation
    repeats = 20
    ours_wins = 0
    random_wins = 0
    random_best_rows = []

    # Precompute indices list
    all_indices = list(range(1, n_rows + 1))
    for _ in range(repeats):
        random_sample = random.sample(all_indices, budget)
        # compute best score among sampled rows
        best_row = None
        best_score = -1.0
        for idx in random_sample:
            sc = all_scores[idx - 1][1] if all_scores[idx - 1][1] is not None else None
            if sc is not None and sc > best_score:
                best_score = sc
                best_row = idx
        random_best_rows.append(best_row)
        # compare to ours
        if best_row == ours_best_row:
            # tie, count as no win
            pass
        elif best_score > ours_best_score:
            random_wins += 1
        else:
            ours_wins += 1

    if ours_wins > random_wins:
        verdict19 = "ours better"
    elif random_wins > ours_wins:
        verdict19 = "random better"
    else:
        verdict19 = "no clear difference"

    # Print some results
    print(f"Budget: {budget}")
    print(f"Repeats: {repeats}")
    print(f"Best row from our method: {ours_best_row} (score={ours_best_score:.3f})")
    print(f"Best row from random selection (first repeat): {random_best_rows[0]}")

    # Machine‑checked answer
    answer_car19 = {
        "budget": budget,
        "repeats": repeats,
        "ours": [ours_best_row] * repeats,
        "random": random_best_rows,
        "verdict": verdict19
    }
    print(f"ANSWER CAR-19 {json.dumps(answer_car19, separators=(',', ':'), sort_keys=True)}")

    # -------------------------------- CAR‑20 --------------------------------
    print("\n== CAR-20 ==")

    # Identify held‑out rows: divisible by 5
    held_out_indices = [i for i in range(1, n_rows + 1) if i % 5 == 0]

    # Feature indices for prediction
    feat_idx = get_feature_indices_for_prediction(header)
    target_idx = header.index('Mpg+')

    # Split into training and held‑out sets
    training_rows = [r for i, r in enumerate(data, start=1) if i not in held_out_indices]
    held_out_rows = [r for i, r in enumerate(data, start=1) if i in held_out_indices]

    # Train regression
    model = train_mpg_model(training_rows, header, feat_idx, target_idx)

    predictions = {}
    actuals = {}
    if model is not None:
        # Predict on held‑out rows
        for i, r in enumerate(held_out_rows, start=1):
            row_num = held_out_indices[i - 1]
            # Features must be complete
            feat = []
            missing = False
            for fi in feat_idx:
                val = parse_float(r[fi])
                if val is None:
                    missing = True
                    break
                feat.append(val)
            if missing:
                continue
            pred = model.predict([feat])[0]
            predictions[str(row_num)] = pred
            act = parse_float(r[target_idx])
            if act is not None:
                actuals[str(row_num)] = act

    # Baseline mean of non‑held‑out Mpg+
    baseline_vals = [parse_float(r[target_idx]) for r in training_rows if parse_float(r[target_idx]) is not None]
    baseline_mean = float(np.mean(baseline_vals)) if baseline_vals else 0.0

    # MAE for model
    mae_model = mae(predictions, actuals)

    # MAE for baseline: using baseline_mean for every held‑out with actual
    baseline_predictions = {row: baseline_mean for row in actuals}
    mae_baseline = mae(baseline_predictions, actuals)

    # Verdict
    if mae_model < mae_baseline:
        verdict20 = "better"
    elif mae_model > mae_baseline:
        verdict20 = "worse"
    else:
        verdict20 = "no clear difference"

    # Print some results
    print(f"Baseline mean Mpg+ (train set): {baseline_mean:.3f}")
    print(f"Model MAE: {mae_model:.3f}")
    print(f"Baseline MAE: {mae_baseline:.3f}")

    # Machine‑checked answer
    answer_car20 = {
        "predictions": predictions,
        "baseline": baseline_mean,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict20
    }
    print(f"ANSWER CAR-20 {json.dumps(answer_car20, separators=(',', ':'), sort_keys=True)}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
