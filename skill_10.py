#!/usr/bin/env python3
"""
Skill 10: Evaluation of our configuration-selection system versus random
and of our prediction model versus a historical-average baseline.

APACHE-19: Compare our configuration-selection system against random
configuration selection under the same benchmarking budget, and show
which approach discovers better-performing configurations.

APACHE-20: Compare our performance predictions against a simple
historical-average estimate and show whether our system provides a
meaningful improvement on previously unseen configurations.
"""

import csv
import json
import sys
import random
import math
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------
# CSV reader – identical to earlier skills
# --------------------------------------------------------------------
def read_csv(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            cleaned = {}
            for k, v in row.items():
                if v == '?':
                    cleaned[k] = None
                else:
                    try:
                        cleaned[k] = float(v) if '.' in v else int(v)
                    except ValueError:
                        cleaned[k] = v
            rows.append((idx, cleaned))
    return rows, reader.fieldnames


# --------------------------------------------------------------------
# Column identification
# --------------------------------------------------------------------
def identify_columns(fieldnames):
    goal_columns = [c for c in fieldnames if c.endswith(('+', '-'))]
    setting_columns = [c for c in fieldnames
                       if c not in goal_columns and not c.endswith('X')]
    return goal_columns, setting_columns


# --------------------------------------------------------------------
# Build linear regression model from a set of evaluated indices
# --------------------------------------------------------------------
def build_model(evaluated_indices, rows, setting_columns, goal_col):
    """Return (model, means) where means maps setting->mean for imputation."""
    means = {}
    for col in setting_columns:
        vals = [r[col] for idx, r in rows if idx in evaluated_indices and r[col] is not None]
        means[col] = sum(vals) / len(vals) if vals else 0.0

    X_train, y_train = [], []
    for idx, r in rows:
        if idx not in evaluated_indices:
            continue
        if r[goal_col] is None:
            continue
        vec = []
        for col in setting_columns:
            val = r[col]
            if val is None:
                val = means[col]
            vec.append(val)
        X_train.append(vec)
        y_train.append(r[goal_col])

    if len(X_train) >= 2:
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        return lr, means
    # If not enough data, fallback to constant model
    class ConstModel:
        def predict(self, X):
            return [y_train[0] if X else 0.0] * len(X)
    return ConstModel(), means


# --------------------------------------------------------------------
# Predict a row using a model
# --------------------------------------------------------------------
def predict_row(row, model, means, setting_columns):
    vec = []
    for col in setting_columns:
        val = row[col]
        if val is None:
            val = means[col]
        vec.append(val)
    return model.predict([vec])[0]


# --------------------------------------------------------------------
# APACHE-13 (greedy active selection)
# --------------------------------------------------------------------
def apache_13(rows, setting_columns, goal_col, budget):
    evaluated = []
    evaluated_set = set()
    best_idx = None
    best_val = None

    def select_next():
        if len(evaluated) < 2:
            # Pick smallest index not yet evaluated
            for idx, _ in rows:
                if idx not in evaluated_set:
                    return idx
        # Build model on evaluated rows
        model, means = build_model(evaluated_set, rows, setting_columns, goal_col)
        # Predict remaining
        best_pred = None
        best_row = None
        for idx, r in rows:
            if idx in evaluated_set:
                continue
            pred = predict_row(r, model, means, setting_columns)
            if best_pred is None or pred < best_pred:
                best_pred = pred
                best_row = idx
        return best_row

    total = len(rows)
    while len(evaluated) < min(budget, total):
        nxt = select_next()
        if nxt is None:
            break
        evaluated.append(nxt)
        evaluated_set.add(nxt)
        _, r = next(item for item in rows if item[0] == nxt)
        val = r[goal_col]
        if val is not None:
            if best_val is None or val < best_val:
                best_val = val
                best_idx = nxt

    return evaluated, best_idx


# --------------------------------------------------------------------
# Scoring for a row: sum of value * sign (+1 for '+', -1 for '-')
# --------------------------------------------------------------------
def compute_score(row, goal_columns):
    score = 0.0
    for col in goal_columns:
        val = row.get(col)
        if val is None:
            continue
        sign = 1.0 if col.endswith('+') else -1.0
        score += sign * val
    return score


# --------------------------------------------------------------------
# Best row index across all rows using the scoring function
# --------------------------------------------------------------------
def best_row_index(rows, goal_columns):
    best_idx = None
    best_score = None
    for idx, r in rows:
        sc = compute_score(r, goal_columns)
        if best_score is None or sc > best_score:
            best_score = sc
            best_idx = idx
    return best_idx, best_score


# --------------------------------------------------------------------
# Train regression on non-held-out rows
# --------------------------------------------------------------------
def train_regression(rows, setting_columns, goal_col):
    # Use all rows that have a goal value
    train_rows = [(idx, r) for idx, r in rows if r[goal_col] is not None]
    setting_means = {}
    for col in setting_columns:
        vals = [r[col] for _, r in train_rows if r[col] is not None]
        setting_means[col] = sum(vals) / len(vals) if vals else 0.0
    X, y = [], []
    for _, r in train_rows:
        vec = []
        for col in setting_columns:
            val = r[col]
            if val is None:
                val = setting_means[col]
            vec.append(val)
        X.append(vec)
        y.append(r[goal_col])
    lr = LinearRegression()
    lr.fit(X, y)
    return lr, setting_means


# --------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, fieldnames = read_csv(csv_path)
    goal_columns, setting_columns = identify_columns(fieldnames)

    if not goal_columns:
        print("Error: No goal column found.")
        sys.exit(1)
    goal_col = goal_columns[0]  # assume single goal for simplicity

    # ----------------------------------------------------------------
    # APACHE-19
    # ----------------------------------------------------------------
    print("== APACHE-19 ==")
    budget = min(30, len(rows) // 5)
    repeats = 20

    # Our method: evaluate once, get best row
    _, ours_best_idx = apache_13(rows, setting_columns, goal_col, budget)

    ours_best_indices = []
    random_best_indices = []

    ours_wins = 0
    random_wins = 0

    # Compute score of ours best row
    _, ours_best_score = best_row_index(rows, goal_columns)

    for rep in range(1, repeats + 1):
        # Random selection
        random.seed(rep)  # deterministic but different each repeat
        selected = random.sample([idx for idx, _ in rows], budget)
        # Find best among selected
        best_rand_idx = None
        best_rand_score = None
        for idx in selected:
            row = dict(rows[idx - 1][1])  # rows list is 1-indexed
            sc = compute_score(row, goal_columns)
            if best_rand_score is None or sc > best_rand_score:
                best_rand_score = sc
                best_rand_idx = idx

        ours_best_indices.append(ours_best_idx)
        random_best_indices.append(best_rand_idx)

        # Compare scores
        if ours_best_score > best_rand_score:
            ours_wins += 1
        elif ours_best_score < best_rand_score:
            random_wins += 1
        # tie -> no win

    # Verdict
    if ours_wins > random_wins:
        verdict = "ours better"
    elif random_wins > ours_wins:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    # Print some intermediate information
    print(f"Budget: {budget}")
    print(f"Repeats: {repeats}")
    print(f"Best our row index: {ours_best_idx}")
    print(f"First three best random indices: {random_best_indices[:3]}")

    ans_19 = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best_indices,
        "random": random_best_indices,
        "verdict": verdict
    }
    print(f"ANSWER APACHE-19 {json.dumps(ans_19)}")

    # ----------------------------------------------------------------
    # APACHE-20
    # ----------------------------------------------------------------
    print("\n== APACHE-20 ==")
    # Identify held-out rows (divisible by 5)
    held_out_ids = {idx for idx, _ in rows if idx % 5 == 0}
    non_held_out = [(idx, r) for idx, r in rows if idx not in held_out_ids]

    # Train model on non-held-out
    lr, means = train_regression(non_held_out, setting_columns, goal_col)

    predictions = {}
    errors_model = []
    errors_base = []

    # Compute baseline mean
    baseline_vals = [r[goal_col] for _, r in non_held_out if r[goal_col] is not None]
    baseline_mean = sum(baseline_vals) / len(baseline_vals) if baseline_vals else 0.0

    for idx, r in rows:
        if idx in held_out_ids:
            true = r[goal_col]
            if true is None:
                continue
            pred = predict_row(r, lr, means, setting_columns)
            predictions[str(idx)] = round(pred, 6)
            errors_model.append(abs(true - pred))
            errors_base.append(abs(true - baseline_mean))

    mae_model = sum(errors_model) / len(errors_model) if errors_model else 0.0
    mae_baseline = sum(errors_base) / len(errors_base) if errors_base else 0.0

    if mae_model < mae_baseline:
        verdict20 = "better"
    elif mae_model > mae_baseline:
        verdict20 = "worse"
    else:
        verdict20 = "no clear difference"

    print(f"Baseline mean Performance-: {round(baseline_mean, 6)}")
    print(f"MAE model: {round(mae_model, 6)}")
    print(f"MAE baseline: {round(mae_baseline, 6)}")

    ans_20 = {
        "predictions": predictions,
        "baseline": round(baseline_mean, 6),
        "mae_model": round(mae_model, 6),
        "mae_baseline": round(mae_baseline, 6),
        "verdict": verdict20
    }
    print(f"ANSWER APACHE-20 {json.dumps(ans_20)}")


if __name__ == "__main__":
    main()
