#!/usr/bin/env python3
"""
skill_10.py

Usage:
    python skill_10.py <csv_path>
"""

import csv
import json
import sys
import random
from statistics import mean
from collections import defaultdict

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    print("scikit-learn is required for this script.", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def is_goal_col(col_name):
    """Goal columns end with '+' or '-'. """
    return col_name.endswith('+') or col_name.endswith('-')


def is_input_col(col_name):
    """Input columns are those that are not goal columns and do not end with 'X'."""
    return not col_name.endswith('X') and not is_goal_col(col_name)


def parse_value(val, col_name):
    """Parse a CSV cell value."""
    if val == '?':
        return None
    if is_goal_col(col_name):
        try:
            return float(val)
        except ValueError:
            return None
    else:
        try:
            return int(val)
        except ValueError:
            return None


# --------------------------------------------------------------------------- #
# Core logic
# --------------------------------------------------------------------------- #
def main(csv_path):
    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        raw_rows = list(reader)

    # Parse all rows into typed values
    parsed_rows = []
    for r in raw_rows:
        parsed = {}
        for col in fieldnames:
            parsed[col] = parse_value(r[col], col)
        parsed_rows.append(parsed)

    n_rows = len(parsed_rows)

    # Identify goal and input columns
    goal_cols = [c for c in fieldnames if is_goal_col(c)]
    if not goal_cols:
        print("No goal column found.", file=sys.stderr)
        sys.exit(1)

    # Choose the primary goal (minimization if present, else first)
    target_col = None
    for c in goal_cols:
        if c.endswith('-'):
            target_col = c
            break
    if not target_col:
        target_col = goal_cols[0]

    input_cols = [c for c in fieldnames if is_input_col(c)]

    # Helper to get rows with known target
    rows_with_perf = [i + 1 for i, r in enumerate(parsed_rows) if r[target_col] is not None]

    # -------------------------------------------------------------------- #
    # APACHE-19: Compare selection strategy vs random
    # -------------------------------------------------------------------- #
    print("== APACHE-19 ==")

    budget = min(30, len(rows_with_perf) // 5)
    repeats = 20

    ours_best_rows = []
    random_best_rows = []

    # Train a predictive model on all rows with known performance (used for both strategies)
    X_all = [[parsed_rows[i - 1][c] for c in input_cols] for i in rows_with_perf]
    y_all = [parsed_rows[i - 1][target_col] for i in rows_with_perf]

    # RandomForest for prediction
    model_global = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        min_samples_split=2,
        max_depth=None,
    )
    if X_all:
        model_global.fit(X_all, y_all)

    for rep in range(1, repeats + 1):
        # -------- Our strategy --------
        # Predict all rows
        predictions = {}
        for idx in rows_with_perf:
            X_row = [parsed_rows[idx - 1][c] for c in input_cols]
            predictions[idx] = model_global.predict([X_row])[0]

        # Select best predicted
        selected_ours = sorted(predictions.items(), key=lambda x: x[1])[:budget]
        selected_indices_ours = [idx for idx, _ in selected_ours]

        # Evaluate actual
        best_ours = min(selected_indices_ours,
                        key=lambda i: parsed_rows[i - 1][target_col])
        ours_best_rows.append(best_ours)

        # -------- Random strategy --------
        random.seed(rep)
        if len(rows_with_perf) <= budget:
            selected_random = rows_with_perf[:]
        else:
            selected_random = random.sample(rows_with_perf, budget)

        best_random = min(selected_random,
                          key=lambda i: parsed_rows[i - 1][target_col])
        random_best_rows.append(best_random)

    # Compute verdict
    ours_wins = sum(
        1 for o, r in zip(ours_best_rows, random_best_rows)
        if parsed_rows[o - 1][target_col] < parsed_rows[r - 1][target_col]
    )
    random_wins = repeats - ours_wins
    if ours_wins > random_wins:
        verdict19 = "ours better"
    elif random_wins > ours_wins:
        verdict19 = "random better"
    else:
        verdict19 = "no clear difference"

    # Print some summary
    print(f"Budget: {budget}")
    print(f"Repeats: {repeats}")
    print(f"First repeat best row (ours): {ours_best_rows[0]}")
    print(f"First repeat best row (random): {random_best_rows[0]}")

    # Machine‑checked answer
    answer_19 = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best_rows,
        "random": random_best_rows,
        "verdict": verdict19
    }
    print("ANSWER APACHE-19", json.dumps(answer_19, separators=(',', ':')))

    # -------------------------------------------------------------------- #
    # APACHE-20: Prediction vs baseline
    # -------------------------------------------------------------------- #
    print("\n== APACHE-20 ==")

    # Identify held‑out rows: divisible by 5 (1‑based)
    held_out_indices = [i + 1 for i in range(n_rows) if (i + 1) % 5 == 0]
    train_indices = [i + 1 for i in range(n_rows) if (i + 1) % 5 != 0]

    # Build training data (exclude held‑out)
    X_train = []
    y_train = []
    for idx in train_indices:
        val = parsed_rows[idx - 1][target_col]
        if val is not None:
            x = [parsed_rows[idx - 1][c] for c in input_cols]
            X_train.append(x)
            y_train.append(val)

    model_pred = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        min_samples_split=2,
        max_depth=None,
    )
    if X_train:
        model_pred.fit(X_train, y_train)

    predictions_map = {}
    mae_model_vals = []

    for idx in held_out_indices:
        val = parsed_rows[idx - 1][target_col]
        if val is None:
            continue
        X_row = [parsed_rows[idx - 1][c] for c in input_cols]
        pred = model_pred.predict([X_row])[0]
        predictions_map[str(idx)] = round(pred, 3)
        mae_model_vals.append(abs(pred - val))

    mae_model = round(mean(mae_model_vals), 3) if mae_model_vals else None

    # Baseline: mean of training performance
    baseline_vals = [parsed_rows[idx - 1][target_col] for idx in train_indices if parsed_rows[idx - 1][target_col] is not None]
    baseline_mean = round(mean(baseline_vals), 3) if baseline_vals else None
    mae_baseline_vals = []
    for idx in held_out_indices:
        val = parsed_rows[idx - 1][target_col]
        if val is not None:
            mae_baseline_vals.append(abs(baseline_mean - val))
    mae_baseline = round(mean(mae_baseline_vals), 3) if mae_baseline_vals else None

    # Verdict
    if mae_model is not None and mae_baseline is not None:
        if mae_model < mae_baseline:
            verdict20 = "better"
        elif mae_model > mae_baseline:
            verdict20 = "worse"
        else:
            verdict20 = "no clear difference"
    else:
        verdict20 = "no clear difference"

    # Print summary
    print(f"Baseline mean: {baseline_mean}")
    print(f"Model MAE: {mae_model}")
    print(f"Baseline MAE: {mae_baseline}")

    # Machine‑checked answer
    answer_20 = {
        "predictions": predictions_map,
        "baseline": baseline_mean,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict20
    }
    print("ANSWER APACHE-20", json.dumps(answer_20, separators=(',', ':')))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
