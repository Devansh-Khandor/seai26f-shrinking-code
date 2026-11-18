#!/usr/bin/env python3
"""
skill_10.py

Entry point for APACHE-19 and APACHE-20 requests.

Usage:
    python skill_10.py <csv_path>
"""

import csv
import sys
import json
import math
import random
from collections import defaultdict
from itertools import combinations

# --------------------------------------------------------------------------- #
# Shared utilities (copied from earlier skills)

def parse_cell(val: str):
    """Parse a CSV cell value: convert to int or None if '?'."""
    val = val.strip()
    if val == "?":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def load_csv(path: str):
    """Load CSV, return list of (row_num, dict_of_columns)."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for i, raw in enumerate(reader, start=1):
            row = {}
            for k, v in raw.items():
                row[k] = parse_cell(v)
            rows.append((i, row))
    return rows


def identify_columns(rows):
    """Return goal_cols, input_cols as lists."""
    col_keys = rows[0][1].keys()
    goal_cols = [col for col in col_keys if col.endswith('+') or col.endswith('-')]
    input_cols = [col for col in col_keys if not (col.endswith('+') or col.endswith('-') or col.endswith('X'))]
    return goal_cols, input_cols


# --------------------------------------------------------------------------- #
# Adaptive evaluation (from skill_07)

def build_feature_matrix(rows, input_cols, target_col):
    """Build X, y from rows where target_col is not None and all inputs are present."""
    X = []
    y = []
    for rnum, row in rows:
        if row.get(target_col) is None:
            continue
        feat = []
        missing = False
        for col in input_cols:
            val = row.get(col)
            if val is None:
                missing = True
                break
            feat.append(val)
        if missing:
            continue
        X.append(feat)
        y.append(row[target_col])
    return X, y


def adaptive_evaluation(rows, input_cols, target_col, budget=30):
    """
    Greedy strategy: start with 5 seeded rows, then iteratively
    train a linear model on evaluated rows and pick next row
    with lowest predicted target among un‑evaluated candidates.
    """
    row_dict = {rnum: row for rnum, row in rows}

    evaluatable = [rnum for rnum, row in rows
                   if row.get(target_col) is not None and
                   all(row.get(col) is not None for col in input_cols)]

    seed = evaluatable[:5]
    evaluated = []
    evaluated_set = set()

    for rnum in seed:
        evaluated.append(rnum)
        evaluated_set.add(rnum)

    remaining = [r for r in evaluatable if r not in evaluated_set]

    while len(evaluated) < budget and remaining:
        train_rows = [(rnum, row_dict[rnum]) for rnum in evaluated]
        X_train, y_train = build_feature_matrix(train_rows, input_cols, target_col)
        if X_train:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X_train, y_train)
            preds = []
            for rnum in remaining:
                row = row_dict[rnum]
                feat = [row[col] for col in input_cols]
                pred = model.predict([feat])[0]
                preds.append((pred, rnum))
            preds.sort()
            _, next_rnum = preds[0]
        else:
            next_rnum = remaining[0]

        evaluated.append(next_rnum)
        evaluated_set.add(next_rnum)
        remaining = [r for r in remaining if r not in evaluated_set]

    return evaluated


def find_best_row(rows, target_col, evaluated):
    """Return the row number with lowest actual target among evaluated."""
    best_val = None
    best_row = None
    for rnum in evaluated:
        val = rows[rnum-1][1].get(target_col)
        if val is None:
            continue
        if best_val is None or val < best_val:
            best_val = val
            best_row = rnum
    return best_row


# --------------------------------------------------------------------------- #
# Prediction functions (from skill_02)

def build_feature_matrix(rows, input_cols, target_col):
    X = []
    y = []
    for rnum, row in rows:
        if row.get(target_col) is None:
            continue
        feat = []
        missing = False
        for col in input_cols:
            val = row.get(col)
            if val is None:
                missing = True
                break
            feat.append(val)
        if missing:
            continue
        X.append(feat)
        y.append(row[target_col])
    return X, y


def predict_held_out(rows, input_cols, target_col):
    """Fit linear model on all non‑held‑out rows and predict on held‑out rows."""
    held_out = [rnum for rnum, _ in rows if rnum % 5 == 0]
    train_rows = [ (rnum, row) for rnum, row in rows if rnum % 5 != 0 ]
    X_train, y_train = build_feature_matrix(train_rows, input_cols, target_col)

    if not X_train:
        return {}, 0.0

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    predictions = {}
    abs_errors = []
    for rnum in held_out:
        row = next(row for rn, row in rows if rn == rnum)
        feat = []
        missing = False
        for col in input_cols:
            val = row.get(col)
            if val is None:
                missing = True
                break
            feat.append(val)
        if missing:
            continue
        pred = lr.predict([feat])[0]
        predictions[rnum] = pred
        actual = row[target_col]
        if actual is not None:
            abs_errors.append(abs(pred - actual))

    mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
    return predictions, mae


# --------------------------------------------------------------------------- #
# Utility: best row score for comparison across all goals

def best_row_score(row, goal_cols):
    """Return a tuple for lexicographic comparison: higher better for '+' goals, lower better for '-' goals."""
    values = []
    for g in goal_cols:
        val = row.get(g)
        if val is None:
            # Missing goal values treated as worst
            if g.endswith('+'):
                val = -float('inf')
            else:
                val = float('inf')
        if g.endswith('+'):
            values.append(val)
        else:
            values.append(-val)
    return tuple(values)


def best_row_num(rows, goal_cols):
    best_num = None
    best_score = None
    for rnum, row in rows:
        score = best_row_score(row, goal_cols)
        if best_score is None or score > best_score:
            best_score = score
            best_num = rnum
    return best_num


# --------------------------------------------------------------------------- #
# Main logic

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)
    goal_cols, input_cols = identify_columns(rows)
    target_col = 'Performance-'

    # ------------------------------------------------------------------ APACHE-19
    print("== APACHE-19 ==")

    budget = min(30, len(rows) // 5)
    repeats = 20
    ours_best = []
    random_best = []

    for rep in range(1, repeats + 1):
        # ours
        evaluated = adaptive_evaluation(rows, input_cols, target_col, budget=budget)
        ours_row = best_row_num([(rnum, rows[rnum-1][1]) for rnum in evaluated], goal_cols)
        ours_best.append(ours_row)

        # random
        rng = random.Random(rep)  # reproducible per repeat
        random_indices = rng.sample([i for i, _ in rows], budget)
        random_row = best_row_num([(rnum, rows[rnum-1][1]) for rnum in random_indices], goal_cols)
        random_best.append(random_row)

        # report first three repeats
        if rep <= 3:
            ours_score = best_row_score(rows[ours_row-1][1], goal_cols)
            rand_score = best_row_score(rows[random_row-1][1], goal_cols)
            print(f"Repeat {rep}: ours row {ours_row} vs random row {random_row}")

    # Summary of which approach found better best row
    ours_count = 0
    rand_count = 0
    for o, r in zip(ours_best, random_best):
        o_score = best_row_score(rows[o-1][1], goal_cols)
        r_score = best_row_score(rows[r-1][1], goal_cols)
        if o_score > r_score:
            ours_count += 1
        elif r_score > o_score:
            rand_count += 1

    if ours_count > rand_count:
        verdict = "ours better"
    elif rand_count > ours_count:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    answer_19 = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best,
        "random": random_best,
        "verdict": verdict
    }
    print("\nANSWER APACHE-19", json.dumps(answer_19))

    # ------------------------------------------------------------------ APACHE-20
    print("\n== APACHE-20 ==")

    predictions, mae_model = predict_held_out(rows, input_cols, target_col)

    # baseline mean of non‑held‑out Performance-
    non_held_vals = [row.get(target_col) for _, row in rows if row.get(target_col) is not None and _ % 5 != 0]
    baseline_mean = sum(non_held_vals) / len(non_held_vals) if non_held_vals else 0.0

    # baseline predictions for held‑out rows
    abs_errors_baseline = []
    for rnum, row in rows:
        if rnum % 5 == 0 and row.get(target_col) is not None:
            abs_errors_baseline.append(abs(row.get(target_col) - baseline_mean))
    mae_baseline = sum(abs_errors_baseline) / len(abs_errors_baseline) if abs_errors_baseline else 0.0

    # Determine verdict
    if mae_model < mae_baseline:
        verdict20 = "better"
    elif mae_model > mae_baseline:
        verdict20 = "worse"
    else:
        verdict20 = "no clear difference"

    answer_20 = {
        "predictions": {str(k): v for k, v in predictions.items()},
        "baseline": baseline_mean,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict20
    }

    print(f"MAE model: {mae_model:.4f}, MAE baseline: {mae_baseline:.4f}")
    print("\nANSWER APACHE-20", json.dumps(answer_20))

if __name__ == "__main__":
    main()
