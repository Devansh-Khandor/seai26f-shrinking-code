#!/usr/bin/env python3
"""
skill_10.py

Implements NASA-19 and NASA-20 analyses for the NASA project dataset.

NASA-19: Compare our project‑effort estimation system against a simple historical‑average estimate and show whether it makes meaningfully better predictions on previously unseen projects.
NASA-20: Compare our project‑selection approach with randomly selecting proposals under the same evaluation budget, and show whether our system identifies more desirable project outcomes.
"""

import sys
import csv
import json
import math
import random
from collections import defaultdict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ----------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------
def parse_value(v: str):
    """Parse a cell value, returning float if possible, else original string or None for missing."""
    v = v.strip()
    if v == '' or v == '?':
        return None
    try:
        return float(v)
    except ValueError:
        return v

# ----------------------------------------------------------------------
# Composite score helpers
# ----------------------------------------------------------------------
def compute_min_max(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return (0.0, 0.0)
    return (min(vals), max(vals))

def normalize(val, mn, mx):
    if mn is None or mx is None or mx == mn:
        return 0.5
    return (val - mn) / (mx - mn)

def compute_composite_scores(rows, header, k_min, k_max, e_min, e_max, d_min, d_max):
    """Return dict row_num->score or None if any goal missing."""
    scores = {}
    for r in rows:
        idx_k = header.index('Kloc+')
        idx_e = header.index('Effort-')
        idx_d = header.index('Defects-')
        kv = parse_value(r[idx_k])
        ev = parse_value(r[idx_e])
        dv = parse_value(r[idx_d])
        if kv is None or ev is None or dv is None:
            scores[r['_rownum']] = None
            continue
        nk = normalize(kv, k_min, k_max)
        ne = normalize(ev, e_min, e_max)
        nd = normalize(dv, d_min, d_max)
        scores[r['_rownum']] = nk + ne + nd
    return scores

# ----------------------------------------------------------------------
# Loading CSV with row numbers
# ----------------------------------------------------------------------
def load_csv_with_rownums(path):
    header = []
    rows = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, r in enumerate(reader, start=1):
            row = {}
            for col, val in zip(header, r):
                row[col] = parse_value(val)
            row['_rownum'] = i
            rows.append(row)
    return header, rows

# ----------------------------------------------------------------------
# NASA-19
# ----------------------------------------------------------------------
def nasa_19(csv_path):
    header, rows = load_csv_with_rownums(csv_path)

    # Identify held-out rows
    held_out = [r for r in rows if r['_rownum'] % 5 == 0]
    history = [r for r in rows if r['_rownum'] % 5 != 0]

    # Prepare feature columns: all except X and target
    target_effort = 'Effort-'
    feature_cols = [c for c in header if not c.endswith('X') and c != target_effort]
    feature_indices = [header.index(c) for c in feature_cols]

    # Build X matrix and y vector
    X_all = []
    y_all = []
    for r in rows:
        vec = [r[c] for c in feature_cols]
        X_all.append(vec)
        y_all.append(r[target_effort])

    # Build transformer
    numeric_cols = []
    categorical_cols = []
    for c in feature_cols:
        for v in X_all:
            if v[feature_cols.index(c)] is not None and isinstance(v[feature_cols.index(c)], float):
                numeric_cols.append(c)
                break
        else:
            categorical_cols.append(c)

    numeric_indices = [feature_cols.index(c) for c in numeric_cols]
    categorical_indices = [feature_cols.index(c) for c in categorical_cols]

    numeric_tf = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_tf = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    ct = ColumnTransformer([
        ('num', numeric_tf, numeric_indices),
        ('cat', categorical_tf, categorical_indices)
    ])

    # Train model on history rows with known Effort-
    train_indices = [i for i, r in enumerate(history) if r[target_effort] is not None]
    X_train = [X_all[history[i]['_rownum']-1] for i in train_indices]
    y_train = [y_all[history[i]['_rownum']-1] for i in train_indices]

    X_train_tf = ct.fit_transform(X_train)
    model = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    model.fit(X_train_tf, y_train)

    # Predict for all held‑out rows
    predictions = {}
    actuals = []
    preds_for_mae = []
    for r in held_out:
        vec = [r[c] for c in feature_cols]
        vec_tf = ct.transform([vec])
        pred = float(model.predict(vec_tf)[0])
        predictions[str(r['_rownum'])] = pred
        if r[target_effort] is not None:
            actuals.append(r[target_effort])
            preds_for_mae.append(pred)

    # Baseline mean on history Effort-
    baseline_vals = [r[target_effort] for r in history if r[target_effort] is not None]
    baseline_mean = sum(baseline_vals) / len(baseline_vals) if baseline_vals else 0.0

    # Compute MAEs
    mae_model = mean_absolute_error(actuals, preds_for_mae) if actuals else 0.0
    mae_baseline = mean_absolute_error(actuals, [baseline_mean]*len(actuals)) if actuals else 0.0

    # Verdict
    eps = 1e-3
    if mae_model < mae_baseline - eps:
        verdict = "better"
    elif mae_model > mae_baseline + eps:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    # Output
    print("== NASA-19 ==")
    print(f"Baseline mean Effort-: {baseline_mean:.3f}")
    print(f"Model MAE: {mae_model:.3f}")
    print(f"Baseline MAE: {mae_baseline:.3f}")
    print("Answer:")
    answer = {
        "predictions": predictions,
        "baseline": baseline_mean,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict
    }
    print("ANSWER NASA-19 " + json.dumps(answer, separators=(',', ':')))

# ----------------------------------------------------------------------
# NASA-20
# ----------------------------------------------------------------------
def nasa_20(csv_path):
    header, rows = load_csv_with_rownums(csv_path)

    # Compute global min/max for goals
    k_vals = [r['Kloc+'] for r in rows if r['Kloc+'] is not None]
    e_vals = [r['Effort-'] for r in rows if r['Effort-'] is not None]
    d_vals = [r['Defects-'] for r in rows if r['Defects-'] is not None]
    k_min, k_max = compute_min_max(k_vals)
    e_min, e_max = compute_min_max(e_vals)
    d_min, d_max = compute_min_max(d_vals)

    # Determine arrivals
    total_rows = len(rows)
    arrivals_count = round(0.2 * total_rows)
    arrivals = rows[-arrivals_count:]
    arrivals_set = {r['_rownum'] for r in arrivals}
    # Compute composite scores for all rows
    score_dict = compute_composite_scores(rows, header, k_min, k_max, e_min, e_max, d_min, d_max)

    # Filter arrivals that have a score
    arrivals_with_scores = [r for r in arrivals if score_dict[r['_rownum']] is not None]
    if not arrivals_with_scores:
        print("== NASA-20 ==")
        print("No arrivals with complete goal data.")
        print("ANSWER NASA-20 " + json.dumps({"budget": 0, "repeats": 0, "ours": [], "random": [], "verdict": "no clear difference"}, separators=(',', ':')))
        return

    budget = min(30, total_rows // 5)
    repeats = 20
    ours_best_rows = []
    random_best_rows = []
    ours_wins = 0
    random_wins = 0

    # Precompute sorted arrivals by score (descending)
    arrivals_sorted = sorted(arrivals_with_scores, key=lambda r: score_dict[r['_rownum']], reverse=True)

    # Random selection
    random.seed(0)
    for _ in range(repeats):
        # Our selection: top budget rows
        ours_sel = arrivals_sorted[:budget]
        ours_best = ours_sel[0]['_rownum']
        ours_best_rows.append(ours_best)

        # Random selection
        random_sel = random.sample(arrivals_with_scores, min(budget, len(arrivals_with_scores)))
        # find best among random
        rand_best = max(random_sel, key=lambda r: score_dict[r['_rownum']])['_rownum']
        random_best_rows.append(rand_best)

        # Compare scores
        ours_score = score_dict[ours_best]
        rand_score = score_dict[rand_best]
        if ours_score > rand_score:
            ours_wins += 1
        elif rand_score > ours_score:
            random_wins += 1
        # ties count as neither

    if ours_wins > random_wins:
        verdict = "ours better"
    elif random_wins > ours_wins:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    print("== NASA-20 ==")
    print(f"Budget: {budget}")
    print(f"Repeats: {repeats}")
    print(f"Our best rows (first 3): {ours_best_rows[:3]}")
    print(f"Random best rows (first 3): {random_best_rows[:3]}")
    print(f"Ours wins: {ours_wins}, Random wins: {random_wins}")
    answer = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best_rows,
        "random": random_best_rows,
        "verdict": verdict
    }
    print("ANSWER NASA-20 " + json.dumps(answer, separators=(',', ':')))

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    nasa_19(csv_path)
    nasa_20(csv_path)

if __name__ == "__main__":
    main()
