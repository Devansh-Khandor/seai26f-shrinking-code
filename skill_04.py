#!/usr/bin/env python3
"""
skill_04.py

Computes NASA-07 and NASA-08 analyses on a NASA project CSV dataset.

Usage:
    python skill_04.py <csv_path>
"""

import sys
import csv
import json
import math
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

def parse_value(v: str):
    """Parse a numeric cell, returning None for missing or non-numeric."""
    v = v.strip()
    if v == '' or v == '?':
        return None
    try:
        return float(v)
    except ValueError:
        return None

def round3(x):
    """Round to 3 decimal places."""
    return round(float(x), 3)

def load_csv(csv_path):
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows

def get_feature_columns(header, target_cols):
    """Return feature columns (exclude targets and X columns)."""
    feature_cols = []
    for c in header:
        if c.endswith('X') or c in target_cols:
            continue
        feature_cols.append(c)
    return feature_cols

def is_numeric_column(rows, col_idx):
    """Determine if a column is numeric by inspecting the first non-missing value."""
    for r in rows:
        val = parse_value(r[col_idx])
        if val is not None:
            return True
    return False

def build_transformer(feature_cols, rows):
    """Return a fitted ColumnTransformer for the given feature columns."""
    numeric_cols = []
    categorical_cols = []
    for idx, col in enumerate(feature_cols):
        if is_numeric_column(rows, idx):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

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
    return ct

def main(csv_path):
    header, rows = load_csv(csv_path)
    total_rows = len(rows)

    # Goal columns (end with '+' or '-')
    goal_cols = [c for c in header if c.endswith('+') or c.endswith('-')]

    target_effort = 'Effort-'
    target_defects = 'Defects-'
    target_cols = [target_effort, target_defects]

    # Determine feature columns
    feature_cols = get_feature_columns(header, target_cols)

    # Indices for columns
    col_index = {c: i for i, c in enumerate(header)}

    # -------- NASA-07 ----------
    # Determine arrivals and history
    arrivals_count = round(0.2 * total_rows)
    history_rows = total_rows - arrivals_count
    arrival_indices = list(range(history_rows, total_rows))
    history_indices = list(range(history_rows))

    # Prepare data structures
    predictions = []  # list of (effort_pred, defects_pred)
    for k, idx_row in enumerate(arrival_indices):
        # Build training data from history + earlier arrivals
        training_indices = [i for i in history_indices + arrival_indices[:k]
                           if parse_value(rows[i][col_index[target_effort]]) is not None
                           and parse_value(rows[i][col_index[target_defects]]) is not None]

        X_train = []
        y_train_effort = []
        y_train_defects = []
        for i in training_indices:
            feats = [parse_value(rows[i][c]) for c in feature_cols]
            X_train.append(feats)
            y_train_effort.append(parse_value(rows[i][col_index[target_effort]]))
            y_train_defects.append(parse_value(rows[i][col_index[target_defects]]))

        if not X_train:
            # No training data; cannot predict
            eff_pred = 0.0
            def_pred = 0.0
            predictions.append((eff_pred, def_pred))
            continue

        # Build transformer and fit
        ct = build_transformer(feature_cols, X_train)
        X_train_tf = ct.fit_transform(X_train)

        # Train models
        rf_eff = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
        rf_def = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
        rf_eff.fit(X_train_tf, y_train_effort)
        rf_def.fit(X_train_tf, y_train_defects)

        # Predict for current arrival row
        feats_row = [parse_value(rows[idx_row][c]) for c in feature_cols]
        X_row_tf = ct.transform([feats_row])
        eff_pred = float(rf_eff.predict(X_row_tf)[0])
        def_pred = float(rf_def.predict(X_row_tf)[0])

        predictions.append((eff_pred, def_pred))

    # Compute thresholds for alerts
    eff_vals = [p[0] for p in predictions]
    def_vals = [p[1] for p in predictions]
    if len(eff_vals) > 1:
        eff_mean = sum(eff_vals) / len(eff_vals)
        eff_std = np.std(eff_vals, ddof=1)
    else:
        eff_mean, eff_std = eff_vals[0], 0.0
    if len(def_vals) > 1:
        def_mean = sum(def_vals) / len(def_vals)
        def_std = np.std(def_vals, ddof=1)
    else:
        def_mean, def_std = def_vals[0], 0.0
    eff_thresh = eff_mean + 2 * eff_std
    def_thresh = def_mean + 2 * def_std

    # Prepare assessments list
    assessments = []
    for i, (eff_pred, def_pred) in enumerate(predictions):
        row_num = arrival_indices[i] + 1  # 1-based
        alert = (eff_pred > eff_thresh) or (def_pred > def_thresh)
        assessments.append({
            "row": row_num,
            "predicted_effort": round3(eff_pred),
            "predicted_defects": round3(def_pred),
            "alert": alert
        })

    # Print results for NASA-07
    print("== NASA-07 ==")
    # Show at least three lines of results (first three arrivals)
    for assess in assessments[:3]:
        print(f"Row {assess['row']}: Effort={assess['predicted_effort']:.3f} "
              f"Defects={assess['predicted_defects']:.3f} Alert={assess['alert']}")
    print("ANSWER NASA-07 " + json.dumps({"assessments": assessments}, separators=(',', ':')))

    # -------- NASA-08 ----------
    checkpoints = []
    for m in [0, 9, 19]:
        # Number of arrivals to include
        m = min(m, arrivals_count)
        checkpoint_rows = rows[:history_rows + m]

        # Summary statistics for each goal column
        summary = {}
        for col in goal_cols:
            col_idx = col_index[col]
            vals = [parse_value(r[col_idx]) for r in checkpoint_rows if parse_value(r[col_idx]) is not None]
            if vals:
                n = len(vals)
                mean = sum(vals) / n
                mn = min(vals)
                mx = max(vals)
                summary[col] = {
                    "n": n,
                    "mean": round3(mean),
                    "min": round3(mn),
                    "max": round3(mx)
                }
            else:
                # If no data, set to nulls (will appear as null in JSON)
                summary[col] = {"n": 0, "mean": None, "min": None, "max": None}

        # Shortlist: top 5 rows based on composite score
        # Collect rows with all three goals present
        subset_rows = []
        for idx, r in enumerate(checkpoint_rows, start=1):
            vals = {}
            missing = False
            for g in ['Kloc+', 'Effort-', 'Defects-']:
                val = parse_value(r[col_index[g]])
                if val is None:
                    missing = True
                    break
                vals[g] = val
            if not missing:
                subset_rows.append((idx, vals))

        # If there are no rows with full data, shortlist empty
        if subset_rows:
            # Compute min/max for each goal in this subset
            k_vals = [v['Kloc+'] for _, v in subset_rows]
            e_vals = [v['Effort-'] for _, v in subset_rows]
            d_vals = [v['Defects-'] for _, v in subset_rows]
            k_min, k_max = min(k_vals), max(k_vals)
            e_min, e_max = min(e_vals), max(e_vals)
            d_min, d_max = min(d_vals), max(d_vals)

            scored = []
            for idx, v in subset_rows:
                # Normalize
                nk = 0.5 if k_max == k_min else (v['Kloc+'] - k_min) / (k_max - k_min)
                ne = 0.5 if e_max == e_min else (e_max - v['Effort-']) / (e_max - e_min)
                nd = 0.5 if d_max == d_min else (d_max - v['Defects-']) / (d_max - d_min)
                score = nk + ne + nd
                scored.append((score, idx))

            scored.sort(reverse=True)
            shortlist = [idx for _, idx in scored[:5]]
        else:
            shortlist = []

        checkpoints.append({
            "after": m,
            "summary": summary,
            "shortlist": shortlist
        })

    # Print at least three lines of results for NASA-08
    print("\n== NASA-08 ==")
    for cp in checkpoints:
        after = cp["after"]
        best = cp["shortlist"][0] if cp["shortlist"] else None
        print(f"Checkpoint after {after}: Best={best}")

    print("ANSWER NASA-08 " + json.dumps({"checkpoints": checkpoints}, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_04.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
