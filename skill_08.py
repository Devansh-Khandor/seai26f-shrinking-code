#!/usr/bin/env python3
"""
skill_08.py

Implements NASA-15 and NASA-16 analyses for the NASA project dataset.

NASA-15: Suggest plausible changes to reduce predicted effort for the
         most effort‑intensive held‑out proposal.

NASA-16: Suggest at most three plausible changes to reduce predicted
         defects for the most defect‑intensive held‑out proposal,
         while showing the effort trade‑off.

Usage:
    python skill_08.py <csv_path>
"""

import sys
import csv
import json
import math
import numpy as np
from collections import defaultdict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def parse_value(v: str):
    """Return numeric value or original string if non‑numeric."""
    v = v.strip()
    if v == '' or v == '?':
        return None
    try:
        return float(v)
    except ValueError:
        return v  # keep original string for categorical


def load_csv(path):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader]
    return header, rows


def get_feature_indices(header, target_cols):
    """Return indices of feature columns (exclude X and goal columns)."""
    idx = {name: i for i, name in enumerate(header)}
    feature_cols = [
        c for c in header
        if not c.endswith('X') and c not in target_cols
    ]
    feature_indices = [idx[c] for c in feature_cols]
    return feature_cols, feature_indices, idx


def build_transformer(feature_cols, rows, feature_indices):
    """Return a fitted ColumnTransformer."""
    # Determine numeric vs categorical per column
    numeric_cols = []
    categorical_cols = []

    for col in feature_cols:
        col_idx = feature_indices[feature_cols.index(col)]
        for r in rows:
            v = parse_value(r[col_idx])
            if v is not None and isinstance(v, float):
                numeric_cols.append(col)
                break
        else:
            categorical_cols.append(col)

    num_indices = [feature_cols.index(c) for c in numeric_cols]
    cat_indices = [feature_cols.index(c) for c in categorical_cols]

    numeric_tf = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_tf = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    ct = ColumnTransformer([
        ('num', numeric_tf, num_indices),
        ('cat', categorical_tf, cat_indices)
    ])
    return ct, numeric_cols, categorical_cols


def build_features(rows, feature_indices):
    """Return list of feature vectors parsed for training."""
    X = []
    for r in rows:
        vec = [parse_value(r[i]) for i in feature_indices]
        X.append(vec)
    return X


def train_model(X, y):
    """Train RandomForestRegressor and return model."""
    model = RandomForestRegressor(
        n_estimators=200, random_state=0, n_jobs=-1
    )
    model.fit(X, y)
    return model


def predict_row(row_vec, ct, model):
    """Predict using trained transformer and model."""
    vec = [row_vec]  # transformer expects list
    X_t = ct.transform(vec)
    return model.predict(X_t)[0]


# ----------------------------------------------------------------------
# Main NASA‑15 & NASA‑16 logic
# ----------------------------------------------------------------------
def main(csv_path):
    header, rows = load_csv(csv_path)
    n_rows = len(rows)

    # Goal and target columns
    goal_cols = [c for c in header if c.endswith('+') or c.endswith('-')]
    target_effort = 'Effort-'
    target_defects = 'Defects-'

    # Feature columns
    feature_cols, feature_indices, idx = get_feature_indices(
        header, [target_effort, target_defects]
    )

    # Build transformer and feature matrix
    ct, num_cols, cat_cols = build_transformer(feature_cols, rows, feature_indices)
    X_all = build_features(rows, feature_indices)

    # Parse targets
    y_effort = [parse_value(r[idx[target_effort]]) for r in rows]
    y_defects = [parse_value(r[idx[target_defects]]) for r in rows]

    # Identify held‑out rows (index divisible by 5)
    held_out_indices = [i for i in range(n_rows) if (i + 1) % 5 == 0]
    history_indices = [i for i in range(n_rows) if (i + 1) % 5 != 0]

    # Fit models on history only
    X_train_eff = [X_all[i] for i in history_indices]
    y_train_eff = [y_effort[i] for i in history_indices]
    X_train_def = [X_all[i] for i in history_indices]
    y_train_def = [y_defects[i] for i in history_indices]

    # Fit transformer on training data
    ct.fit(X_train_eff)

    # Train models
    model_eff = train_model([ct.transform([x])[0] for x in X_train_eff], y_train_eff)
    model_def = train_model([ct.transform([x])[0] for x in X_train_def], y_train_def)

    # Predict held‑out rows
    preds_eff = {}
    preds_def = {}
    for i in held_out_indices:
        vec = X_all[i]
        pred_eff = predict_row(vec, ct, model_eff)
        pred_def = predict_row(vec, ct, model_def)
        preds_eff[i] = pred_eff
        preds_def[i] = pred_def

    # --- NASA-15: excessive effort ---
    # Choose held‑out row with highest predicted effort
    project_idx = max(preds_eff, key=preds_eff.get)
    project_row_num = project_idx + 1
    effort_before = preds_eff[project_idx]

    # Find adjustable columns
    adjustable = [c for c in feature_cols if c != 'Kloc+' and not c.endswith('X')]

    best_change = None
    best_effort_after = effort_before
    # For each adjustable column, try all distinct values
    for col in adjustable:
        col_idx = feature_indices[feature_cols.index(col)]
        current_raw = rows[project_idx][col_idx]
        current_parsed = parse_value(current_raw)
        # collect all distinct values from this column
        unique_vals = set(r[col_idx] for r in rows if r[col_idx] not in ('', '?'))
        for new_raw in unique_vals:
            if new_raw == current_raw:
                continue
            # Build new feature vector
            new_vec = X_all[project_idx][:]  # shallow copy
            new_vec[feature_cols.index(col)] = parse_value(new_raw)
            effort_after = predict_row(new_vec, ct, model_eff)
            if effort_after < best_effort_after:
                best_effort_after = effort_after
                best_change = (col, current_raw, new_raw)

    if best_change is None:
        # No beneficial change found; just keep current
        best_change = (adjustable[0], rows[project_idx][feature_indices[feature_cols.index(adjustable[0])]],
                       rows[project_idx][feature_indices[feature_cols.index(adjustable[0])]])
        best_effort_after = effort_before

    change_dict = {best_change[0]: [best_change[1], best_change[2]]}

    # Output NASA‑15
    print("== NASA-15 ==")
    print(f"Project {project_row_num}: Effort before = {effort_before:.3f}, after = {best_effort_after:.3f}")
    print(f"Change: {best_change[0]} from {best_change[1]} to {best_change[2]}")
    nasa15_json = {
        "project": project_row_num,
        "changes": change_dict,
        "predicted_effort": {"before": effort_before, "after": best_effort_after}
    }
    print("ANSWER NASA-15 " + json.dumps(nasa15_json, separators=(',', ':')))

    # --- NASA-16: high defects ---
    # Choose held‑out row with highest predicted defects
    defect_project_idx = max(preds_def, key=preds_def.get)
    defect_project_row_num = defect_project_idx + 1
    defects_before = preds_def[defect_project_idx]
    effort_before_def = preds_eff[defect_project_idx]

    best_changes = []
    best_defects_after = defects_before
    best_effort_after_def = effort_before_def
    # Try at most one change per column, keep best
    for col in adjustable:
        col_idx = feature_indices[feature_cols.index(col)]
        current_raw = rows[defect_project_idx][col_idx]
        # collect unique values
        unique_vals = set(r[col_idx] for r in rows if r[col_idx] not in ('', '?'))
        for new_raw in unique_vals:
            if new_raw == current_raw:
                continue
            new_vec = X_all[defect_project_idx][:]  # copy
            new_vec[feature_cols.index(col)] = parse_value(new_raw)
            defects_after = predict_row(new_vec, ct, model_def)
            effort_after = predict_row(new_vec, ct, model_eff)
            # prioritize defect reduction; if same, choose minimal effort increase
            if defects_after < best_defects_after:
                best_defects_after = defects_after
                best_effort_after_def = effort_after
                best_changes = [(col, current_raw, new_raw)]
            elif defects_after == best_defects_after:
                # if same reduction, prefer lower effort increase
                if effort_after < best_effort_after_def:
                    best_effort_after_def = effort_after
                    best_changes = [(col, current_raw, new_raw)]

    # Keep at most three changes
    if not best_changes:
        # No beneficial change found; keep current
        best_changes = [(adjustable[0], rows[defect_project_idx][feature_indices[feature_cols.index(adjustable[0])]],
                         rows[defect_project_idx][feature_indices[feature_cols.index(adjustable[0])]])]
    changes_dict = {c[0]: [c[1], c[2]] for c in best_changes[:3]}

    # Output NASA‑16
    print("\n== NASA-16 ==")
    print(f"Project {defect_project_row_num}: Defects before = {defects_before:.3f}, after = {best_defects_after:.3f}")
    print(f"Effort before = {effort_before_def:.3f}, after = {best_effort_after_def:.3f}")
    print(f"Changes: {changes_dict}")
    nasa16_json = {
        "project": defect_project_row_num,
        "changes": changes_dict,
        "predicted_defects": {"before": defects_before, "after": best_defects_after},
        "predicted_effort": {"before": effort_before_def, "after": best_effort_after_def}
    }
    print("ANSWER NASA-16 " + json.dumps(nasa16_json, separators=(',', ':')))


# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
