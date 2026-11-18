#!/usr/bin/env python3
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

def parse_value(v):
    v = v.strip()
    if v == '?' or v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None

def main(csv_path):
    # Read CSV
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    n_rows = len(rows)  # number of data rows
    # Determine indices for columns
    idx = {name: i for i, name in enumerate(header)}

    # Identify feature columns: all except target and X columns
    target_effort = 'Effort-'
    target_defects = 'Defects-'
    exclude = set([target_effort, target_defects])
    # Exclude columns ending with X
    exclude.update([c for c in header if c.endswith('X')])
    feature_cols = [c for c in header if c not in exclude]
    feature_indices = [idx[c] for c in feature_cols]

    # Determine numeric vs categorical feature columns
    is_numeric = []
    for col in feature_cols:
        col_idx = idx[col]
        num = False
        for r in rows:
            v = parse_value(r[col_idx])
            if v is not None:
                num = True
                break
        is_numeric.append(num)

    numeric_cols = [c for c, isnum in zip(feature_cols, is_numeric) if isnum]
    categorical_cols = [c for c, isnum in zip(feature_cols, is_numeric) if not isnum]

    numeric_indices = [feature_indices[feature_cols.index(c)] for c in numeric_cols]
    categorical_indices = [feature_indices[feature_cols.index(c)] for c in categorical_cols]

    # Build feature matrix (list of lists). Missing values represented as None.
    X_all = []
    y_effort_raw = []
    y_defects_raw = []
    for r in rows:
        row_feats = []
        for i in feature_indices:
            val = parse_value(r[i])
            if val is None:
                # keep None; imputer will handle
                row_feats.append(None)
            else:
                row_feats.append(val)
        X_all.append(row_feats)

        # Targets
        y_effort_raw.append(parse_value(r[idx[target_effort]]))
        y_defects_raw.append(parse_value(r[idx[target_defects]]))

    # Determine held‑out rows: those with 1‑based number divisible by 5
    held_out_indices = [i for i in range(n_rows) if (i+1) % 5 == 0]
    history_indices = [i for i in range(n_rows) if (i+1) % 5 != 0]

    # Prepare ColumnTransformer
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

    # ---------- NASA‑03: Effort prediction ----------
    # Build training data for Effort
    train_eff_indices = [i for i in history_indices if y_effort_raw[i] is not None]
    X_train_eff = [X_all[i] for i in train_eff_indices]
    y_train_eff = [y_effort_raw[i] for i in train_eff_indices]

    # Transform features
    X_train_eff_tf = ct.fit_transform(X_train_eff)

    # Train model
    rf_eff = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    rf_eff.fit(X_train_eff_tf, y_train_eff)

    # Predict on held‑out rows
    X_holdout_eff = [X_all[i] for i in held_out_indices]
    X_holdout_eff_tf = ct.transform(X_holdout_eff)
    y_pred_eff = rf_eff.predict(X_holdout_eff_tf)

    # Compute MAE
    y_true_eff = [y_effort_raw[i] for i in held_out_indices]
    mae_eff = np.mean(np.abs(np.array(y_pred_eff) - np.array(y_true_eff)))

    # Output NASA‑03 section
    print("== NASA-03 ==")
    for idx_row, pred in zip(held_out_indices, y_pred_eff):
        print(f"Row {idx_row+1}: Effort- = {pred:.3f}")
    print(f"MAE = {mae_eff:.3f}")

    nasa03_json = {"predictions": {str(idx_row+1): float(pred) for idx_row, pred in zip(held_out_indices, y_pred_eff)},
                   "mae": float(mae_eff)}
    print("ANSWER NASA-03 " + json.dumps(nasa03_json, separators=(',', ':')))

    # ---------- NASA‑04: Defects prediction ----------
    # Build training data for Defects
    train_def_indices = [i for i in history_indices if y_defects_raw[i] is not None]
    X_train_def = [X_all[i] for i in train_def_indices]
    y_train_def = [y_defects_raw[i] for i in train_def_indices]

    # Train model (use a fresh transformer or same as above? We'll reuse ct because fitted already)
    # For simplicity, use the same transformer (fitted on eff training). It may have seen all history rows.
    X_train_def_tf = ct.transform(X_train_def)

    rf_def = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    rf_def.fit(X_train_def_tf, y_train_def)

    # Predict on held‑out
    X_holdout_def_tf = ct.transform(X_holdout_eff)
    y_pred_def = rf_def.predict(X_holdout_def_tf)

    # Compute residual standard deviation on training set
    residuals = np.array(y_train_def) - rf_def.predict(X_train_def_tf)
    sigma = np.std(residuals, ddof=1)

    # Build predictions with interval
    predictions_def = {}
    for idx_row, pred in zip(held_out_indices, y_pred_def):
        low = pred - 1.96 * sigma
        high = pred + 1.96 * sigma
        # Ensure low not negative
        low = max(low, 0.0)
        predictions_def[str(idx_row+1)] = {"estimate": float(pred), "low": float(low), "high": float(high)}

    # Output NASA‑04 section
    print("\n== NASA-04 ==")
    for idx_row in held_out_indices:
        pred = predictions_def[str(idx_row+1)]
        print(f"Row {idx_row+1}: estimate={pred['estimate']:.3f} low={pred['low']:.3f} high={pred['high']:.3f}")

    nasa04_json = {"predictions": predictions_def}
    print("ANSWER NASA-04 " + json.dumps(nasa04_json, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
