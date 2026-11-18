#!/usr/bin/env python3
"""
skill_09.py

Implements NASA-17 and NASA-18 analyses for the NASA project dataset.

NASA-17: Flag historical projects whose combinations of project characteristics are unusually different
         from the rest of the records.

NASA-18: Identify project records whose reported Effort- or Defects- differ substantially from what
         comparable projects would lead us to expect, flagging potential measurement issues.
"""

import sys
import csv
import json
import math
import numpy as np
from collections import defaultdict, Counter
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# ----------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------
def parse_value(v: str):
    """Parse a cell value: float if possible, otherwise keep string; None for missing."""
    v = v.strip()
    if v == '' or v == '?':
        return None
    try:
        return float(v)
    except ValueError:
        return v

# ----------------------------------------------------------------------
# Feature extraction helpers
# ----------------------------------------------------------------------
def get_feature_columns(header, target_cols):
    """Return list of feature column names (exclude X and goal columns)."""
    feature_cols = []
    for c in header:
        if c.endswith('X') or c in target_cols:
            continue
        feature_cols.append(c)
    return feature_cols

def build_transformer(feature_cols, rows, feature_indices):
    """Build and fit a ColumnTransformer for given rows."""
    numeric_cols = []
    categorical_cols = []
    for col in feature_cols:
        col_idx = feature_indices[feature_cols.index(col)]
        for r in rows:
            val = parse_value(r[col_idx])
            if val is not None and isinstance(val, float):
                numeric_cols.append(col)
                break
        else:
            categorical_cols.append(col)

    num_idx = [feature_cols.index(c) for c in numeric_cols]
    cat_idx = [feature_cols.index(c) for c in categorical_cols]

    numeric_tf = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_tf = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    ct = ColumnTransformer([
        ('num', numeric_tf, num_idx),
        ('cat', categorical_tf, cat_idx)
    ])
    return ct, numeric_cols, categorical_cols

def build_feature_matrix(rows, feature_indices):
    """Return list of parsed feature vectors for all rows."""
    X = []
    for r in rows:
        vec = [parse_value(r[i]) for i in feature_indices]
        X.append(vec)
    return X

# ----------------------------------------------------------------------
# NASA-17: Unusual characteristic combinations
# ----------------------------------------------------------------------
def compute_numerical_stats(values):
    """Return mean and std for numeric values (excluding None)."""
    nums = [v for v in values if v is not None]
    if not nums:
        return (0.0, 0.0)
    mean = sum(nums) / len(nums)
    std = math.sqrt(sum((v - mean) ** 2 for v in nums) / (len(nums) - 1)) if len(nums) > 1 else 0.0
    return mean, std

def flag_unusual_combinations(header, rows):
    """Return list of flagged rows and their unusualness scores (top 10)."""
    target_cols = ['Effort-', 'Defects-']
    feature_cols = get_feature_columns(header, target_cols)
    feature_indices = [header.index(c) for c in feature_cols]

    # Compute numeric statistics per feature
    numeric_stats = {}
    for col in feature_cols:
        vals = [parse_value(r[header.index(col)]) for r in rows]
        # check if numeric
        if any(v is not None and isinstance(v, float) for v in vals):
            mean, std = compute_numerical_stats(vals)
            numeric_stats[col] = (mean, std)

    scores = {}
    for idx, r in enumerate(rows, start=1):
        z_sum = 0.0
        z_count = 0
        # numeric features
        for col, (mean, std) in numeric_stats.items():
            val = parse_value(r[header.index(col)])
            if val is not None and isinstance(val, float) and std > 0:
                z = abs((val - mean) / std)
                z_sum += z
                z_count += 1
        # categorical features: treat rarity as unusualness
        for col in feature_cols:
            if col in numeric_stats:
                continue
            val = r[header.index(col)]
            if val == '' or val == '?':
                continue
            # compute frequency
            freq = sum(1 for rr in rows if rr[header.index(col)] == val)
            if freq == 0:
                continue
            # unusualness contribution inversely proportional to frequency
            z_sum += math.log2(freq + 1)  # higher freq => larger log => less unusual
            z_count += 1
        # Normalize by number of features considered
        score = z_sum / z_count if z_count > 0 else 0.0
        scores[idx] = score

    # Select top 10 rows by score
    flagged = sorted(scores.items(), key=lambda x: -x[1])[:10]
    flagged_rows = [r for r, s in flagged]
    flagged_scores = {str(r): round(s, 3) for r, s in flagged}
    return flagged_rows, flagged_scores

# ----------------------------------------------------------------------
# NASA-18: Measurement anomalies
# ----------------------------------------------------------------------
def predict_and_flag(csv_path, header, rows):
    """Predict Effort- and Defects- for held-out rows and flag significant deviations."""
    target_effort = 'Effort-'
    target_defects = 'Defects-'
    target_cols = [target_effort, target_defects]
    feature_cols = get_feature_columns(header, target_cols)
    feature_indices = [header.index(c) for c in feature_cols]

    # Build feature matrix
    X_all = build_feature_matrix(rows, feature_indices)

    # Parse targets
    y_effort = [parse_value(r[header.index(target_effort)]) for r in rows]
    y_defects = [parse_value(r[header.index(target_defects)]) for r in rows]

    # Determine indices
    n_rows = len(rows)
    held_out_indices = [i for i in range(n_rows) if (i + 1) % 5 == 0]
    history_indices = [i for i in range(n_rows) if (i + 1) % 5 != 0]

    # Train transformer on history
    ct, num_cols, cat_cols = build_transformer(feature_cols, rows, feature_indices)
    ct.fit([X_all[i] for i in history_indices])

    # Train models
    model_eff = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    model_def = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    # transform training data
    X_train_eff = ct.transform([X_all[i] for i in history_indices])
    X_train_def = ct.transform([X_all[i] for i in history_indices])
    y_train_eff = [y_effort[i] for i in history_indices if y_effort[i] is not None]
    y_train_def = [y_defects[i] for i in history_indices if y_defects[i] is not None]
    # Filter to rows where target not missing
    train_eff_indices = [i for i in history_indices if y_effort[i] is not None]
    train_def_indices = [i for i in history_indices if y_defects[i] is not None]
    X_train_eff = ct.transform([X_all[i] for i in train_eff_indices])
    X_train_def = ct.transform([X_all[i] for i in train_def_indices])

    model_eff.fit(X_train_eff, [y_effort[i] for i in train_eff_indices])
    model_def.fit(X_train_def, [y_defects[i] for i in train_def_indices])

    # Compute residuals on training set for std calculation
    preds_eff_train = model_eff.predict(X_train_eff)
    preds_def_train = model_def.predict(X_train_def)
    residuals_eff = [y_effort[i] - preds_eff_train[j] for j, i in enumerate(train_eff_indices)]
    residuals_def = [y_defects[i] - preds_def_train[j] for j, i in enumerate(train_def_indices)]
    std_eff = np.std(residuals_eff, ddof=1) if residuals_eff else 0.0
    std_def = np.std(residuals_def, ddof=1) if residuals_def else 0.0

    flagged = []
    for i in held_out_indices:
        row_num = i + 1
        # Predict
        vec = X_all[i]
        try:
            pred_eff = model_eff.predict(ct.transform([vec]))[0]
            pred_def = model_def.predict(ct.transform([vec]))[0]
        except Exception:
            continue

        # Actual values
        act_eff = y_effort[i]
        act_def = y_defects[i]
        if act_eff is not None:
            if abs(act_eff - pred_eff) > 2 * std_eff:
                flagged.append({"row": row_num,
                               "column": target_effort,
                               "actual": round(act_eff, 3),
                               "expected": round(pred_eff, 3)})
        if act_def is not None:
            if abs(act_def - pred_def) > 2 * std_def:
                flagged.append({"row": row_num,
                               "column": target_defects,
                               "actual": round(act_def, 3),
                               "expected": round(pred_def, 3)})

    # Keep at most 10
    flagged = flagged[:10]
    return flagged

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path):
    # Read data
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader]

    # NASA-17
    flagged_rows, flagged_scores = flag_unusual_combinations(header, rows)
    print("== NASA-17 ==")
    print(f"Flagged {len(flagged_rows)} rows with high unusualness scores.")
    for r in flagged_rows[:3]:
        print(f"Row {r}: score={flagged_scores[str(r)]}")
    print("ANSWER NASA-17 " + json.dumps({"flagged": flagged_rows, "scores": flagged_scores}, separators=(',', ':')))

    # NASA-18
    flagged_measurements = predict_and_flag(csv_path, header, rows)
    print("\n== NASA-18 ==")
    print(f"Flagged {len(flagged_measurements)} measurement anomalies.")
    for m in flagged_measurements[:3]:
        print(f"Row {m['row']}: {m['column']} actual={m['actual']} expected={m['expected']}")
    print("ANSWER NASA-18 " + json.dumps({"flagged": flagged_measurements}, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
