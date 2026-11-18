#!/usr/bin/env python3
"""
Skill 09: NASA-17 & NASA-18 outlier & anomaly detection

Usage:
    python skill_09.py <path-to-nasa93dem.csv>
"""

import sys
import json
import math

# Import utilities from skill_01
try:
    from skill_01 import read_csv, format_number
except ImportError:
    # Minimal reimplementation if skill_01 is not available
    import csv
    def read_csv(path):
        rows = []
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = {}
                for k, v in row.items():
                    if v.strip() == '?':
                        parsed[k] = None
                    else:
                        try:
                            parsed[k] = float(v)
                        except ValueError:
                            parsed[k] = v
                rows.append(parsed)
        return rows

    def format_number(x):
        if x is None:
            return "null"
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        return f"{x:.6f}".rstrip('0').rstrip('.')

# ----------------------------------------------------------------------
# Helper functions for feature handling
# ----------------------------------------------------------------------
def is_numeric_column(rows, col):
    """Return True if every non‑None value in col can be interpreted as float."""
    for r in rows:
        v = r[col]
        if v is None:
            continue
        if isinstance(v, float):
            continue
        try:
            float(v)
        except Exception:
            return False
    return True

def get_numeric_cols(rows):
    """Return list of non‑goal, non‑X columns that are numeric."""
    cols = list(rows[0].keys())
    return [c for c in cols
            if not (c.endswith('+') or c.endswith('-')) and not c.endswith('X')
            and is_numeric_column(rows, c)]

def mean_std(values):
    n = len(values)
    if n == 0:
        return (None, None)
    m = sum(values) / n
    var = sum((x - m) ** 2 for x in values) / n
    s = math.sqrt(var)
    return (m, s)

# ----------------------------------------------------------------------
# NASA-17: flag unusual input‑feature combinations
# ----------------------------------------------------------------------
def nasa_17(rows):
    numeric_cols = get_numeric_cols(rows)

    # Compute mean and std for each numeric column
    col_stats = {}
    for col in numeric_cols:
        vals = [r[col] for r in rows if r[col] is not None]
        m, s = mean_std(vals)
        col_stats[col] = (m, s if s > 0 else None)  # avoid zero std

    # Compute unusualness score for each row
    scores = {}
    for idx, r in enumerate(rows, start=1):
        score = 0.0
        count = 0
        for col in numeric_cols:
            m, s = col_stats[col]
            if s is None or r[col] is None:
                continue
            diff = (r[col] - m) / s
            score += diff * diff
            count += 1
        # If row had no usable numeric data, score remains 0
        scores[idx] = score

    # Flag top 10 rows with highest scores
    flagged = sorted(scores.items(), key=lambda t: t[1], reverse=True)[:10]
    flagged_rows = [i for i, _ in flagged]
    flagged_scores = {str(i): round(scores[i], 6) for i, _ in flagged}

    # Results output
    print("== NASA-17 ==")
    print(f"Flagged rows (up to 10): {flagged_rows}")
    print("Unusualness scores:")
    for i, s in flagged:
        print(f"  Row {i}: {round(scores[i], 6)}")
    json_obj = {"flagged": flagged_rows, "scores": flagged_scores}
    json_str = json.dumps(json_obj, separators=(',', ':'))
    print(f"ANSWER NASA-17 {json_str}")

# ----------------------------------------------------------------------
# Helper functions for prediction (used by NASA-18)
# ----------------------------------------------------------------------
def determine_feature_types(rows, feature_cols):
    numeric_cols = []
    categorical_cols = []
    for col in feature_cols:
        is_num = True
        for r in rows:
            v = r[col]
            if v is None:
                continue
            if isinstance(v, float):
                continue
            try:
                float(v)
            except Exception:
                is_num = False
                break
        if is_num:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols

def build_encodings(rows, categorical_cols):
    encodings = {}
    for col in categorical_cols:
        mapping = {}
        code = 0
        for r in rows:
            v = r[col]
            if v is None:
                continue
            if v not in mapping:
                mapping[v] = code
                code += 1
        encodings[col] = mapping
    return encodings

def build_feature_matrix(rows, feature_cols, numeric_cols, categorical_cols, encodings):
    X = []
    for r in rows:
        vec = []
        for col in feature_cols:
            v = r[col]
            if v is None:
                vec.append(0.0)
                continue
            if col in numeric_cols:
                vec.append(float(v))
            else:
                vec.append(encodings[col][v])
        X.append(vec)
    return np.array(X, dtype=float)

import numpy as np

def train_regressor(rows, feature_cols, target_col):
    train_rows = [r for r in rows if r[target_col] is not None and all(r[c] is not None for c in feature_cols)]
    if not train_rows:
        return None, None, None, None, None
    numeric_cols, categorical_cols = determine_feature_types(train_rows, feature_cols)
    encodings = build_encodings(train_rows, categorical_cols)
    X_train = build_feature_matrix(train_rows, feature_cols, numeric_cols, categorical_cols, encodings)
    y_train = np.array([r[target_col] for r in train_rows], dtype=float)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, numeric_cols, categorical_cols, encodings, train_rows

def predict_row(row, model, numeric_cols, categorical_cols, encodings, feature_cols):
    vec = []
    for col in feature_cols:
        v = row[col]
        if v is None:
            return None
        if col in numeric_cols:
            vec.append(float(v))
        else:
            if v not in encodings[col]:
                return None
            vec.append(encodings[col][v])
    return float(model.predict(np.array([vec], dtype=float))[0])

# ----------------------------------------------------------------------
# NASA-18: flag projects with large prediction errors
# ----------------------------------------------------------------------
def nasa_18(rows):
    # Feature columns are all except goals and X
    all_cols = list(rows[0].keys())
    feature_cols = [c for c in all_cols if not (c.endswith('+') or c.endswith('-')) and not c.endswith('X')]

    # Train models for Effort- and Defects-
    model_eff, num_eff, cat_eff, enc_eff, train_eff_rows = train_regressor(rows, feature_cols, 'Effort-')
    model_def, num_def, cat_def, enc_def, train_def_rows = train_regressor(rows, feature_cols, 'Defects-')

    # If models not available, cannot proceed
    if model_eff is None or model_def is None:
        print("== NASA-18 ==")
        print("Insufficient data for prediction.")
        print("ANSWER NASA-18 {}")
        return

    # Compute residuals on training sets to determine sigma
    def compute_sigma(model, train_rows):
        if not train_rows:
            return 0.0
        X_train = build_feature_matrix(train_rows, feature_cols,
                                       num_eff if train_rows==train_eff_rows else num_def,
                                       cat_eff if train_rows==train_eff_rows else cat_def,
                                       enc_eff if train_rows==train_eff_rows else enc_def)
        y_train = np.array([r['Effort-'] if train_rows==train_eff_rows else r['Defects-']
                            for r in train_rows], dtype=float)
        preds = model.predict(X_train)
        residuals = y_train - preds
        var = np.mean(residuals ** 2)
        return math.sqrt(var)

    sigma_eff = compute_sigma(model_eff, train_eff_rows)
    sigma_def = compute_sigma(model_def, train_def_rows)
    thresh_eff = 2 * sigma_eff
    thresh_def = 2 * sigma_def

    flagged = []

    # Evaluate held‑out rows (index divisible by 5)
    for idx, r in enumerate(rows, start=1):
        if idx % 5 != 0:
            continue
        # Effort- prediction
        eff_pred = predict_row(r, model_eff, num_eff, cat_eff, enc_eff, feature_cols)
        if eff_pred is not None:
            err = abs(eff_pred - r['Effort-'])
            if err > thresh_eff:
                flagged.append({"row": idx, "column": "Effort-", "actual": r['Effort-'], "expected": round(eff_pred, 6)})

        # Defects- prediction
        def_pred = predict_row(r, model_def, num_def, cat_def, enc_def, feature_cols)
        if def_pred is not None:
            err = abs(def_pred - r['Defects-'])
            if err > thresh_def:
                flagged.append({"row": idx, "column": "Defects-", "actual": r['Defects-'], "expected": round(def_pred, 6)})

    # If no row exceeded thresholds, pick top 10 by absolute error
    if not flagged:
        errors = []
        for idx, r in enumerate(rows, start=1):
            if idx % 5 != 0:
                continue
            eff_pred = predict_row(r, model_eff, num_eff, cat_eff, enc_eff, feature_cols)
            def_pred = predict_row(r, model_def, num_def, cat_def, enc_def, feature_cols)
            if eff_pred is not None:
                errors.append({"row": idx, "column": "Effort-", "actual": r['Effort-'],
                               "expected": round(eff_pred, 6),
                               "error": abs(eff_pred - r['Effort-'])})
            if def_pred is not None:
                errors.append({"row": idx, "column": "Defects-", "actual": r['Defects-'],
                               "expected": round(def_pred, 6),
                               "error": abs(def_pred - r['Defects-'])})
        # Sort by error descending
        errors.sort(key=lambda t: t["error"], reverse=True)
        flagged = [{k: v for k, v in e.items() if k != "error"} for e in errors[:10]]

    # Results
    print("== NASA-18 ==")
    print(f"Flagged anomalies (up to 10): {flagged}")
    json_obj = {"flagged": flagged}
    json_str = json.dumps(json_obj, separators=(',', ':'))
    print(f"ANSWER NASA-18 {json_str}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)
    path = sys.argv[1]
    rows = read_csv(path)
    nasa_17(rows)
    nasa_18(rows)

if __name__ == "__main__":
    main()
