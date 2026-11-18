#!/usr/bin/env python3
"""
Skill 08: NASA-15 & NASA-16 analysis

Usage:
    python skill_08.py <path-to-nasa93dem.csv>
"""

import sys
import json
import numpy as np
from sklearn.linear_model import LinearRegression

# Import helpers from previous modules
try:
    from skill_01 import read_csv, format_number
except ImportError:
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

try:
    from skill_06 import determine_feature_types, build_encodings, build_feature_matrix
except ImportError:
    # Minimal fallback (identical to skill_06 implementation)
    def determine_feature_types(rows, feature_cols):
        numeric_cols = []
        categorical_cols = []
        for col in feature_cols:
            is_numeric = True
            for r in rows:
                v = r[col]
                if v is None:
                    continue
                if isinstance(v, float):
                    continue
                try:
                    float(v)
                except Exception:
                    is_numeric = False
                    break
            if is_numeric:
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

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def get_adjustable_columns(rows):
    """Return list of columns that can be changed."""
    all_cols = list(rows[0].keys())
    non_goal = [c for c in all_cols
                if not (c.endswith('+') or c.endswith('-')) and not c.endswith('X')]
    return [c for c in non_goal if c != 'Kloc+']

def distinct_values(rows, columns):
    """Return dict of column -> set of non‑None values."""
    d = {c: set() for c in columns}
    for r in rows:
        for c in columns:
            v = r[c]
            if v is not None:
                d[c].add(v)
    return d

def build_model(rows, feature_cols, target_col):
    """
    Build linear regression model using all rows except held‑out (divisible by 5).
    Returns model, numeric_cols, categorical_cols, encodings.
    """
    train_rows = [r for idx, r in enumerate(rows, start=1)
                  if idx % 5 != 0 and r[target_col] is not None
                  and all(r[c] is not None for c in feature_cols)]
    numeric_cols, categorical_cols = determine_feature_types(train_rows, feature_cols)
    encodings = build_encodings(train_rows, categorical_cols)
    X_train = build_feature_matrix(train_rows, feature_cols,
                                   numeric_cols, categorical_cols, encodings)
    y_train = np.array([r[target_col] for r in train_rows], dtype=float)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, numeric_cols, categorical_cols, encodings

def predict_row(row, model, numeric_cols, categorical_cols, encodings, feature_cols):
    vec = []
    for col in feature_cols:
        v = row[col]
        if v is None:
            return None
        if col in numeric_cols:
            vec.append(float(v))
        else:
            vec.append(encodings[col][v])
    return float(model.predict(np.array([vec], dtype=float))[0])

# ----------------------------------------------------------------------
# NASA-15: effort reduction
# ----------------------------------------------------------------------
def nasa_15(rows):
    all_cols = list(rows[0].keys())
    feature_cols = [c for c in all_cols if c not in ('Effort-', 'Defects-')]

    # Build effort prediction model
    model_eff, num_eff, cat_eff, enc_eff = build_model(rows, feature_cols, 'Effort-')

    # Predict for held‑out rows
    predictions = {}
    for idx, r in enumerate(rows, start=1):
        if idx % 5 == 0 and r['Effort-'] is not None:
            pred = predict_row(r, model_eff, num_eff, cat_eff, enc_eff, feature_cols)
            predictions[idx] = pred

    if not predictions:
        # No predictions available
        print("== NASA-15 ==")
        print("No held‑out rows with sufficient data.")
        print(f"ANSWER NASA-15 {{}}")
        return

    # Select the project with the highest predicted effort
    project = max(predictions, key=lambda k: predictions[k])
    before_effort = predictions[project]
    current_row = rows[project-1]

    adjustable_cols = get_adjustable_columns(rows)
    distinct_vals = distinct_values(rows, adjustable_cols)

    best_change = None
    best_after = before_effort
    for col in adjustable_cols:
        for val in distinct_vals[col]:
            if val == current_row[col]:
                continue
            mod_row = current_row.copy()
            mod_row[col] = val
            after = predict_row(mod_row, model_eff, num_eff, cat_eff, enc_eff, feature_cols)
            if after is None:
                continue
            if after < best_after:
                best_after = after
                best_change = (col, val)

    # Output
    print("== NASA-15 ==")
    print(f"Project: {project}")
    print(f"Predicted effort before: {format_number(before_effort)}")
    if best_change:
        col, new_val = best_change
        cur_val = current_row[col]
        print(f"Change: {col} from {format_number(cur_val)} to {format_number(new_val)}")
    else:
        print("No single change improves effort.")
    print(f"Predicted effort after: {format_number(best_after)}")

    # JSON answer
    changes = {}
    if best_change:
        col, new_val = best_change
        changes[col] = [current_row[col], new_val]
    answer = {
        "project": project,
        "changes": changes,
        "predicted_effort": {
            "before": before_effort,
            "after": best_after
        }
    }
    print(f"ANSWER NASA-15 {json.dumps(answer, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# NASA-16: defect reduction
# ----------------------------------------------------------------------
def nasa_16(rows):
    all_cols = list(rows[0].keys())
    feature_cols = [c for c in all_cols if c not in ('Effort-', 'Defects-')]

    # Build models
    model_eff, num_eff, cat_eff, enc_eff = build_model(rows, feature_cols, 'Effort-')
    model_def, num_def, cat_def, enc_def = build_model(rows, feature_cols, 'Defects-')

    # Predict for held‑out rows
    eff_predictions = {}
    def_predictions = {}
    for idx, r in enumerate(rows, start=1):
        if idx % 5 == 0 and r['Defects-'] is not None:
            eff_predictions[idx] = predict_row(r, model_eff, num_eff, cat_eff, enc_eff, feature_cols)
            def_predictions[idx] = predict_row(r, model_def, num_def, cat_def, enc_def, feature_cols)

    if not def_predictions:
        print("== NASA-16 ==")
        print("No held‑out rows with sufficient data.")
        print(f"ANSWER NASA-16 {{}}")
        return

    # Select the project with the highest predicted defects
    project = max(def_predictions, key=lambda k: def_predictions[k])
    before_defects = def_predictions[project]
    before_effort = eff_predictions[project]
    current_row = rows[project-1]

    adjustable_cols = get_adjustable_columns(rows)
    distinct_vals = distinct_values(rows, adjustable_cols)

    # Gather all single‑change improvements
    change_candidates = []
    for col in adjustable_cols:
        for val in distinct_vals[col]:
            if val == current_row[col]:
                continue
            mod_row = current_row.copy()
            mod_row[col] = val
            after_def = predict_row(mod_row, model_def, num_def, cat_def, enc_def, feature_cols)
            after_eff = predict_row(mod_row, model_eff, num_eff, cat_eff, enc_eff, feature_cols)
            if after_def is None or after_eff is None:
                continue
            defect_reduction = before_defects - after_def
            if defect_reduction > 0:
                change_candidates.append((defect_reduction, col, val, after_eff, after_def))

    # Select up to 3 best changes
    change_candidates.sort(key=lambda t: t[0], reverse=True)
    selected = change_candidates[:3]

    # Apply selected changes to compute final predictions
    combined_row = current_row.copy()
    changes_dict = {}
    for _, col, val, _, _ in selected:
        changes_dict[col] = [current_row[col], val]
        combined_row[col] = val

    combined_defects = predict_row(combined_row, model_def, num_def, cat_def, enc_def, feature_cols)
    combined_effort = predict_row(combined_row, model_eff, num_eff, cat_eff, enc_eff, feature_cols)

    # Output
    print("== NASA-16 ==")
    print(f"Project: {project}")
    print(f"Predicted defects before: {format_number(before_defects)}")
    print(f"Predicted effort before: {format_number(before_effort)}")
    if selected:
        print("Changes:")
        for _, col, val, _, _ in selected:
            cur_val = current_row[col]
            print(f"  {col}: {format_number(cur_val)} -> {format_number(val)}")
    else:
        print("No single change improves defects.")
    print(f"Predicted defects after: {format_number(combined_defects)}")
    print(f"Predicted effort after: {format_number(combined_effort)}")

    # JSON answer
    answer = {
        "project": project,
        "changes": changes_dict,
        "predicted_defects": {
            "before": before_defects,
            "after": combined_defects
        },
        "predicted_effort": {
            "before": before_effort,
            "after": combined_effort
        }
    }
    print(f"ANSWER NASA-16 {json.dumps(answer, separators=(',', ':'))}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)
    path = sys.argv[1]
    rows = read_csv(path)
    nasa_15(rows)
    nasa_16(rows)

if __name__ == "__main__":
    main()
