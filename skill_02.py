#!/usr/bin/env python3
"""
Skill 02: NASA-03 & NASA-04 predictions

Usage:
    python skill_02.py <path-to-nasa93dem.csv>
"""

import sys
import json
import numpy as np
from sklearn.linear_model import LinearRegression

# Import helpers from skill_01
try:
    from skill_01 import read_csv, format_number
except ImportError:
    # In case skill_01 is not available, reimplement minimal helpers
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
# Helper functions
# ----------------------------------------------------------------------
def determine_feature_types(rows, feature_cols):
    """Return lists of numeric and categorical feature columns."""
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
    """Return mapping dictionaries for each categorical column."""
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
    """Return numpy array of features for given rows."""
    X = []
    for r in rows:
        vec = []
        for col in feature_cols:
            v = r[col]
            if v is None:
                # Should not happen for rows passed to this function
                vec.append(0.0)
                continue
            if col in numeric_cols:
                vec.append(float(v))
            else:
                vec.append(encodings[col][v])
        X.append(vec)
    return np.array(X, dtype=float)

# ----------------------------------------------------------------------
# NASA-03: Effort estimation
# ----------------------------------------------------------------------
def nasa_03(rows):
    # Identify feature columns: all except Effort- and Defects-
    all_cols = list(rows[0].keys())
    feature_cols = [c for c in all_cols if c not in ('Effort-', 'Defects-')]

    # Determine held-out rows (indices divisible by 5)
    held_out_indices = set(i + 1 for i in range(len(rows)) if (i + 1) % 5 == 0)

    # Separate training rows (history) and test rows (held-out)
    train_rows = [r for idx, r in enumerate(rows, start=1) if idx not in held_out_indices
                  and r['Effort-'] is not None
                  and all(r[c] is not None for c in feature_cols)]
    test_rows = [r for idx, r in enumerate(rows, start=1) if idx in held_out_indices
                 and r['Effort-'] is not None
                 and all(r[c] is not None for c in feature_cols)]

    # Feature types and encodings
    numeric_cols, categorical_cols = determine_feature_types(train_rows, feature_cols)
    encodings = build_encodings(train_rows, categorical_cols)

    # Build matrices
    X_train = build_feature_matrix(train_rows, feature_cols, numeric_cols, categorical_cols, encodings)
    y_train = np.array([r['Effort-'] for r in train_rows], dtype=float)

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on held-out rows
    predictions = {}
    errors = []
    for idx, r in enumerate(rows, start=1):
        if idx not in held_out_indices:
            continue
        if any(r[c] is None for c in feature_cols):
            continue  # skip rows with missing features
        # Encode features
        vec = []
        for col in feature_cols:
            v = r[col]
            if col in numeric_cols:
                vec.append(float(v))
            else:
                # If unseen category, skip prediction
                if v not in encodings[col]:
                    break
                vec.append(encodings[col][v])
        else:
            X_test = np.array([vec], dtype=float)
            pred = model.predict(X_test)[0]
            predictions[str(idx)] = float(pred)
            errors.append(abs(pred - r['Effort-']))

    mae = float(sum(errors) / len(errors)) if errors else None

    # Print results
    print("== NASA-03 ==")
    for rid, pred in predictions.items():
        print(f"Row {rid}: Effort- predicted={format_number(pred)}")
    print(f"Mean absolute error = {format_number(mae)}")

    # JSON answer
    json_obj = {"predictions": predictions, "mae": mae}
    json_str = json.dumps(json_obj, separators=(',', ':'))
    print(f"ANSWER NASA-03 {json_str}")

# ----------------------------------------------------------------------
# NASA-04: Defect estimation with uncertainty
# ----------------------------------------------------------------------
def nasa_04(rows):
    all_cols = list(rows[0].keys())
    feature_cols = [c for c in all_cols if c not in ('Effort-', 'Defects-')]

    held_out_indices = set(i + 1 for i in range(len(rows)) if (i + 1) % 5 == 0)

    train_rows = [r for idx, r in enumerate(rows, start=1) if idx not in held_out_indices
                  and r['Defects-'] is not None
                  and all(r[c] is not None for c in feature_cols)]
    test_rows = [r for idx, r in enumerate(rows, start=1) if idx in held_out_indices
                 and r['Defects-'] is not None
                 and all(r[c] is not None for c in feature_cols)]

    numeric_cols, categorical_cols = determine_feature_types(train_rows, feature_cols)
    encodings = build_encodings(train_rows, categorical_cols)

    X_train = build_feature_matrix(train_rows, feature_cols, numeric_cols, categorical_cols, encodings)
    y_train = np.array([r['Defects-'] for r in train_rows], dtype=float)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Residual standard deviation on training set
    residuals = y_train - model.predict(X_train)
    sigma = np.sqrt(np.mean(residuals ** 2))
    interval = 1.96 * sigma

    predictions = {}
    for idx, r in enumerate(rows, start=1):
        if idx not in held_out_indices:
            continue
        if any(r[c] is None for c in feature_cols):
            continue
        vec = []
        for col in feature_cols:
            v = r[col]
            if col in numeric_cols:
                vec.append(float(v))
            else:
                if v not in encodings[col]:
                    break
                vec.append(encodings[col][v])
        else:
            X_test = np.array([vec], dtype=float)
            pred = model.predict(X_test)[0]
            low = float(pred - interval)
            high = float(pred + interval)
            predictions[str(idx)] = {"estimate": float(pred), "low": low, "high": high}

    # Print results
    print("== NASA-04 ==")
    for rid, vals in predictions.items():
        est = vals["estimate"]
        low = vals["low"]
        high = vals["high"]
        print(f"Row {rid}: Defects- estimate={format_number(est)} low={format_number(low)} high={format_number(high)}")

    # JSON answer
    json_obj = {"predictions": predictions}
    json_str = json.dumps(json_obj, separators=(',', ':'))
    print(f"ANSWER NASA-04 {json_str}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)
    path = sys.argv[1]
    rows = read_csv(path)
    nasa_03(rows)
    nasa_04(rows)

if __name__ == "__main__":
    main()
