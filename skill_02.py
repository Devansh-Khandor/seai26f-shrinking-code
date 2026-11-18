#!/usr/bin/env python3
"""
Skill 02: Fuel-efficiency and weight prediction with uncertainty.

The program receives a CSV path on the command line.
It prints two sections:
  == CAR-03 ==  (fuel‑efficiency prediction for held‑out rows)
  == CAR-04 ==  (weight prediction with uncertainty for held‑out rows)
followed by the machine‑checked ANSWER lines as required.
"""

import sys
import csv
import json
import math
from statistics import mean
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def read_csv(path):
    """Read CSV file and return header and list of rows (dicts)."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    return header, rows

def parse_value(v):
    """Parse numeric value or return None for missing data."""
    if v in ('', '?'):
        return None
    try:
        return float(v)
    except ValueError:
        return None

def is_goal(col):
    """Return True if column ends in '+' or '-'."""
    return col.endswith('+') or col.endswith('-')

def is_input(col):
    """Return True if column is numeric, not a goal, and does not end in 'X'."""
    return not is_goal(col) and not col.endswith('X')

def feature_columns(header, exclude=None):
    """Return list of columns to be used as features."""
    if exclude is None:
        exclude = set()
    return [c for c in header if is_input(c) and c not in exclude]

def get_row_numeric(row, cols):
    """Return list of numeric values for the specified columns, or None if any missing."""
    vals = []
    for c in cols:
        v = parse_value(row[c])
        if v is None:
            return None
        vals.append(v)
    return vals

# --------------------------------------------------------------------------- #
# Prediction helper
# --------------------------------------------------------------------------- #

class LinearModel:
    """Wrapper around sklearn LinearRegression to handle missing data."""
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        self.model = LinearRegression()
        self.trained = False

    def fit(self, rows, target_col):
        X = []
        y = []
        for row in rows:
            if parse_value(row[target_col]) is None:
                continue
            fv = get_row_numeric(row, self.feature_cols)
            if fv is None:
                continue
            X.append(fv)
            y.append(parse_value(row[target_col]))
        if not X:
            raise ValueError("No training data available for target '{}'".format(target_col))
        self.model.fit(X, y)
        self.trained = True

    def predict(self, row):
        if not self.trained:
            raise NotFittedError("Model has not been fitted.")
        fv = get_row_numeric(row, self.feature_cols)
        if fv is None:
            return None
        return float(self.model.predict([fv])[0])

    def residual_std(self, rows, target_col):
        """Return standard deviation of residuals on the training data."""
        preds = []
        residuals = []
        for row in rows:
            if parse_value(row[target_col]) is None:
                continue
            fv = get_row_numeric(row, self.feature_cols)
            if fv is None:
                continue
            pred = self.model.predict([fv])[0]
            actual = parse_value(row[target_col])
            residuals.append(actual - pred)
        if not residuals:
            return 0.0
        return math.sqrt(mean([r ** 2 for r in residuals]))

# --------------------------------------------------------------------------- #
# Main processing
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    # Identify held-out rows (row numbers divisible by 5, 1-based)
    held_out_indices = [i + 1 for i in range(len(rows)) if (i + 1) % 5 == 0]
    held_out_rows = [rows[i - 1] for i in held_out_indices]
    training_rows = [row for idx, row in enumerate(rows, start=1) if idx not in held_out_indices]

    # ----------------------------------------------------------------------- #
    # CAR-03: Fuel‑efficiency (Mpg+) prediction
    # ----------------------------------------------------------------------- #
    print("== CAR-03 ==")
    mpg_col = next((c for c in header if c == 'Mpg+'), None)
    if mpg_col is None:
        print("Error: No 'Mpg+' column found.")
        sys.exit(1)

    mpg_features = feature_columns(header, exclude={mpg_col})
    mpg_model = LinearModel(mpg_features)
    mpg_model.fit(training_rows, mpg_col)

    # Predictions for held‑out rows
    mpg_predictions = {}
    actuals = {}
    for idx, row in zip(held_out_indices, held_out_rows):
        pred = mpg_model.predict(row)
        if pred is None:
            continue
        mpg_predictions[str(idx)] = round(pred, 4)
        actuals[idx] = parse_value(row[mpg_col])

    # Compute MAE
    mae = mean([abs(actuals[i] - mpg_predictions[str(i)]) for i in held_out_indices if str(i) in mpg_predictions])

    # Print at least three lines of results
    print(f"Total held‑out rows: {len(held_out_indices)}")
    print(f"Predictions made: {len(mpg_predictions)}")
    print(f"Mean absolute error: {round(mae, 4)}")

    # Machine‑checked answer
    answer_mpg = {
        "predictions": mpg_predictions,
        "mae": round(mae, 4)
    }
    print(f"ANSWER CAR-03 {json.dumps(answer_mpg)}")

    # ----------------------------------------------------------------------- #
    # CAR-04: Weight (Lbs-) prediction with uncertainty
    # ----------------------------------------------------------------------- #
    print("\n== CAR-04 ==")
    lbs_col = next((c for c in header if c == 'Lbs-'), None)
    if lbs_col is None:
        print("Error: No 'Lbs-' column found.")
        sys.exit(1)

    lbs_features = feature_columns(header, exclude={lbs_col})
    lbs_model = LinearModel(lbs_features)
    lbs_model.fit(training_rows, lbs_col)

    # Residual standard deviation for uncertainty interval
    resid_std = lbs_model.residual_std(training_rows, lbs_col)

    lbs_predictions = {}
    for idx, row in zip(held_out_indices, held_out_rows):
        est = lbs_model.predict(row)
        if est is None:
            continue
        low = est - 1.96 * resid_std
        high = est + 1.96 * resid_std
        lbs_predictions[str(idx)] = {
            "estimate": round(est, 4),
            "low": round(low, 4),
            "high": round(high, 4)
        }

    # Print at least three lines of results
    print(f"Total held‑out rows: {len(held_out_indices)}")
    print(f"Predictions made: {len(lbs_predictions)}")
    print(f"Residual std dev: {round(resid_std, 4)}")

    # Machine‑checked answer
    answer_lbs = {
        "predictions": lbs_predictions
    }
    print(f"ANSWER CAR-04 {json.dumps(answer_lbs)}")

if __name__ == "__main__":
    main()
