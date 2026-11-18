#!/usr/bin/env python3
"""
skill_02.py - Estimates missing fuel efficiency and weight with uncertainty

Usage:
    python skill_02.py <csv_path>
"""

import sys
import csv
import json
import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def read_csv(path):
    """Read CSV and return list of dicts (rows). Convert numeric columns to float or None."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric columns to float, keep '?' as None
            for col in ['Lbs-', 'Acc+', 'Mpg+']:
                val = row[col]
                if val == '?':
                    row[col] = None
                else:
                    try:
                        row[col] = float(val)
                    except ValueError:
                        row[col] = None
            # Convert feature columns to float as well
            for col in ['Clndrs', 'Volume', 'HpX', 'Model', 'origin']:
                row[col] = float(row[col])
            rows.append(row)
    return rows

def get_held_out_indices(n_rows):
    """Rows whose 1‑based number is divisible by 5."""
    return [i for i in range(1, n_rows + 1) if i % 5 == 0]

# ----------------------------------------------------------------------
# CAR-03: Predict missing Mpg+ for held‑out rows
# ----------------------------------------------------------------------
def car03(rows, held_out_indices):
    # Training data: all rows except held‑out, with known Mpg+
    train_rows = [row for i, row in enumerate(rows, 1)
                  if i not in held_out_indices and row['Mpg+'] is not None]
    X_train = np.array([[row['Clndrs'], row['Volume'], row['HpX'],
                         row['Model'], row['origin']] for row in train_rows])
    y_train = np.array([row['Mpg+'] for row in train_rows])

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = {}
    errors = []
    for i, row in enumerate(rows, 1):
        if i not in held_out_indices:
            continue
        X = np.array([[row['Clndrs'], row['Volume'], row['HpX'],
                       row['Model'], row['origin']]])
        pred = float(model.predict(X)[0])
        predictions[str(i)] = pred
        if row['Mpg+'] is not None:
            errors.append(abs(pred - row['Mpg+']))

    mae = float(np.mean(errors)) if errors else None
    return predictions, mae

# ----------------------------------------------------------------------
# CAR-04: Predict missing Lbs- for held‑out rows with uncertainty
# ----------------------------------------------------------------------
def car04(rows, held_out_indices):
    # Training data: all rows except held‑out, with known Lbs-
    train_rows = [row for i, row in enumerate(rows, 1)
                  if i not in held_out_indices and row['Lbs-'] is not None]
    X_train = np.array([[row['Clndrs'], row['Volume'], row['HpX'],
                         row['Model'], row['origin']] for row in train_rows])
    y_train = np.array([row['Lbs-'] for row in train_rows])

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Residual standard deviation on training set
    y_pred_train = model.predict(X_train)
    residuals = y_train - y_pred_train
    stddev = float(np.std(residuals, ddof=0))

    predictions = {}
    for i, row in enumerate(rows, 1):
        if i not in held_out_indices:
            continue
        X = np.array([[row['Clndrs'], row['Volume'], row['HpX'],
                       row['Model'], row['origin']]])
        est = float(model.predict(X)[0])
        predictions[str(i)] = {
            "estimate": est,
            "low": est - stddev,
            "high": est + stddev
        }
    return predictions

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_csv(csv_path)
    n_rows = len(rows)
    held_out_indices = get_held_out_indices(n_rows)

    # CAR-03
    print("== CAR-03 ==")
    pred_mpg, mae = car03(rows, held_out_indices)
    print(f"MAE: {mae:.6g}")
    # Show a few sample predictions
    sample = list(pred_mpg.items())[:5]
    print(f"Sample predictions: {sample}")

    ans_car03 = {
        "predictions": pred_mpg,
        "mae": mae
    }
    print("ANSWER CAR-03", json.dumps(ans_car03, separators=(',', ':')))

    # CAR-04
    print("\n== CAR-04 ==")
    pred_lbs = car04(rows, held_out_indices)
    print(f"Number of predictions: {len(pred_lbs)}")
    # Show a few sample predictions
    sample_lbs = list(pred_lbs.items())[:5]
    print(f"Sample predictions: {sample_lbs}")

    ans_car04 = {
        "predictions": pred_lbs
    }
    print("ANSWER CAR-04", json.dumps(ans_car04, separators=(',', ':')))

if __name__ == "__main__":
    main()
