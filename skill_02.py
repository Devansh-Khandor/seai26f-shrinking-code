#!/usr/bin/env python3
"""
skill_02.py

Usage:
    python skill_02.py <path-to-auto93.csv>

This script implements two new features for the dealership:

  CAR-03 : Estimate missing fuel‑efficiency (Mpg+) values from known vehicle attributes.
  CAR-04 : Estimate missing vehicle weight (Lbs‑) from the available information and report
           the uncertainty of the estimate.

The output follows the requested format exactly: for each request, a line that is
exactly `== <ID> ==` followed by at least three lines of results, including the actual numbers.
"""

import csv
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------

def read_data(csv_path):
    """
    Reads the CSV file and returns a list of dicts.
    Missing numeric values are stored as None.
    Expected columns (in order):
        Clndrs,Volume,HpX,Model,origin,Lbs-,Acc+,Mpg+
    """
    records = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all(cell.strip() == '' for cell in row):
                continue
            try:
                rec = {
                    'Clndrs': int(row[0]) if row[0].strip() else None,
                    'Volume': int(row[1]) if row[1].strip() else None,
                    'HpX': int(row[2]) if row[2].strip() else None,
                    'Model': int(row[3]) if row[3].strip() else None,
                    'origin': int(row[4]) if row[4].strip() else None,
                    'Lbs-': float(row[5]) if row[5].strip() else None,
                    'Acc+': float(row[6]) if row[6].strip() else None,
                    'Mpg+': float(row[7]) if row[7].strip() else None,
                }
            except (IndexError, ValueError):
                continue
            # Skip rows with no data at all
            if all(v is None for v in rec.values()):
                continue
            records.append(rec)
    return records

# ---------------------------------------------------------------------------

def build_regressor(X, y):
    """Fit a simple linear regression model."""
    reg = LinearRegression()
    reg.fit(X, y)
    return reg

# ---------------------------------------------------------------------------

def estimate_missing_mpg(records):
    """
    Estimate missing Mpg+ values.
    Uses weight (Lbs-), horsepower (HpX), engine displacement (Volume),
    car length (Clndrs), model year (Model), and origin as features.
    Returns a list of tuples (index, actual_mpg, predicted_mpg).
    """
    # Prepare training data: rows where mpg is known
    train = [r for r in records if r['Mpg+'] is not None]
    if not train:
        return []

    X_train = np.column_stack([
        r['Clndrs'],
        r['Volume'],
        r['HpX'],
        r['Model'],
        r['origin'],
        r['Lbs-'],
    ])
    y_train = np.array([r['Mpg+'] for r in train])

    # Train model
    reg = build_regressor(X_train, y_train)

    # Estimate on missing mpg rows
    results = []
    for idx, r in enumerate(records):
        if r['Mpg+'] is None:
            # Ensure all features present; skip if any missing
            if None in [r['Clndrs'], r['Volume'], r['HpX'],
                        r['Model'], r['origin'], r['Lbs-']]:
                continue
            x = np.array([[
                r['Clndrs'],
                r['Volume'],
                r['HpX'],
                r['Model'],
                r['origin'],
                r['Lbs-'],
            ]])
            pred = reg.predict(x)[0]
            results.append((idx + 1, None, pred))
    return results

# ---------------------------------------------------------------------------

def estimate_missing_weight(records):
    """
    Estimate missing vehicle weight (Lbs-).
    Uses mpg, horsepower, displacement, car length, model year, and origin as features.
    Returns a list of tuples (index, predicted_weight, uncertainty).
    """
    # Training data: rows with weight known
    train = [r for r in records if r['Lbs-'] is not None]
    if not train:
        return []

    X_train = np.column_stack([
        r['Clndrs'],
        r['Volume'],
        r['HpX'],
        r['Model'],
        r['origin'],
        r['Mpg+'],
    ])
    y_train = np.array([r['Lbs-'] for r in train])

    # Train model
    reg = build_regressor(X_train, y_train)

    # Residual standard deviation as uncertainty estimate
    y_pred_train = reg.predict(X_train)
    residuals = y_train - y_pred_train
    sigma = np.sqrt(mean_squared_error(y_train, y_pred_train))

    # Estimate on missing weight rows
    results = []
    for idx, r in enumerate(records):
        if r['Lbs-'] is None:
            # Ensure all features present; skip if any missing
            if None in [r['Clndrs'], r['Volume'], r['HpX'],
                        r['Model'], r['origin'], r['Mpg+']]:
                continue
            x = np.array([[
                r['Clndrs'],
                r['Volume'],
                r['HpX'],
                r['Model'],
                r['origin'],
                r['Mpg+'],
            ]])
            pred = reg.predict(x)[0]
            # Simple uncertainty: ± sigma
            results.append((idx + 1, pred, sigma))
    return results

# ---------------------------------------------------------------------------

def print_car_03(results):
    """Print CAR-03 results."""
    print("\n== CAR-03 ==")
    if not results:
        print("No missing Mpg+ values to estimate.")
        return
    # Show at least three lines
    for i, (idx, _, pred) in enumerate(results[:3], start=1):
        print(f"{i}) Vehicle row {idx}: Predicted Mpg+ = {pred:.1f}")
    if len(results) > 3:
        print(f"... and {len(results)-3} more estimations.")

def print_car_04(results):
    """Print CAR-04 results."""
    print("\n== CAR-04 ==")
    if not results:
        print("No missing Lbs- values to estimate.")
        return
    # Show at least three lines
    for i, (idx, pred, sigma) in enumerate(results[:3], start=1):
        print(f"{i}) Vehicle row {idx}: Predicted weight = {pred:.0f} lbs ± {sigma:.0f} lbs")
    if len(results) > 3:
        print(f"... and {len(results)-3} more estimations.")

# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-auto93.csv>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    records = read_data(csv_path)

    if not records:
        print("No valid data found in the CSV.", file=sys.stderr)
        sys.exit(1)

    mpg_results = estimate_missing_mpg(records)
    weight_results = estimate_missing_weight(records)

    print_car_03(mpg_results)
    print_car_04(weight_results)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
