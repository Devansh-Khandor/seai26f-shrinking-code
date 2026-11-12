#!/usr/bin/env python3
"""
skill_02.py - Predict effort and defects for new software projects.

Usage:
    python skill_02.py <csv_path>
"""

import csv
import sys
import math
import numpy as np
from statistics import mean, median
from typing import List, Dict, Tuple

# ----------------------------------------------------------------------
# Reuse helpers from skill_01
# ----------------------------------------------------------------------
try:
    # Import existing utility functions if skill_01.py is available
    from skill_01 import load_csv, get_numeric_column
except Exception:
    # Fallback to minimal local implementations if skill_01 is not importable
    def load_csv(csv_path: str) -> Tuple[List[Dict[str, str]], List[str]]:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            rows = [row for row in reader]
        return rows, header

    def get_numeric_column(rows: List[Dict[str, str]], col: str) -> List[float]:
        values = []
        for r in rows:
            try:
                val = float(r[col])
                values.append(val)
            except (ValueError, TypeError):
                pass
        return values

# ----------------------------------------------------------------------
# Feature encoding utilities
# ----------------------------------------------------------------------
def identify_numeric_columns(rows: List[Dict[str, str]], header: List[str]) -> Tuple[set, set]:
    """Return sets of numeric and categorical column names."""
    numeric = set()
    categorical = set()
    for col in header:
        if col == 'idX':  # Skip id column
            continue
        # Try to parse first non-empty value
        sample = None
        for r in rows:
            val = r.get(col, None)
            if val is not None and val != '':
                sample = val
                break
        if sample is None:
            # Default to categorical
            categorical.add(col)
        else:
            try:
                float(sample)
                numeric.add(col)
            except ValueError:
                categorical.add(col)
    return numeric, categorical

def build_mappings(rows: List[Dict[str, str]], categorical: set) -> Dict[str, Dict[str, int]]:
    """Create integer mapping dictionaries for each categorical column."""
    mappings = {}
    for col in categorical:
        mapping = {}
        idx = 0
        for r in rows:
            val = r.get(col, '')
            if val == '':
                continue
            if val not in mapping:
                mapping[val] = idx
                idx += 1
        mappings[col] = mapping
    return mappings

def encode_row(r: Dict[str, str],
               numeric: set,
               categorical: set,
               mappings: Dict[str, Dict[str, int]]) -> List[float]:
    """Encode a single row into numeric feature vector."""
    vec = []
    for col in numeric:
        try:
            val = float(r.get(col, '0'))
        except ValueError:
            val = 0.0
        vec.append(val)
    for col in categorical:
        val = r.get(col, '')
        mapping = mappings[col]
        vec.append(float(mapping.get(val, -1)))  # unseen categories map to -1
    return vec

def prepare_data(rows: List[Dict[str, str]],
                 header: List[str],
                 target_col: str) -> Tuple[np.ndarray, np.ndarray,
                                            set, set, Dict[str, Dict[str, int]]]:
    """Return X, y, numeric cols, categorical cols, mappings."""
    numeric, categorical = identify_numeric_columns(rows, header)
    # Exclude target and id columns from features
    numeric.discard(target_col)
    categorical.discard(target_col)
    numeric.discard('idX')
    categorical.discard('idX')

    mappings = build_mappings(rows, categorical)

    X = []
    y = []
    for r in rows:
        try:
            y_val = float(r.get(target_col, '0'))
        except ValueError:
            continue
        X.append(encode_row(r, numeric, categorical, mappings))
        y.append(y_val)
    return np.array(X, dtype=float), np.array(y, dtype=float), numeric, categorical, mappings

# ----------------------------------------------------------------------
# Regression utilities
# ----------------------------------------------------------------------
def train_regression(X: np.ndarray, y: np.ndarray):
    """Train a simple linear regression model and return it."""
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_with_uncertainty(model, X_train, y_train, X_new):
    """Return predicted value and an uncertainty estimate (2*std residual)."""
    y_pred_train = model.predict(X_train)
    residuals = y_train - y_pred_train
    std_resid = np.sqrt(np.mean(residuals ** 2))
    y_pred_new = model.predict(X_new)[0]
    uncertainty = 2 * std_resid  # Rough 95% CI
    return y_pred_new, uncertainty, std_resid

# ----------------------------------------------------------------------
# Reporting functions
# ----------------------------------------------------------------------
def report_nasa_03(rows: List[Dict[str, str]], header: List[str]) -> None:
    """Predict effort for a hypothetical project (first row)."""
    print("\n== NASA-03 ==")
    X, y, _, _, _ = prepare_data(rows, header, 'Effort-')
    model = train_regression(X, y)
    # Predict for the first project as a sample "new" project
    X_new = X[0].reshape(1, -1)
    pred, uncert, std_resid = predict_with_uncertainty(model, X, y, X_new)

    print(f"Linear regression R²: {model.score(X, y):.3f}")
    print(f"Predicted effort for project 1: {pred:.2f} +/- {uncert:.2f} (residual SD={std_resid:.2f})")
    print("  (Units: effort units as in dataset)")

def report_nasa_04(rows: List[Dict[str, str]], header: List[str]) -> None:
    """Predict defects for a hypothetical project (first row)."""
    print("\n== NASA-04 ==")
    X, y, _, _, _ = prepare_data(rows, header, 'Defects-')
    model = train_regression(X, y)
    X_new = X[0].reshape(1, -1)
    pred, uncert, std_resid = predict_with_uncertainty(model, X, y, X_new)

    print(f"Linear regression R²: {model.score(X, y):.3f}")
    print(f"Predicted defects for project 1: {pred:.2f} +/- {uncert:.2f} (residual SD={std_resid:.2f})")
    print("  (Units: defect counts as in dataset)")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, header = load_csv(csv_path)

    report_nasa_03(rows, header)
    report_nasa_04(rows, header)

if __name__ == "__main__":
    main()
