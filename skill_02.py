#!/usr/bin/env python3
"""
skill_02.py
------------
Command line tool that reads a CSV file of NASA software project data
and answers two new requests:

    NASA-03: estimate Effort- for held‑out projects
    NASA-04: estimate Defects- and uncertainty for held‑out projects

Usage:
    python skill_02.py <csv path>
"""

import csv
import json
import sys
import statistics
import math
from typing import List, Dict, Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def parse_float(s: str) -> float | None:
    """Convert a CSV field to float, treating '?' as missing."""
    try:
        return float(s)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Core functionality
# --------------------------------------------------------------------------- #
def load_csv(path: str) -> List[Dict[str, str]]:
    """Read the CSV and return a list of rows (dict)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def identify_features(rows: List[Dict[str, str]],
                      goal_cols: List[str]) -> Tuple[List[str], List[int], List[int]]:
    """
    Return:
      * feature_names: list of column names used as features
      * numeric_indices: indices of numeric columns
      * categorical_indices: indices of categorical columns
    """
    # All columns except goals
    all_cols = list(rows[0].keys())
    feature_names = [c for c in all_cols if c not in goal_cols]
    # Determine numeric columns by attempting to parse first non‑missing value
    numeric_cols = []
    categorical_cols = []
    for col in feature_names:
        val = None
        for r in rows:
            if r[col] != "?":
                val = r[col]
                break
        if val is None:
            # All missing: treat as categorical
            categorical_cols.append(col)
        else:
            if parse_float(val) is not None:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

    # Build indices according to order in feature_names
    numeric_indices = [feature_names.index(c) for c in numeric_cols]
    categorical_indices = [feature_names.index(c) for c in categorical_cols]
    return feature_names, numeric_indices, categorical_indices


def build_feature_matrix(rows: List[Dict[str, str]],
                         feature_names: List[str]) -> np.ndarray:
    """
    Return a numpy array of shape (n_rows, n_features) where numeric values are
    floats (or np.nan) and categorical values are strings. Mixed dtype is used.
    """
    data = []
    for r in rows:
        row_vals = []
        for col in feature_names:
            v = r[col]
            if v == "?":
                # Missing numeric -> np.nan, missing categorical -> 'missing'
                # We will detect later whether column is numeric
                row_vals.append(np.nan)
            else:
                # Keep string for categorical, numeric will be parsed later
                row_vals.append(v)
        data.append(row_vals)
    return np.array(data, dtype=object)


def prepare_data(rows: List[Dict[str, str]],
                 goal: str,
                 goal_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Split rows into training (not held‑out) and held‑out sets.
    Returns:
        X_train, y_train, held_out_indices (1‑based)
    """
    feature_names, numeric_indices, categorical_indices = identify_features(rows, goal_cols)

    X_all = build_feature_matrix(rows, feature_names)

    # Parse numeric columns now
    X_numeric = X_all[:, numeric_indices].astype(float, copy=False)
    X_categorical = X_all[:, categorical_indices].astype(str, copy=False)

    # Combine back preserving original order
    X = np.empty(X_all.shape, dtype=object)
    X[:, numeric_indices] = X_numeric
    X[:, categorical_indices] = X_categorical

    # Build y array
    y = np.array([parse_float(r[goal]) for r in rows], dtype=float)

    # Identify held‑out indices (1‑based)
    held_out_indices = [i + 1 for i, r in enumerate(rows) if (i + 1) % 5 == 0]

    # Build training mask: not held‑out AND target present
    train_mask = [( (i + 1) % 5 != 0) and (y[i] is not None) for i in range(len(rows))]

    X_train = X[train_mask]
    y_train = y[train_mask]
    return X_train, y_train, held_out_indices


def build_pipeline(numeric_indices: List[int], categorical_indices: List[int]) -> Pipeline:
    """Return a scikit‑learn Pipeline ready for fitting."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_indices),
            ("cat", categorical_pipe, categorical_indices),
        ]
    )

    model = Pipeline(
        steps=[("preprocessor", preprocessor),
               ("regressor", LinearRegression()),
              ]
    )
    return model


def train_and_predict(rows: List[Dict[str, str]],
                      goal: str,
                      goal_cols: List[str]) -> Tuple[Dict[int, float], float]:
    """
    Train on non‑held‑out rows and predict on held‑out rows.
    Returns a dict mapping held‑out row number to predicted value and the MAE.
    """
    X_train, y_train, held_out_indices = prepare_data(rows, goal, goal_cols)

    # Build pipeline with correct indices
    _, numeric_indices, categorical_indices = identify_features(rows, goal_cols)
    model = build_pipeline(numeric_indices, categorical_indices)

    model.fit(X_train, y_train)

    # Build feature matrix for held‑out rows
    feature_names, _, _ = identify_features(rows, goal_cols)
    X_all = build_feature_matrix(rows, feature_names)
    X_numeric = X_all[:, numeric_indices].astype(float, copy=False)
    X_categorical = X_all[:, categorical_indices].astype(str, copy=False)
    X_all[:, numeric_indices] = X_numeric
    X_all[:, categorical_indices] = X_categorical
    X_test = X_all[(np.array([i + 1 for i in range(len(rows))]) % 5 == 0)]

    preds = model.predict(X_test)
    # Map predictions to row numbers
    predictions = {}
    for row_num, pred in zip(held_out_indices, preds):
        predictions[row_num] = float(pred)

    # Compute MAE against ground truth
    actuals = []
    for row_num in held_out_indices:
        r = rows[row_num - 1]
        actual = parse_float(r[goal])
        if actual is not None:
            actuals.append(actual)
    if predictions and actuals:
        mae = np.mean([abs(predictions[row] - act) for row, act in zip(held_out_indices, actuals)])
    else:
        mae = None
    return predictions, mae


def train_and_predict_with_uncertainty(rows: List[Dict[str, str]],
                                       goal: str,
                                       goal_cols: List[str]) -> Tuple[Dict[int, Dict[str, float]], float]:
    """
    Train on non‑held‑out rows and predict on held‑out rows.
    Returns a dict mapping held‑out row number to {'estimate', 'low', 'high'} and MAE.
    """
    X_train, y_train, held_out_indices = prepare_data(rows, goal, goal_cols)

    _, numeric_indices, categorical_indices = identify_features(rows, goal_cols)
    model = build_pipeline(numeric_indices, categorical_indices)

    model.fit(X_train, y_train)

    # Residual standard deviation on training data
    preds_train = model.predict(X_train)
    residuals = y_train - preds_train
    sigma = float(np.sqrt(np.mean(residuals ** 2)))

    # Predictions on held‑out rows
    feature_names, _, _ = identify_features(rows, goal_cols)
    X_all = build_feature_matrix(rows, feature_names)
    X_numeric = X_all[:, numeric_indices].astype(float, copy=False)
    X_categorical = X_all[:, categorical_indices].astype(str, copy=False)
    X_all[:, numeric_indices] = X_numeric
    X_all[:, categorical_indices] = X_categorical
    X_test = X_all[(np.array([i + 1 for i in range(len(rows))]) % 5 == 0)]

    preds = model.predict(X_test)
    predictions = {}
    for row_num, est in zip(held_out_indices, preds):
        low = float(est - 1.96 * sigma)
        high = float(est + 1.96 * sigma)
        predictions[row_num] = {"estimate": float(est), "low": low, "high": high}

    # Compute MAE against ground truth
    actuals = []
    for row_num in held_out_indices:
        r = rows[row_num - 1]
        actual = parse_float(r[goal])
        if actual is not None:
            actuals.append(actual)
    if predictions and actuals:
        mae = np.mean([abs(predictions[row]["estimate"] - act) for row, act in zip(held_out_indices, actuals)])
    else:
        mae = None
    return predictions, mae


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)

    goal_cols = ["Kloc+", "Effort-", "Defects-"]

    # NASA-03
    print("== NASA-03 ==")
    effort_preds, effort_mae = train_and_predict(rows, "Effort-", goal_cols)
    # Print some results
    print(f"Number of predictions: {len(effort_preds)}")
    if effort_preds:
        first_row = min(effort_preds.keys())
        print(f"Prediction for row {first_row}: {effort_preds[first_row]:.2f}")
    if effort_mae is not None:
        print(f"MAE: {effort_mae:.2f}")
    else:
        print("MAE: N/A")

    # ANSWER line
    answer_03 = json.dumps({"predictions": {str(k): v for k, v in effort_preds.items()},
                            "mae": effort_mae},
                           separators=(",", ":"))
    print(f"ANSWER NASA-03 {answer_03}")

    # NASA-04
    print("== NASA-04 ==")
    defect_preds, defect_mae = train_and_predict_with_uncertainty(rows, "Defects-", goal_cols)
    # Print some results
    print(f"Number of predictions: {len(defect_preds)}")
    if defect_preds:
        first_row = min(defect_preds.keys())
        d = defect_preds[first_row]
        print(f"Prediction for row {first_row}: estimate={d['estimate']:.2f}, low={d['low']:.2f}, high={d['high']:.2f}")
    if defect_mae is not None:
        print(f"MAE: {defect_mae:.2f}")
    else:
        print("MAE: N/A")

    answer_04 = json.dumps({"predictions": {str(k): v for k, v in defect_preds.items()},
                            "mae": defect_mae},
                           separators=(",", ":"))
    print(f"ANSWER NASA-04 {answer_04}")


if __name__ == "__main__":
    main()
