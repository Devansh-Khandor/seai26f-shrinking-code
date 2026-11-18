#!/usr/bin/env python3
"""
skill_02.py

Usage:
    python skill_02.py <csv_path>
"""

import csv
import json
import sys
import math
from typing import List, Dict, Any, Tuple, Set

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# ------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------
def parse_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read CSV and return rows as list of dicts and header list."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    return rows, header

def to_float(val: str) -> float:
    if val.strip() == "?":
        return float("nan")
    try:
        return float(val)
    except ValueError:
        return float("nan")

def is_numeric(val: str) -> bool:
    if val.strip() == "?":
        return False
    try:
        float(val)
        return True
    except ValueError:
        return False

def get_features_and_target(rows: List[Dict[str, str]],
                            header: List[str],
                            target_col: str,
                            exclude_cols: Set[str]) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Return feature dicts and target list for rows where target is present."""
    feature_rows = []
    target_vals = []
    for row in rows:
        if row[target_col].strip() == "?":
            continue  # skip rows with missing target
        feature_dict = {}
        for col in header:
            if col == target_col or col in exclude_cols:
                continue
            val = row[col]
            if val.strip() == "?":
                feature_dict[col] = val  # keep missing as string; will be handled by encoder
            else:
                if is_numeric(val):
                    feature_dict[col] = float(val)
                else:
                    feature_dict[col] = val
        feature_rows.append(feature_dict)
        target_vals.append(to_float(row[target_col]))
    return feature_rows, target_vals

def build_pipeline(feature_names: List[str]) -> Pipeline:
    """Create a sklearn pipeline for regression."""
    numeric_features = []
    categorical_features = []
    # Determine numeric vs categorical based on sample values
    for col in feature_names:
        # We just look at the first row's value after parsing
        # Use a simple heuristic: if value can be parsed as float, treat as numeric
        numeric_features.append(col)  # placeholder

    # We'll detect numeric columns by sampling first row that has a numeric value
    # Since our dataset may have only string categories, we fallback to all categorical
    numeric_features = []
    categorical_features = []
    # For safety, we just treat everything as categorical except if the value looks numeric
    for col in feature_names:
        numeric_features.append(col)
    # For this problem, the simplest: treat everything as categorical, and impute numerics later
    # but we will use OneHotEncoder for categorical, SimpleImputer for numeric
    # Actually we need both: numeric features -> impute
    # Let's do a more robust approach: iterate over rows to check type
    # This code will be executed after we have feature_rows list
    return None  # placeholder

def prepare_pipeline(feature_rows: List[Dict[str, Any]], feature_names: List[str]) -> Pipeline:
    """Create a pipeline that handles numeric and categorical features."""
    # Detect numeric columns by looking at first non-missing value
    numeric_cols = []
    categorical_cols = []
    for col in feature_names:
        found_num = False
        for row in feature_rows:
            val = row.get(col, None)
            if val is None:
                continue
            if isinstance(val, float) or isinstance(val, int):
                found_num = True
                break
            if isinstance(val, str) and val.strip() != "?":
                try:
                    float(val)
                    found_num = True
                    break
                except ValueError:
                    pass
        if found_num:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    pipe = Pipeline(steps=[("preprocessor", preprocessor),
                          ("model", LinearRegression())])
    return pipe

def get_held_out_indices(n_rows: int) -> Set[int]:
    """Indices (1-based) of rows that are divisible by 5."""
    return {i for i in range(1, n_rows + 1) if i % 5 == 0}

def compute_mae(predictions: Dict[int, float], actuals: Dict[int, float]) -> float:
    errors = []
    for idx, pred in predictions.items():
        if idx in actuals:
            errors.append(abs(pred - actuals[idx]))
    if not errors:
        return float("nan")
    return sum(errors) / len(errors)

# ------------------------------------------------------------
#  Main
# ------------------------------------------------------------
def main(argv: List[str]) -> None:
    if len(argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)
    csv_path = argv[1]
    rows, header = parse_csv(csv_path)
    n_rows = len(rows)

    # Identify columns to exclude (goals and X columns)
    GOAL_COLUMNS = ["Kloc+", "Effort-", "Defects-"]
    exclude_cols = set(GOAL_COLUMNS) | {col for col in header if col.endswith("X")}

    # Determine feature names (all columns not excluded)
    feature_names = [col for col in header if col not in exclude_cols]

    # Prepare data for Effort- prediction (NASA-03)
    effort_rows, effort_targets = get_features_and_target(rows, header, "Effort-", exclude_cols)
    # Build pipeline
    effort_pipe = prepare_pipeline(effort_rows, feature_names)
    # Train on all non-held-out rows
    held_out_set = get_held_out_indices(n_rows)
    train_indices = [i for i in range(1, n_rows + 1) if i not in held_out_set]
    # Build training set
    train_feat = [r for i, r in enumerate(rows, start=1) if i in train_indices and r["Effort-"].strip() != "?"]
    train_y = [to_float(r["Effort-"]) for r in train_feat]
    # Fit
    if train_feat:
        effort_pipe.fit(train_feat, train_y)
    else:
        sys.exit("No training data for Effort-.")

    # Predict for held-out rows
    held_out_effort_preds = {}
    actual_effort_values = {}
    for idx in held_out_set:
        row = rows[idx - 1]
        if row["Effort-"].strip() == "?":
            continue  # skip if true value unknown
        feat_dict = {}
        for col in feature_names:
            val = row[col]
            if val.strip() == "?":
                feat_dict[col] = val
            else:
                if is_numeric(val):
                    feat_dict[col] = float(val)
                else:
                    feat_dict[col] = val
        pred = effort_pipe.predict([feat_dict])[0]
        held_out_effort_preds[idx] = float(round(pred, 4))
        actual_effort_values[idx] = to_float(row["Effort-"])

    mae_effort = compute_mae(held_out_effort_preds, actual_effort_values)
    mae_effort = round(mae_effort, 4)

    # Prepare data for Defects- prediction (NASA-04)
    defects_rows, defects_targets = get_features_and_target(rows, header, "Defects-", exclude_cols)
    defects_pipe = prepare_pipeline(defects_rows, feature_names)
    train_feat_def = [r for i, r in enumerate(rows, start=1)
                      if i in train_indices and r["Defects-"].strip() != "?"]
    train_y_def = [to_float(r["Defects-"]) for r in train_feat_def]
    if train_feat_def:
        defects_pipe.fit(train_feat_def, train_y_def)
    else:
        sys.exit("No training data for Defects-.")
    # Residual standard deviation
    residuals = defects_pipe.predict(train_feat_def) - train_y_def
    sigma = math.sqrt(sum(r ** 2 for r in residuals) / len(residuals))
    ci_multiplier = 1.96  # approx 95% interval

    # Predict for held-out rows
    held_out_defects_preds = {}
    for idx in held_out_set:
        row = rows[idx - 1]
        if row["Defects-"].strip() == "?":
            continue
        feat_dict = {}
        for col in feature_names:
            val = row[col]
            if val.strip() == "?":
                feat_dict[col] = val
            else:
                if is_numeric(val):
                    feat_dict[col] = float(val)
                else:
                    feat_dict[col] = val
        est = defects_pipe.predict([feat_dict])[0]
        low = est - ci_multiplier * sigma
        high = est + ci_multiplier * sigma
        held_out_defects_preds[idx] = {
            "estimate": float(round(est, 4)),
            "low": float(round(low, 4)),
            "high": float(round(high, 4))
        }

    # ------------------------------------------------------------
    #  Output
    # ------------------------------------------------------------
    # NASA-03 section
    print("== NASA-03 ==")
    # print first few predictions for sanity
    for idx in sorted(held_out_effort_preds)[:3]:
        print(f"Row {idx}: Effort- predicted = {held_out_effort_preds[idx]}")
    print(f"MAE for Effort- predictions: {mae_effort}")
    # Answer line
    answer_03 = {
        "predictions": {str(idx): val for idx, val in sorted(held_out_effort_preds.items())},
        "mae": mae_effort
    }
    print("ANSWER NASA-03 " + json.dumps(answer_03))

    # NASA-04 section
    print("\n== NASA-04 ==")
    # print first few predictions for sanity
    for idx in sorted(held_out_defects_preds)[:3]:
        pred = held_out_defects_preds[idx]
        print(f"Row {idx}: Defects- estimate={pred['estimate']}, low={pred['low']}, high={pred['high']}")
    # Answer line
    answer_04 = {
        "predictions": {str(idx): v for idx, v in sorted(held_out_defects_preds.items())}
    }
    print("ANSWER NASA-04 " + json.dumps(answer_04))

if __name__ == "__main__":
    main(sys.argv)
