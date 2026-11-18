#!/usr/bin/env python3
import csv
import json
import sys
import math
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def is_goal(col_name: str) -> bool:
    return (col_name.endswith('+') or col_name.endswith('-')) and not col_name.endswith('X')

def is_maximize(col_name: str) -> bool:
    return col_name.endswith('+')

def parse_value(val: str):
    if val == '?' or val.strip() == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

# ----------------------------------------------------------------------
# Read data
# ----------------------------------------------------------------------
def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            rows.append(r)
    return rows, reader.fieldnames

# ----------------------------------------------------------------------
# Prepare features and targets
# ----------------------------------------------------------------------
def build_features_and_target(rows, fieldnames, target_col, non_goal_cols):
    X = []
    y = []
    for row in rows:
        # skip rows where target is missing
        tgt_val = parse_value(row[target_col])
        if tgt_val is None:
            continue
        # collect features
        feat = []
        for col in non_goal_cols:
            v = parse_value(row[col])
            feat.append(v)
        X.append(feat)
        y.append(tgt_val)
    return np.array(X, dtype=float), np.array(y, dtype=float)

# ----------------------------------------------------------------------
# Train regression model and return predictions + stats
# ----------------------------------------------------------------------
def train_and_predict(rows, fieldnames, target_col, non_goal_cols):
    X, y = build_features_and_target(rows, fieldnames, target_col, non_goal_cols)

    # Build pipeline
    pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    pipe.fit(X, y)
    y_pred = pipe.predict(X)

    residuals = y - y_pred
    # use unbiased std (ddof=1) if enough samples
    if len(residuals) > 1:
        residual_std = np.std(residuals, ddof=1)
    else:
        residual_std = 0.0

    return pipe, residual_std

# ----------------------------------------------------------------------
# Main processing
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    rows, fieldnames = read_csv(path)

    # Identify goal columns
    goal_cols = [c for c in fieldnames if is_goal(c)]
    non_goal_cols = [c for c in fieldnames if not is_goal(c)]

    # Determine held‑out rows (row numbers divisible by 5, 1‑based)
    held_out_indices = [i + 1 for i in range(len(rows)) if (i + 1) % 5 == 0]
    history_rows = [rows[i] for i in range(len(rows)) if (i + 1) not in held_out_indices]

    # ------------------------------------------------------------------
    # CAR‑03: predict Mpg+
    # ------------------------------------------------------------------
    mpg_target = 'Mpg+'
    mpg_pipe, mpg_std = train_and_predict(history_rows, fieldnames, mpg_target, non_goal_cols)

    mpg_predictions = {}
    mpg_errors = []
    for idx in held_out_indices:
        row = rows[idx - 1]
        # Prepare feature vector for this row
        feat = [parse_value(row[c]) for c in non_goal_cols]
        feat_np = np.array([feat], dtype=float)
        pred = mpg_pipe.predict(feat_np)[0]
        mpg_predictions[str(idx)] = pred
        true_val = parse_value(row[mpg_target])
        if true_val is not None:
            mpg_errors.append(abs(pred - true_val))
    mae_mpg = sum(mpg_errors) / len(mpg_errors) if mpg_errors else 0.0

    # ------------------------------------------------------------------
    # CAR‑04: predict Lbs-
    # ------------------------------------------------------------------
    lbs_target = 'Lbs-'
    lbs_pipe, lbs_std = train_and_predict(history_rows, fieldnames, lbs_target, non_goal_cols)

    lbs_predictions = {}
    for idx in held_out_indices:
        row = rows[idx - 1]
        feat = [parse_value(row[c]) for c in non_goal_cols]
        feat_np = np.array([feat], dtype=float)
        pred = lbs_pipe.predict(feat_np)[0]
        # 95% confidence interval using residual std
        low = pred - 1.96 * lbs_std
        high = pred + 1.96 * lbs_std
        lbs_predictions[str(idx)] = {"estimate": pred, "low": low, "high": high}

    # ------------------------------------------------------------------
    # Output results
    # ------------------------------------------------------------------
    print("== CAR-03 ==")
    # Show at least three lines of results
    for i, (row_num, pred) in enumerate(mpg_predictions.items()):
        if i >= 3:
            break
        true_val = parse_value(rows[int(row_num)-1][mpg_target])
        print(f"Row {row_num}: predicted Mpg+={pred:.6f}", end='')
        if true_val is not None:
            print(f", actual Mpg+={true_val:.6f}")
        else:
            print("")

    print(f"MAE={mae_mpg:.6f}")
    print(f"ANSWER CAR-03 {json.dumps({'predictions': mpg_predictions, 'mae': mae_mpg}, ensure_ascii=False)}")

    print("\n== CAR-04 ==")
    for i, (row_num, pred_dict) in enumerate(lbs_predictions.items()):
        if i >= 3:
            break
        est = pred_dict["estimate"]
        low = pred_dict["low"]
        high = pred_dict["high"]
        print(f"Row {row_num}: estimate Lbs-={est:.6f} low={low:.6f} high={high:.6f}")

    print(f"ANSWER CAR-04 {json.dumps({'predictions': lbs_predictions}, ensure_ascii=False)}")

if __name__ == "__main__":
    main()
