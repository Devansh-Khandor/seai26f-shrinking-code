#!/usr/bin/env python3
import csv
import sys
import json
import math
from typing import List, Dict, Any

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# ---------- Utility functions ----------

def read_csv(path: str):
    """
    Reads the CSV file and returns:
      rows: List of dicts with keys:
            'row'  - 1-based row number (data rows only)
            'values' - dict mapping column name to float or None
      fieldnames: list of column names
    """
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for i, row in enumerate(reader, start=1):
            cleaned = {}
            for k, v in row.items():
                if v == '?':
                    cleaned[k] = None
                else:
                    try:
                        cleaned[k] = float(v)
                    except ValueError:
                        cleaned[k] = v
            rows.append({'row': i, 'values': cleaned})
    return rows, fieldnames


def get_goal_col(fieldnames):
    for c in fieldnames:
        if c.endswith('+') or c.endswith('-'):
            return c
    return None


def get_setting_cols(fieldnames, goal_col):
    """Return list of columns to be used as features."""
    return [c for c in fieldnames if c != goal_col and not c.endswith('X')]


def get_arrival_rows(rows):
    """Return the last round(0.2 * N) rows as arrivals."""
    n = len(rows)
    arr_size = int(round(0.2 * n))
    return rows[-arr_size:]


def get_history_rows(rows):
    """Return all rows except the arrivals."""
    arr_size = int(round(0.2 * len(rows)))
    return rows[:-arr_size]


def build_dataset(rows: List[Dict[str, Any]], feature_cols: List[str], goal_col: str):
    """
    Build X, y arrays for rows that have a non-missing goal value.
    """
    X_list = []
    y_list = []
    for r in rows:
        val = r['values'].get(goal_col)
        if val is None:
            continue
        feature_vals = []
        for c in feature_cols:
            v = r['values'].get(c)
            feature_vals.append(v if v is not None else np.nan)
        X_list.append(feature_vals)
        y_list.append(val)
    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    return X, y


def predict_for_rows(rows, model, feature_cols):
    """
    Return dict mapping row number to predicted value.
    """
    preds = {}
    for r in rows:
        feature_vals = []
        for c in feature_cols:
            v = r['values'].get(c)
            feature_vals.append(v if v is not None else np.nan)
        X = np.array(feature_vals, dtype=float).reshape(1, -1)
        pred = model.predict(X)[0]
        preds[r['row']] = pred
    return preds


def mae(preds: Dict[int, float], truth_rows: List[Dict[str, Any]], goal_col: str):
    errors = []
    for r in truth_rows:
        row_num = r['row']
        true_val = r['values'][goal_col]
        if true_val is None or row_num not in preds:
            continue
        errors.append(abs(preds[row_num] - true_val))
    return sum(errors) / len(errors) if errors else 0.0


# ---------- Main logic for APACHE-03 and APACHE-04 ----------

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)

    path = sys.argv[1]
    rows, fieldnames = read_csv(path)

    goal_col = get_goal_col(fieldnames)
    if goal_col is None:
        print("No goal column found.")
        sys.exit(1)

    setting_cols = get_setting_cols(fieldnames, goal_col)

    # ---- APACHE-03 ----
    print("== APACHE-03 ==")

    # Determine held-out rows: row numbers divisible by 5
    held_out_rows = [r for r in rows if r['row'] % 5 == 0]
    # Build training set: all rows except held-out
    training_rows = [r for r in rows if r not in held_out_rows]

    # Build dataset for training
    X_train, y_train = build_dataset(training_rows, setting_cols, goal_col)

    # Use a simple regression pipeline with mean imputation
    model = make_pipeline(SimpleImputer(strategy='median'), LinearRegression())
    model.fit(X_train, y_train)

    # Predict for held-out rows
    predictions = predict_for_rows(held_out_rows, model, setting_cols)

    # Compute mae on held-out rows
    mae_val = mae(predictions, held_out_rows, goal_col)

    # Prepare predictions dict with row numbers as strings
    pred_dict = {str(k): round(v, 3) for k, v in predictions.items()}

    print(f"MAE: {mae_val:.3f}")
    print(f"Predictions: {pred_dict}")
    print(f"ANSWER APACHE-03 {{\"predictions\": {json.dumps(pred_dict, separators=(',', ':'))}, \"mae\": {mae_val:.3f}}}")

    # ---- APACHE-04 ----
    print("\n== APACHE-04 ==")

    history_rows = get_history_rows(rows)

    # Filter history rows with known goal value
    history_rows = [r for r in history_rows if r['values'].get(goal_col) is not None]

    # Precompute predictions for all rows (including history) for difference estimation
    all_pred = predict_for_rows(rows, model, setting_cols)

    # Find pairs that differ by exactly one setting
    pairs = []
    for i, r1 in enumerate(history_rows):
        for j in range(i + 1, len(history_rows)):
            r2 = history_rows[j]
            diff_settings = []
            for c in setting_cols:
                v1 = r1['values'].get(c)
                v2 = r2['values'].get(c)
                # treat missing as not differing
                if v1 is None or v2 is None:
                    continue
                if v1 != v2:
                    diff_settings.append(c)
            if len(diff_settings) == 1:
                setting = diff_settings[0]
                pred_a = all_pred.get(r1['row'], 0.0)
                pred_b = all_pred.get(r2['row'], 0.0)
                diff = pred_b - pred_a
                low = diff - mae_val
                high = diff + mae_val
                pairs.append({
                    'a': r1['row'],
                    'b': r2['row'],
                    'setting': setting,
                    'difference': round(diff, 3),
                    'low': round(low, 3),
                    'high': round(high, 3)
                })

    # Sort by absolute difference descending, pick top 5
    pairs.sort(key=lambda x: abs(x['difference']), reverse=True)
    top_pairs = pairs[:5]

    # Output
    for p in top_pairs:
        print(f"{p['a']} vs {p['b']}: {p['setting']} diff={p['difference']}")
    print(f"ANSWER APACHE-04 {{\"pairs\": {json.dumps(top_pairs, separators=(',', ':'))}}}")

if __name__ == "__main__":
    main()
