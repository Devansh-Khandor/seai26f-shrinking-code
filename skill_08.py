#!/usr/bin/env python3
"""
skill_08.py
------------
Command line tool that reads a CSV file of NASA software project data
and answers two new requests:

    NASA-15: For a project proposal predicted to require excessive effort,
             identify plausible changes to adjustable planning or process
             characteristics that might reduce its predicted effort.
    NASA-16: For a project proposal with a high predicted defect count,
             identify the smallest plausible combination of changes to
             adjustable planning or process characteristics that could
             improve its predicted quality, while explaining potential
             tradeoffs.

Usage:
    python skill_08.py <csv path>
"""

import csv
import json
import sys
import math
import itertools

import numpy as np

from skill_02 import (
    load_csv,
    parse_float,
    identify_features,
    build_pipeline,
)


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def build_feature_matrix(rows, feature_names):
    """
    Return a numpy array of shape (n_rows, n_features) where numeric values
    are floats (or np.nan) and categorical values are strings. Mixed dtype is used.
    """
    data = []
    for r in rows:
        row_vals = []
        for col in feature_names:
            v = r[col]
            if v == "?":
                # Keep as np.nan; will be converted later
                row_vals.append(np.nan)
            else:
                row_vals.append(v)
        data.append(row_vals)
    return np.array(data, dtype=object)


def predict_for_row(row, model, feature_names, numeric_indices, categorical_indices):
    """
    Predict a single row using the supplied model.
    Returns a float prediction.
    """
    row_arr = np.array([row], dtype=object)
    # Convert numeric columns
    if numeric_indices:
        row_arr[:, numeric_indices] = row_arr[:, numeric_indices].astype(float, copy=False)
    # Categorical columns already strings
    pred = model.predict(row_arr)
    return float(pred[0])


def get_unique_values(rows, col):
    """Return a set of non-missing values that appear in the column."""
    return {r[col] for r in rows if r[col] != "?"}


def round_float(v, ndigits=2):
    """Return v rounded to ndigits, or None if v is None."""
    if v is None:
        return None
    return round(float(v), ndigits)


# --------------------------------------------------------------------------- #
# NASA-15 logic
# --------------------------------------------------------------------------- #

def nasa_15(rows, feature_names, numeric_indices, categorical_indices, model):
    # Predict effort for all rows
    X_all = build_feature_matrix(rows, feature_names)
    # Convert numeric parts
    if numeric_indices:
        X_all[:, numeric_indices] = X_all[:, numeric_indices].astype(float, copy=False)
    effort_preds = model.predict(X_all).reshape(-1)

    # Find row with highest predicted effort
    max_eff_idx = int(np.argmax(effort_preds))
    target_row = rows[max_eff_idx]
    target_row_num = max_eff_idx + 1
    effort_before = round_float(effort_preds[max_eff_idx])

    # Identify adjustable columns (non-goal, non-X, not Kloc+)
    adjustable = [c for c in feature_names if c != "Kloc+"]

    best_change = None
    best_after = None
    best_reduction = 0.0

    for col in adjustable:
        current_val = target_row[col]
        unique_vals = get_unique_values(rows, col)
        for val in unique_vals:
            if val == current_val:
                continue
            mod_row = target_row.copy()
            mod_row[col] = val
            after_pred = predict_for_row(mod_row, model, feature_names, numeric_indices, categorical_indices)
            reduction = round_float(effort_before - after_pred)
            if reduction is None:
                continue
            if reduction > best_reduction:
                best_reduction = reduction
                best_change = {col: [current_val, val]}
                best_after = round_float(after_pred)

    # If no reduction found, pick a no-op change
    if best_change is None:
        best_change = {}
        best_after = effort_before

    result = {
        "project": target_row_num,
        "changes": best_change,
        "predicted_effort": {"before": effort_before, "after": best_after},
    }
    return result


# --------------------------------------------------------------------------- #
# NASA-16 logic
# --------------------------------------------------------------------------- #

def nasa_16(rows, feature_names, numeric_indices, categorical_indices, model):
    # Predict defects for all rows
    X_all = build_feature_matrix(rows, feature_names)
    if numeric_indices:
        X_all[:, numeric_indices] = X_all[:, numeric_indices].astype(float, copy=False)
    defect_preds = model.predict(X_all).reshape(-1)

    # Find row with highest predicted defect
    max_def_idx = int(np.argmax(defect_preds))
    target_row = rows[max_def_idx]
    target_row_num = max_def_idx + 1
    defect_before = round_float(defect_preds[max_def_idx])

    # Also get effort predictions for this row
    effort_preds = model.predict(X_all).reshape(-1)
    effort_before = round_float(effort_preds[max_def_idx])

    adjustable = [c for c in feature_names if c != "Kloc+"]

    best_combo = None
    best_defect = defect_before
    best_effort = effort_before

    # Try 1, 2, 3 column changes
    for k in range(1, 4):
        col_combos = list(itertools.combinations(adjustable, k))
        for cols in col_combos:
            # Get all possible value combinations for these columns
            value_lists = [sorted(get_unique_values(rows, col)) for col in cols]
            for vals in itertools.product(*value_lists):
                mod_row = target_row.copy()
                change_dict = {}
                for col, new_val in zip(cols, vals):
                    current_val = mod_row[col]
                    if current_val == new_val:
                        # no change for this column
                        continue
                    mod_row[col] = new_val
                    change_dict[col] = [current_val, new_val]
                if not change_dict:
                    continue
                # Predict after changes
                new_defect = predict_for_row(mod_row, model, feature_names, numeric_indices, categorical_indices)
                new_effort = effort_before  # predict effort separately
                # For effort, we need separate model? Our model is same for effort and defects.
                # So we can predict effort also from the same model? No, our model currently trained to predict the target variable we passed.
                # We need separate models for effort and defects. For simplicity, train two models: effort_model and defect_model.
                # But we already have one model built on all goals? Actually we built one model earlier. Let's rebuild two models.

        # If a better combo found, break loops
        if best_combo is not None:
            break

    # The above approach is incomplete because we need separate models. We'll reimplement correctly below.

    # -------------------------------------------------------------------- #
    # Rebuild models for effort and defects separately
    # -------------------------------------------------------------------- #
    # Build training data for both goals
    def build_model_for_goal(goal_col):
        # Build feature matrix
        X_all = build_feature_matrix(rows, feature_names)
        if numeric_indices:
            X_all[:, numeric_indices] = X_all[:, numeric_indices].astype(float, copy=False)
        # y: only rows with goal present
        y_all = np.array([parse_float(r[goal_col]) for r in rows], dtype=float)
        train_mask = ~np.isnan(y_all)
        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        pipe = build_pipeline(numeric_indices, categorical_indices)
        pipe.fit(X_train, y_train)
        return pipe

    effort_model = build_model_for_goal("Effort-")
    defect_model = build_model_for_goal("Defects-")

    # Recompute predictions
    X_all = build_feature_matrix(rows, feature_names)
    if numeric_indices:
        X_all[:, numeric_indices] = X_all[:, numeric_indices].astype(float, copy=False)
    effort_preds = effort_model.predict(X_all).reshape(-1)
    defect_preds = defect_model.predict(X_all).reshape(-1)

    effort_before = round_float(effort_preds[max_def_idx])
    defect_before = round_float(defect_preds[max_def_idx])

    # Now search for best combination
    best_combo = None
    best_defect = defect_before
    best_effort = effort_before

    for k in range(1, 4):
        col_combos = list(itertools.combinations(adjustable, k))
        for cols in col_combos:
            value_lists = [sorted(get_unique_values(rows, col)) for col in cols]
            for vals in itertools.product(*value_lists):
                mod_row = target_row.copy()
                change_dict = {}
                for col, new_val in zip(cols, vals):
                    current_val = mod_row[col]
                    if current_val == new_val:
                        continue
                    mod_row[col] = new_val
                    change_dict[col] = [current_val, new_val]
                if not change_dict:
                    continue
                new_defect = defect_model.predict([mod_row])[0]
                new_effort = effort_model.predict([mod_row])[0]
                new_defect = round_float(new_defect)
                new_effort = round_float(new_effort)
                if new_defect < best_defect:
                    best_defect = new_defect
                    best_effort = new_effort
                    best_combo = change_dict
        if best_combo is not None:
            break

    if best_combo is None:
        best_combo = {}
        best_defect = defect_before
        best_effort = effort_before

    result = {
        "project": target_row_num,
        "changes": best_combo,
        "predicted_defects": {"before": defect_before, "after": best_defect},
        "predicted_effort": {"before": effort_before, "after": best_effort},
    }
    return result


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv(csv_path)

    if not rows:
        print("No data rows found.")
        sys.exit(1)

    goal_cols = ["Kloc+", "Effort-", "Defects-"]

    # Identify features and indices
    feature_names, numeric_indices, categorical_indices = identify_features(rows, goal_cols)

    # Build a generic model for predictions (will be refitted later)
    # For NASA-15 we only need effort model; for NASA-16 both effort and defect models
    # We'll construct both below

    # --------------------------------------------------------------------- #
    # NASA-15
    # --------------------------------------------------------------------- #
    print("== NASA-15 ==")

    # Build effort model
    def build_effort_model():
        X_all = build_feature_matrix(rows, feature_names)
        if numeric_indices:
            X_all[:, numeric_indices] = X_all[:, numeric_indices].astype(float, copy=False)
        y_all = np.array([parse_float(r["Effort-"]) for r in rows], dtype=float)
        train_mask = ~np.isnan(y_all)
        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        pipe = build_pipeline(numeric_indices, categorical_indices)
        pipe.fit(X_train, y_train)
        return pipe

    effort_model = build_effort_model()

    res15 = nasa_15(rows, feature_names, numeric_indices, categorical_indices, effort_model)
    # Output some lines
    print(f"Project with highest predicted effort: {res15['project']}")
    if res15['changes']:
        for col, (old, new) in res15['changes'].items():
            print(f"Change {col} from {old} to {new}")
    else:
        print("No change found")
    print(f"Predicted effort before: {res15['predicted_effort']['before']:.2f}")
    print(f"Predicted effort after: {res15['predicted_effort']['after']:.2f}")
    answer_15 = {
        "project": res15["project"],
        "changes": res15["changes"],
        "predicted_effort": res15["predicted_effort"],
    }
    print(f"ANSWER NASA-15 {json.dumps(answer_15, separators=(',', ':'))}")

    # --------------------------------------------------------------------- #
    # NASA-16
    # --------------------------------------------------------------------- #
    print("\n== NASA-16 ==")

    # Build defect model
    def build_defect_model():
        X_all = build_feature_matrix(rows, feature_names)
        if numeric_indices:
            X_all[:, numeric_indices] = X_all[:, numeric_indices].astype(float, copy=False)
        y_all = np.array([parse_float(r["Defects-"]) for r in rows], dtype=float)
        train_mask = ~np.isnan(y_all)
        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        pipe = build_pipeline(numeric_indices, categorical_indices)
        pipe.fit(X_train, y_train)
        return pipe

    defect_model = build_defect_model()

    res16 = nasa_16(rows, feature_names, numeric_indices, categorical_indices, defect_model)
    # Output lines
    print(f"Project with highest predicted defects: {res16['project']}")
    if res16['changes']:
        for col, (old, new) in res16['changes'].items():
            print(f"Change {col} from {old} to {new}")
    else:
        print("No changes found")
    print(f"Predicted defects before: {res16['predicted_defects']['before']:.2f}")
    print(f"Predicted defects after: {res16['predicted_defects']['after']:.2f}")
    print(f"Predicted effort before: {res16['predicted_effort']['before']:.2f}")
    print(f"Predicted effort after: {res16['predicted_effort']['after']:.2f}")
    answer_16 = {
        "project": res16["project"],
        "changes": res16["changes"],
        "predicted_defects": res16["predicted_defects"],
        "predicted_effort": res16["predicted_effort"],
    }
    print(f"ANSWER NASA-16 {json.dumps(answer_16, separators=(',', ':'))}")


if __name__ == "__main__":
    main()
