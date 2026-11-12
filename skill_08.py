#!/usr/bin/env python3
"""
skill_08.py - NASA-15 and NASA-16: Suggested plan/process changes to reduce effort or defects.

Usage:
    python skill_08.py <csv_path>
"""

import csv
import sys
import numpy as np
from statistics import mean, stdev
from typing import List, Dict, Tuple

# ----------------------------------------------------------------------
# Reuse utilities from earlier skills
# ----------------------------------------------------------------------
try:
    from skill_01 import load_csv, get_numeric_column
except Exception:
    # Minimal local implementation if skill_01 is not importable
    def load_csv(csv_path: str) -> Tuple[List[Dict[str, str]], List[str]]:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            rows = [row for row in reader]
        return rows, header

    def get_numeric_column(rows: List[Dict[str, str]], col: str) -> List[float]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[col]))
            except (ValueError, TypeError):
                pass
        return vals

try:
    from skill_02 import (
        identify_numeric_columns,
        build_mappings,
        encode_row,
        prepare_data,
        train_regression,
        predict_with_uncertainty,
    )
except Exception:
    # Minimal local implementation if skill_02 is not importable
    from sklearn.linear_model import LinearRegression

    def identify_numeric_columns(rows: List[Dict[str, str]], header: List[str]) -> Tuple[set, set]:
        numeric = set()
        categorical = set()
        for col in header:
            if col == 'idX':
                continue
            sample = None
            for r in rows:
                val = r.get(col, '')
                if val != '':
                    sample = val
                    break
            if sample is None:
                categorical.add(col)
            else:
                try:
                    float(sample)
                    numeric.add(col)
                except ValueError:
                    categorical.add(col)
        return numeric, categorical

    def build_mappings(rows: List[Dict[str, str]], categorical: set) -> Dict[str, Dict[str, int]]:
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
        vec = []
        for col in numeric:
            try:
                vec.append(float(r.get(col, '0')))
            except ValueError:
                vec.append(0.0)
        for col in categorical:
            val = r.get(col, '')
            mapping = mappings[col]
            vec.append(float(mapping.get(val, -1)))  # unseen -> -1
        return vec

    def prepare_data(rows: List[Dict[str, str]], header: List[str], target_col: str):
        numeric, categorical = identify_numeric_columns(rows, header)
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

    def train_regression(X: np.ndarray, y: np.ndarray):
        model = LinearRegression()
        model.fit(X, y)
        return model

    def predict_with_uncertainty(model, X_train, y_train, X_new):
        y_pred_train = model.predict(X_train)
        residuals = y_train - y_pred_train
        std_resid = np.sqrt(np.mean(residuals ** 2))
        y_pred_new = model.predict(X_new)[0]
        uncertainty = 2 * std_resid
        return y_pred_new, uncertainty, std_resid

# ----------------------------------------------------------------------
# Helper: compute top changes to reduce a target prediction
# ----------------------------------------------------------------------
def get_top_changes(
    idx: int,
    rows: List[Dict[str, str]],
    X: np.ndarray,
    model,
    mappings: Dict[str, Dict[str, int]],
    numeric: set,
    categorical: set,
    target_name: str,
    top_n: int = 3,
) -> List[Tuple[str, str, str, float, float, float]]:
    """
    Return a list of tuples:
    (col, from_value, to_value, reduction, current_pred, new_pred)
    sorted by reduction descending.
    """
    current_pred = model.predict(X[idx].reshape(1, -1))[0]
    row = rows[idx]
    changes = []

    for col in categorical:
        current_val = row.get(col, '')
        for new_val in mappings[col]:
            if new_val == current_val:
                continue
            mod_row = row.copy()
            mod_row[col] = new_val
            new_vec = encode_row(mod_row, numeric, categorical, mappings)
            new_pred = model.predict(np.array([new_vec]))[0]
            reduction = current_pred - new_pred
            if reduction > 0:
                changes.append((col, current_val, new_val, reduction, current_pred, new_pred))

    changes.sort(key=lambda x: x[3], reverse=True)
    return changes[:top_n]

# ----------------------------------------------------------------------
# NASA-15: Reduce excessive effort
# ----------------------------------------------------------------------
def report_nasa_15(rows: List[Dict[str, str]], header: List[str]) -> None:
    print("\n== NASA-15 ==")

    # Prepare effort model
    X_effort, y_effort, num_eff, cat_eff, maps_eff = prepare_data(rows, header, 'Effort-')
    model_eff = train_regression(X_effort, y_effort)

    # Predictions and threshold
    preds_eff = model_eff.predict(X_effort)
    effort_mean = np.mean(preds_eff)
    effort_std = np.std(preds_eff, ddof=1)
    effort_thresh = effort_mean + 2 * effort_std

    # Find projects exceeding threshold
    candidates = [i for i, p in enumerate(preds_eff) if p > effort_thresh]
    if not candidates:
        print("No projects identified as excessively high effort.")
        return

    idx = candidates[0]  # pick first
    proj = rows[idx]
    print(f"Project {proj.get('idX')} is predicted to require excessive effort ({preds_eff[idx]:.2f} units).")
    changes = get_top_changes(
        idx, rows, X_effort, model_eff, maps_eff, num_eff, cat_eff, 'Effort-', top_n=3
    )
    if not changes:
        print("  No single categorical changes found that reduce effort.")
        return
    for col, old, new, red, cur, new_pred in changes:
        print(
            f"  Change '{col}' from '{old}' to '{new}' reduces predicted effort by {red:.2f} units "
            f"(current: {cur:.2f} → new: {new_pred:.2f})."
        )

# ----------------------------------------------------------------------
# NASA-16: Reduce high predicted defect count
# ----------------------------------------------------------------------
def report_nasa_16(rows: List[Dict[str, str]], header: List[str]) -> None:
    print("\n== NASA-16 ==")

    # Prepare defect model
    X_defect, y_defect, num_def, cat_def, maps_def = prepare_data(rows, header, 'Defects-')
    model_def = train_regression(X_defect, y_defect)

    # Predictions and threshold
    preds_def = model_def.predict(X_defect)
    defect_mean = np.mean(preds_def)
    defect_std = np.std(preds_def, ddof=1)
    defect_thresh = defect_mean + 2 * defect_std

    # Find projects exceeding threshold
    candidates = [i for i, p in enumerate(preds_def) if p > defect_thresh]
    if not candidates:
        print("No projects identified as having excessively high defect counts.")
        return

    idx = candidates[0]  # pick first
    proj = rows[idx]
    print(f"Project {proj.get('idX')} is predicted to have high defect count ({preds_def[idx]:.2f} defects).")
    changes = get_top_changes(
        idx, rows, X_defect, model_def, maps_def, num_def, cat_def, 'Defects-', top_n=3
    )
    if not changes:
        print("  No single categorical changes found that reduce defect count.")
        return
    for col, old, new, red, cur, new_pred in changes:
        print(
            f"  Change '{col}' from '{old}' to '{new}' reduces predicted defects by {red:.2f} "
            f"(current: {cur:.2f} → new: {new_pred:.2f})."
        )
        print(
            f"  (Trade‑off: altering '{col}' may affect scheduling or resource allocation.)"
        )

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows, header = load_csv(csv_path)

    report_nasa_15(rows, header)
    report_nasa_16(rows, header)

if __name__ == "__main__":
    main()
