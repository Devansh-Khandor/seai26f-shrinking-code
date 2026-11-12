#!/usr/bin/env python3
"""
Skill 08: NASA-15 and NASA-16 – Suggested changes to reduce effort and defects
Author: Your Name
"""

import csv
import sys
import math
import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Utility – read CSV and convert numeric values where possible
# ----------------------------------------------------------------------
def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = []
        for r in reader:
            row = {}
            for k, v in r.items():
                v = v.strip() if v is not None else ''
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return header, rows

# ----------------------------------------------------------------------
# Build feature matrix for a given target column
# ----------------------------------------------------------------------
def build_matrix(rows, target_col, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    numeric_cols = []
    for key in rows[0].keys():
        if key in exclude_cols or key == target_col:
            continue
        # column is numeric if all non‑None values are numeric
        if all(isinstance(r.get(key, None), (int, float)) or r.get(key, None) is None
               for r in rows):
            numeric_cols.append(key)

    X_rows = []
    y_rows = []
    for r in rows:
        tgt = r.get(target_col)
        if not isinstance(tgt, (int, float)):
            continue
        features = []
        missing = False
        for c in numeric_cols:
            val = r.get(c)
            if isinstance(val, (int, float)):
                features.append(val)
            else:
                missing = True
                break
        if missing:
            continue
        X_rows.append(features)
        y_rows.append(tgt)
    if not X_rows:
        raise ValueError(f"No valid rows for target '{target_col}'.")
    return np.array(X_rows), np.array(y_rows), numeric_cols

# ----------------------------------------------------------------------
# Identify adjustable columns (excluding IDs, size, target, and year/month)
# ----------------------------------------------------------------------
def get_adjustable_columns(rows):
    exclude = {'idX', 'YearX', 'MonthsX', 'Kloc+', 'Effort-', 'Defects-'}
    numeric_cols, categorical_cols = [], []
    for key in rows[0].keys():
        if key in exclude:
            continue
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if not vals:
            continue
        if all(isinstance(v, (int, float)) for v in vals):
            numeric_cols.append(key)
        else:
            categorical_cols.append(key)
    return numeric_cols, categorical_cols

# ----------------------------------------------------------------------
# Compute best value for each adjustable column (lowest mean effort)
# ----------------------------------------------------------------------
def best_values_for_effort(rows, numeric_cols, categorical_cols):
    # Build regression model once for efficiency comparison
    X, y, feat_names = build_matrix(rows, 'Effort-', exclude_cols=['idX', 'YearX', 'MonthsX'])
    model = LinearRegression().fit(X, y)

    best_val = {}
    # Numeric: choose min observed value
    for col in numeric_cols:
        vals = [r[col] for r in rows if isinstance(r[col], (int, float))]
        if vals:
            best_val[col] = min(vals)
    # Categorical: choose value that yields lowest mean effort in data
    for col in categorical_cols:
        groups = {}
        for r in rows:
            val = r[col]
            eff = r.get('Effort-', None)
            if isinstance(val, (int, float)) and isinstance(eff, (int, float)):
                groups.setdefault(val, []).append(eff)
        if groups:
            best = min(groups.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))[0]
            best_val[col] = best
    return best_val, model, feat_names

# ----------------------------------------------------------------------
# Predict effort for a single row given optional overrides
# ----------------------------------------------------------------------
def predict_effort(row, overrides, model, feat_names):
    vec = []
    for name in feat_names:
        if name in overrides:
            vec.append(overrides[name])
        else:
            vec.append(row.get(name, 0))
    return model.predict(np.array([vec]))[0]

# ----------------------------------------------------------------------
# Compute best changes to reduce effort (top 3)
# ----------------------------------------------------------------------
def nasa_15(rows):
    # Select proposal as first row
    proposal = rows[0]
    proj_id = int(proposal.get('idX', 0))

    numeric_cols, categorical_cols = get_adjustable_columns(rows)
    best_val, effort_model, feat_names = best_values_for_effort(rows, numeric_cols, categorical_cols)

    # Base prediction
    base_pred = predict_effort(proposal, {}, effort_model, feat_names)

    changes = []
    for col in numeric_cols + categorical_cols:
        override = best_val.get(col, None)
        if override is None:
            continue
        new_pred = predict_effort(proposal, {col: override}, effort_model, feat_names)
        reduction = base_pred - new_pred
        if reduction <= 0:
            continue
        changes.append((reduction, col, override, new_pred))

    changes.sort(reverse=True, key=lambda x: x[0])

    print("== NASA-15 ==")
    if not changes:
        print("No adjustable characteristics found that reduce effort.")
        print()
        return
    print(f"Proposal ID={proj_id}, Base predicted effort={base_pred:.2f}")
    print("Top 3 suggested changes (effort reduction, new predicted effort):")
    for i, (red, col, val, new_pred) in enumerate(changes[:3], start=1):
        val_disp = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
        print(f"{i}. {col} -> {val_disp} | Reduction={red:.2f} | New effort={new_pred:.2f}")
    print()

# ----------------------------------------------------------------------
# Compute best values for defects (lowest mean defect) – used in NASA-16
# ----------------------------------------------------------------------
def best_values_for_defects(rows, numeric_cols, categorical_cols):
    # Build regression model for defects
    X, y, feat_names = build_matrix(rows, 'Defects-', exclude_cols=['idX', 'YearX', 'MonthsX'])
    model = LinearRegression().fit(X, y)

    best_val = {}
    # Numeric: choose min observed value
    for col in numeric_cols:
        vals = [r[col] for r in rows if isinstance(r[col], (int, float))]
        if vals:
            best_val[col] = min(vals)
    # Categorical: choose value that yields lowest mean defect in data
    for col in categorical_cols:
        groups = {}
        for r in rows:
            val = r[col]
            defc = r.get('Defects-', None)
            if isinstance(val, (int, float)) and isinstance(defc, (int, float)):
                groups.setdefault(val, []).append(defc)
        if groups:
            best = min(groups.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))[0]
            best_val[col] = best
    return best_val, model, feat_names

# ----------------------------------------------------------------------
# Predict defect for a single row given overrides
# ----------------------------------------------------------------------
def predict_defects(row, overrides, model, feat_names):
    vec = []
    for name in feat_names:
        if name in overrides:
            vec.append(overrides[name])
        else:
            vec.append(row.get(name, 0))
    return model.predict(np.array([vec]))[0]

# ----------------------------------------------------------------------
# Greedy selection of changes to reduce defects (minimal number)
# ----------------------------------------------------------------------
def nasa_16(rows):
    proposal = rows[0]
    proj_id = int(proposal.get('idX', 0))

    numeric_cols, categorical_cols = get_adjustable_columns(rows)
    best_val, defect_model, feat_names = best_values_for_defects(rows, numeric_cols, categorical_cols)

    base_pred = predict_defects(proposal, {}, defect_model, feat_names)

    # Compute mean defect in data to use as target for improvement
    all_defs = [r.get('Defects-', None) for r in rows if isinstance(r.get('Defects-', None), (int, float))]
    target_defect = sum(all_defs) / len(all_defs) if all_defs else 0.0

    changes = []
    current_overrides = {}
    current_pred = base_pred

    # Greedy loop up to 3 changes
    for step in range(3):
        best_step = None
        best_red = 0
        for col in numeric_cols + categorical_cols:
            if col in current_overrides:
                continue
            override_val = best_val.get(col, None)
            if override_val is None:
                continue
            new_pred = predict_defects(proposal, {**current_overrides, col: override_val}, defect_model, feat_names)
            reduction = current_pred - new_pred
            if reduction > best_red:
                best_red = reduction
                best_step = (col, override_val, new_pred)
        if best_step is None or best_red <= 0:
            break
        col, val, new_pred = best_step
        current_overrides[col] = val
        current_pred = new_pred
        changes.append((col, val, new_pred, best_red))

    print("== NASA-16 ==")
    if not changes:
        print("No single or combination of adjustments improves defect prediction.")
        print()
        return
    print(f"Proposal ID={proj_id}, Base predicted defects={base_pred:.2f}, Target mean defects={target_defect:.2f}")
    print("Suggested adjustments (minimal number, defect reduction, new predicted defects):")
    for i, (col, val, new_pred, red) in enumerate(changes, start=1):
        val_disp = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
        print(f"{i}. {col} -> {val_disp} | Reduction={red:.2f} | New defects={new_pred:.2f}")
    print()

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        header, rows = read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if not rows:
        print("No data found in CSV.", file=sys.stderr)
        sys.exit(1)

    try:
        nasa_15(rows)
    except Exception as e:
        print(f"NASA-15 error: {e}", file=sys.stderr)

    try:
        nasa_16(rows)
    except Exception as e:
        print(f"NASA-16 error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
