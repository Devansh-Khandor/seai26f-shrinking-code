#!/usr/bin/env python3
"""
skill_08.py: NASA-15 and NASA-16 analyses.
Usage:  python skill_08.py <csv path>
"""

import csv
import json
import math
import sys
import itertools

from skill_01 import parse_float

# ----------------------------------------------------------------------
# Utility: solve linear system via Gaussian elimination
# ----------------------------------------------------------------------
def solve_linear_system(A, b):
    """Solve A x = b for x. A is list of lists, square."""
    n = len(A)
    # Augment
    M = [row[:] + [b_i] for row, b_i in zip(A, b)]

    # Forward elimination
    for i in range(n):
        # Pivot
        pivot_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[pivot_row][i]) < 1e-12:
            # Singular, return zeros
            return [0.0] * n
        if pivot_row != i:
            M[i], M[pivot_row] = M[pivot_row], M[i]
        pivot = M[i][i]
        # Normalize pivot row
        for j in range(i, n + 1):
            M[i][j] /= pivot
        # Eliminate below
        for r in range(i + 1, n):
            factor = M[r][i]
            for j in range(i, n + 1):
                M[r][j] -= factor * M[i][j]

    # Back substitution
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
    return x

# ----------------------------------------------------------------------
# Build linear regression model
# ----------------------------------------------------------------------
def build_ols_model(rows, target_col, predictor_cols):
    """
    rows: list of dicts (including row_num)
    target_col: string
    predictor_cols: list of strings (numeric, non‑goal)
    Returns (beta0, betas) where betas correspond to predictor_cols
    """
    X = []
    y = []
    for r in rows:
        t = parse_float(r.get(target_col))
        if t is None:
            continue
        xs = []
        skip = False
        for p in predictor_cols:
            v = parse_float(r.get(p))
            if v is None:
                skip = True
                break
            xs.append(v)
        if skip:
            continue
        X.append([1.0] + xs)  # intercept
        y.append(t)

    if not X:
        # No data
        return 0.0, [0.0] * len(predictor_cols)

    # Compute (X^T X) and X^T y
    m = len(predictor_cols) + 1
    XT_X = [[0.0] * m for _ in range(m)]
    XT_y = [0.0] * m
    for xi, yi in zip(X, y):
        for i in range(m):
            XT_y[i] += xi[i] * yi
            for j in range(m):
                XT_X[i][j] += xi[i] * xi[j]

    # Solve for betas
    betas = solve_linear_system(XT_X, XT_y)
    return betas[0], betas[1:]  # intercept, slopes

# ----------------------------------------------------------------------
# Predict using model
# ----------------------------------------------------------------------
def predict_row(row, intercept, betas, predictor_cols):
    xs = []
    for p in predictor_cols:
        v = parse_float(row.get(p))
        if v is None:
            # Missing predictor: cannot predict
            return None
        xs.append(v)
    pred = intercept + sum(b * x for b, x in zip(betas, xs))
    return pred

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main(csv_path):
    # Read data
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        for idx, r in enumerate(reader, start=1):
            r = dict(r)  # copy
            r["row_num"] = idx
            rows.append(r)

    # Identify adjustable numeric columns (exclude goal and X)
    adjustable_cols = []
    for col in header:
        if col.endswith('+') or col.endswith('-') or col.endswith('X'):
            continue
        # Check if numeric in at least one row
        for r in rows:
            v = parse_float(r.get(col))
            if v is not None:
                adjustable_cols.append(col)
                break
    # Build regression models
    effort_intercept, effort_betas = build_ols_model(rows, "Effort-", adjustable_cols)
    defect_intercept, defect_betas = build_ols_model(rows, "Defects-", adjustable_cols)

    # Predict for all rows (including held‑out)
    for r in rows:
        r["_effort_pred"] = predict_row(r, effort_intercept, effort_betas, adjustable_cols)
        r["_defect_pred"] = predict_row(r, defect_intercept, defect_betas, adjustable_cols)

    # Held‑out rows
    held_out = [r for r in rows if r["row_num"] % 5 == 0]

    # ---------- NASA‑15 ----------
    # Find held‑out with highest predicted Effort-
    best_row = None
    best_effort = -math.inf
    for r in held_out:
        val = r["_effort_pred"]
        if val is None:
            continue
        if val > best_effort:
            best_effort = val
            best_row = r
    if best_row is None:
        print("== NASA-15 ==")
        print("No held‑out rows with computable Effort- prediction.")
        print(f"ANSWER NASA-15 {{}}")
    else:
        row_num = best_row["row_num"]
        orig_effort = best_effort

        # Find best single change
        best_change = None
        best_new_effort = orig_effort
        for col in adjustable_cols:
            current = parse_float(best_row.get(col))
            # gather distinct values in dataset for this col
            distinct = set()
            for r in rows:
                v = parse_float(r.get(col))
                if v is not None:
                    distinct.add(v)
            for new_val in distinct:
                if new_val == current:
                    continue
                # create modified row copy
                mod = best_row.copy()
                mod[col] = str(new_val)
                new_effort = predict_row(mod, effort_intercept, effort_betas, adjustable_cols)
                if new_effort is None:
                    continue
                if new_effort < best_new_effort:
                    best_new_effort = new_effort
                    best_change = {col: [current, new_val]}

        # If no beneficial change found, keep original
        if best_change is None:
            # pick first column with any alternative value
            for col in adjustable_cols:
                current = parse_float(best_row.get(col))
                distinct = set()
                for r in rows:
                    v = parse_float(r.get(col))
                    if v is not None:
                        distinct.add(v)
                for new_val in distinct:
                    if new_val == current:
                        continue
                    best_change = {col: [current, new_val]}
                    best_new_effort = predict_row(best_row, effort_intercept, effort_betas, adjustable_cols)
                    break
                if best_change:
                    break

        # Output
        print("== NASA-15 ==")
        print(f"Project row {row_num} (held‑out)")
        print(f"Predicted Effort- before change: {round(orig_effort,4)}")
        print(f"Predicted Effort- after change: {round(best_new_effort,4)}")
        print(f"Proposed changes: {best_change}")
        nasa15_json = {
            "project": row_num,
            "changes": best_change,
            "predicted_effort": {"before": round(orig_effort,4), "after": round(best_new_effort,4)}
        }
        print(f"ANSWER NASA-15 {json.dumps(nasa15_json)}")

    # ---------- NASA‑16 ----------
    # Find held‑out with highest predicted Defects-
    best_row_def = None
    best_defects = -math.inf
    for r in held_out:
        val = r["_defect_pred"]
        if val is None:
            continue
        if val > best_defects:
            best_defects = val
            best_row_def = r
    if best_row_def is None:
        print("\n== NASA-16 ==")
        print("No held‑out rows with computable Defects- prediction.")
        print(f"ANSWER NASA-16 {{}}")
    else:
        row_num_def = best_row_def["row_num"]
        orig_defects = best_defects
        orig_effort_def = best_row_def["_effort_pred"] if best_row_def["_effort_pred"] is not None else 0.0

        # Search for minimal number of changes up to 3
        found = False
        best_combo = None
        best_defects_after = orig_defects
        best_effort_after = orig_effort_def

        # Helper to compute defect prediction for a given row
        def defect_pred_for(mod):
            return predict_row(mod, defect_intercept, defect_betas, adjustable_cols)

        # Helper to compute effort prediction for a given row
        def effort_pred_for(mod):
            return predict_row(mod, effort_intercept, effort_betas, adjustable_cols)

        # Prepare distinct values per column
        distinct_per_col = {}
        for col in adjustable_cols:
            vals = set()
            for r in rows:
                v = parse_float(r.get(col))
                if v is not None:
                    vals.add(v)
            distinct_per_col[col] = vals

        for num_cols in range(1, 4):
            if found:
                break
            for cols_combo in itertools.combinations(adjustable_cols, num_cols):
                # Build list of possible new values per column (exclude current)
                value_lists = []
                for col in cols_combo:
                    current = parse_float(best_row_def.get(col))
                    choices = [v for v in distinct_per_col[col] if v != current]
                    if not choices:
                        # cannot change this column meaningfully
                        choices = [current]
                    value_lists.append(choices)
                # Enumerate all product combinations
                for prod in itertools.product(*value_lists):
                    mod = best_row_def.copy()
                    for col, new_val in zip(cols_combo, prod):
                        mod[col] = str(new_val)
                    new_def = defect_pred_for(mod)
                    if new_def is None:
                        continue
                    if new_def < best_defects_after:
                        # Found improvement
                        best_combo = {col: [parse_float(best_row_def.get(col)), new_val]
                                      for col, new_val in zip(cols_combo, prod)}
                        best_defects_after = new_def
                        best_effort_after = effort_pred_for(mod) if effort_pred_for(mod) is not None else orig_effort_def
                        found = True
                        break
                if found:
                    break

        # If no improvement, keep original
        if not found:
            best_combo = {}
            best_defects_after = orig_defects
            best_effort_after = orig_effort_def

        # Output
        print("\n== NASA-16 ==")
        print(f"Project row {row_num_def} (held‑out)")
        print(f"Predicted Defects- before change: {round(orig_defects,4)}")
        print(f"Predicted Defects- after change: {round(best_defects_after,4)}")
        print(f"Predicted Effort- before change: {round(orig_effort_def,4)}")
        print(f"Predicted Effort- after change: {round(best_effort_after,4)}")
        print(f"Proposed changes (max 3): {best_combo}")
        nasa16_json = {
            "project": row_num_def,
            "changes": best_combo,
            "predicted_defects": {"before": round(orig_defects,4), "after": round(best_defects_after,4)},
            "predicted_effort": {"before": round(orig_effort_def,4), "after": round(best_effort_after,4)}
        }
        print(f"ANSWER NASA-16 {json.dumps(nasa16_json)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv path>")
        sys.exit(1)
    main(sys.argv[1])
