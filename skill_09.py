#!/usr/bin/env python3
"""
skill_09.py: NASA-17 and NASA-18 analyses.
Usage:  python skill_09.py <csv path>
"""

import csv
import json
import math
import sys

# ----------------------------------------------------------------------
# Import helper functions from earlier skill
# ----------------------------------------------------------------------
from skill_01 import parse_float

# ----------------------------------------------------------------------
# Linear algebra helpers (copied from skill_08)
# ----------------------------------------------------------------------
def solve_linear_system(A, b):
    """Solve A x = b for x. A is list of lists, square."""
    n = len(A)
    M = [row[:] + [b_i] for row, b_i in zip(A, b)]

    # Forward elimination
    for i in range(n):
        # pivot
        pivot_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[pivot_row][i]) < 1e-12:
            return [0.0] * n
        if pivot_row != i:
            M[i], M[pivot_row] = M[pivot_row], M[i]
        pivot = M[i][i]
        # Normalize
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

def build_ols_model(rows, target_col, predictor_cols):
    """Return (intercept, betas) for OLS regression."""
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
        X.append([1.0] + xs)
        y.append(t)

    if not X:
        return 0.0, [0.0] * len(predictor_cols)

    m = len(predictor_cols) + 1
    XT_X = [[0.0] * m for _ in range(m)]
    XT_y = [0.0] * m
    for xi, yi in zip(X, y):
        for i in range(m):
            XT_y[i] += xi[i] * yi
            for j in range(m):
                XT_X[i][j] += xi[i] * xi[j]

    betas = solve_linear_system(XT_X, XT_y)
    return betas[0], betas[1:]

def predict_row(row, intercept, betas, predictor_cols):
    xs = []
    for p in predictor_cols:
        v = parse_float(row.get(p))
        if v is None:
            return None
        xs.append(v)
    return intercept + sum(b * x for b, x in zip(betas, xs))

# ----------------------------------------------------------------------
# NASA-17: Flag unusual input combinations
# ----------------------------------------------------------------------
def nasa17(rows, input_cols):
    # Compute mean and std for each numeric input column
    means = {}
    stds = {}
    for col in input_cols:
        vals = [parse_float(r.get(col)) for r in rows]
        nums = [v for v in vals if v is not None]
        if not nums:
            means[col] = stds[col] = 0.0
            continue
        mean_val = sum(nums) / len(nums)
        var = sum((v - mean_val) ** 2 for v in nums) / len(nums)
        std_val = math.sqrt(var) if var > 0 else 1.0
        means[col] = mean_val
        stds[col] = std_val

    # Compute unusualness score for each row (Mahalanobis-like)
    scores = {}
    for r in rows:
        zsum = 0.0
        count = 0
        for col in input_cols:
            v = parse_float(r.get(col))
            if v is None:
                continue
            z = (v - means[col]) / stds[col]
            zsum += z * z
            count += 1
        score = math.sqrt(zsum) if count > 0 else 0.0
        scores[r["row_num"]] = round(score, 4)

    # Flag top 10 rows with highest score
    flagged = sorted(scores.items(), key=lambda x: -x[1])[:10]
    flagged_rows = [row for row, _ in flagged]
    flagged_scores = {str(row): scores[row] for row, _ in flagged}

    # Print section
    print("== NASA-17 ==")
    print(f"Computed unusualness scores for {len(rows)} rows.")
    print(f"Top {len(flagged_rows)} flagged rows: {flagged_rows}")
    print("Scores of flagged rows:")
    for row in flagged_rows:
        print(f"  Row {row}: {flagged_scores[str(row)]}")

    nasa17_json = {"flagged": flagged_rows, "scores": flagged_scores}
    print(f"ANSWER NASA-17 {json.dumps(nasa17_json)}")

# ----------------------------------------------------------------------
# NASA-18: Flag effort/defect measurement issues
# ----------------------------------------------------------------------
def nasa18(rows, input_cols):
    # Build regression models for Effort- and Defects-
    effort_intercept, effort_betas = build_ols_model(rows, "Effort-", input_cols)
    defect_intercept, defect_betas = build_ols_model(rows, "Defects-", input_cols)

    # Predict and compute residuals
    effort_residuals = []
    defect_residuals = []

    for r in rows:
        eff_actual = parse_float(r.get("Effort-"))
        eff_pred = predict_row(r, effort_intercept, effort_betas, input_cols)
        if eff_actual is not None and eff_pred is not None:
            eff_residuals.append(eff_actual - eff_pred)

        def_actual = parse_float(r.get("Defects-"))
        def_pred = predict_row(r, defect_intercept, defect_betas, input_cols)
        if def_actual is not None and def_pred is not None:
            defect_residuals.append(def_actual - def_pred)

    # Compute std of residuals
    def std_of(lst):
        if not lst:
            return 0.0
        mean_val = sum(lst) / len(lst)
        var = sum((v - mean_val) ** 2 for v in lst) / len(lst)
        return math.sqrt(var)

    eff_std = std_of(eff_residuals)
    def_std = std_of(defect_residuals)

    flagged_entries = []

    # Threshold: 2 standard deviations
    for r in rows:
        row_num = r["row_num"]
        eff_actual = parse_float(r.get("Effort-"))
        eff_pred = predict_row(r, effort_intercept, effort_betas, input_cols)
        if eff_actual is not None and eff_pred is not None:
            diff = abs(eff_actual - eff_pred)
            if diff > 2 * eff_std:
                flagged_entries.append({
                    "row": row_num,
                    "column": "Effort-",
                    "actual": round(eff_actual, 4),
                    "expected": round(eff_pred, 4)
                })

        def_actual = parse_float(r.get("Defects-"))
        def_pred = predict_row(r, defect_intercept, defect_betas, input_cols)
        if def_actual is not None and def_pred is not None:
            diff = abs(def_actual - def_pred)
            if diff > 2 * def_std:
                flagged_entries.append({
                    "row": row_num,
                    "column": "Defects-",
                    "actual": round(def_actual, 4),
                    "expected": round(def_pred, 4)
                })

    # Keep up to 10
    flagged_entries = flagged_entries[:10]

    # Print section
    print("\n== NASA-18 ==")
    print(f"Evaluated {len(rows)} rows for prediction consistency.")
    print(f"Flagged {len(flagged_entries)} rows with large residuals.")
    for e in flagged_entries:
        print(f"  Row {e['row']} {e['column']} actual {e['actual']} expected {e['expected']}")

    nasa18_json = {"flagged": flagged_entries}
    print(f"ANSWER NASA-18 {json.dumps(nasa18_json)}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main(csv_path):
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        for idx, r in enumerate(reader, start=1):
            r = dict(r)
            r["row_num"] = idx
            rows.append(r)

    # Identify numeric input columns (exclude goals and X)
    input_cols = []
    for col in header:
        if col.endswith('+') or col.endswith('-') or col.endswith('X'):
            continue
        # Check if column has at least one numeric value
        for r in rows:
            if parse_float(r.get(col)) is not None:
                input_cols.append(col)
                break

    nasa17(rows, input_cols)
    nasa18(rows, input_cols)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv path>")
        sys.exit(1)
    main(sys.argv[1])
