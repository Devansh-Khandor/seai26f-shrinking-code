#!/usr/bin/env python3
"""
skill_08.py <csv_path>

Implements NASA-15 and NASA-16.

NASA-15: Identify a proposal with the highest predicted effort and
suggest changes to adjustable columns that may reduce its predicted effort.
NASA-16: Identify a proposal with the highest predicted defect count and
suggest up to three changes that may improve its predicted quality,
listing the effort trade‑off.

The program prints, for each request, a header line
'== <ID> ==', at least three lines of textual output, and a machine‑checked
answer line:
    ANSWER <ID> <json>
The JSON objects are defined in the problem statement.
"""

import sys
import csv
import json
import math
from statistics import mean

# --------------------------------------------------------------------
# CSV utilities
# --------------------------------------------------------------------
def parse_csv(path):
    """Read CSV into list of dict rows."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def to_float(val):
    """Convert cell to float, return None for missing."""
    if val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

# --------------------------------------------------------------------
# Numeric column handling
# --------------------------------------------------------------------
def is_numeric_column(col, rows):
    """True if at least one row can be parsed as float for this column."""
    for r in rows:
        if to_float(r[col]) is not None:
            return True
    return False

def numeric_features(rows):
    """Return list of numeric columns that are adjustable (not +, -, X)."""
    header = rows[0].keys()
    feats = [c for c in header
             if not c.endswith('+') and not c.endswith('-') and not c.endswith('X')
             and is_numeric_column(c, rows)]
    # Always include Kloc+ as a feature (goal)
    if 'Kloc+' not in feats:
        feats.append('Kloc+')
    return feats

# --------------------------------------------------------------------
# Linear algebra helpers
# --------------------------------------------------------------------
def transpose(A):
    return [list(col) for col in zip(*A)]

def matrix_multiply(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]

def solve_linear_system(A, b):
    """Solve Ax = b for x. A is square matrix, b is vector."""
    n = len(A)
    # Augment matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    # Gaussian elimination with partial pivot
    for i in range(n):
        # Pivot
        piv = i
        for j in range(i+1, n):
            if abs(M[j][i]) > abs(M[piv][i]):
                piv = j
        if abs(M[piv][i]) < 1e-12:
            # Singular matrix; return zeros
            return [0.0]*n
        M[i], M[piv] = M[piv], M[i]
        # Normalize row
        div = M[i][i]
        for k in range(i, n+1):
            M[i][k] /= div
        # Eliminate
        for j in range(n):
            if j == i:
                continue
            factor = M[j][i]
            for k in range(i, n+1):
                M[j][k] -= factor * M[i][k]
    # Back substitution
    x = [M[i][-1] for i in range(n)]
    return x

# --------------------------------------------------------------------
# Regression helpers
# --------------------------------------------------------------------
def build_design_matrix(rows, feats):
    """Return X matrix (list of rows) and Y vector for Effort regression."""
    X = []
    Y = []
    for r in rows:
        # Skip rows with missing Effort- or any feature
        eff = to_float(r["Effort-"])
        if eff is None:
            continue
        row = []
        skip = False
        for f in feats:
            val = to_float(r[f])
            if val is None:
                skip = True
                break
            row.append(val)
        if skip:
            continue
        X.append(row)
        Y.append(eff)
    return X, Y

def least_squares_solution(X, Y):
    """Return beta coefficients via normal equations."""
    if not X:
        return []
    Xt = transpose(X)
    XtX = matrix_multiply(Xt, X)
    XtY = [sum(Xt[i][j] * Y[j] for j in range(len(Y))) for i in range(len(Xt))]
    beta = solve_linear_system(XtX, XtY)
    return beta

def predict_with_beta(beta, feats, row):
    """Predict target using beta and feats for a given row."""
    if len(beta) == 0:
        return 0.0
    values = []
    skip = False
    for f in feats:
        v = to_float(row[f])
        if v is None:
            skip = True
            break
        values.append(v)
    if skip:
        return 0.0
    return sum(beta[i] * values[i] for i in range(len(beta)))

# --------------------------------------------------------------------
# Defect model (Kloc+ and Effort- -> Defects-)
# --------------------------------------------------------------------
def build_defect_design(rows):
    X = []
    Y = []
    for r in rows:
        defc = to_float(r["Defects-"])
        if defc is None:
            continue
        k = to_float(r["Kloc+"])
        e = to_float(r["Effort-"])
        if k is None or e is None:
            continue
        X.append([1.0, k, e])
        Y.append(defc)
    return X, Y

def defect_beta_solution(X, Y):
    if not X:
        return [0.0, 0.0, 0.0]
    Xt = transpose(X)
    XtX = matrix_multiply(Xt, X)
    XtY = [sum(Xt[i][j] * Y[j] for j in range(len(Y))) for i in range(len(Xt))]
    beta = solve_linear_system(XtX, XtY)
    return beta

def predict_defects(beta, row):
    if len(beta) < 3:
        return 0.0
    k = to_float(row["Kloc+"])
    e = to_float(row["Effort-"])
    if k is None or e is None:
        return 0.0
    return beta[0] + beta[1]*k + beta[2]*e

# --------------------------------------------------------------------
# NASA-15 implementation
# --------------------------------------------------------------------
def nasa_15(rows, effort_beta, effort_feats):
    # Identify row with highest predicted effort
    max_eff = -float('inf')
    proj_idx = None
    for idx, r in enumerate(rows, start=1):
        pred = predict_with_beta(effort_beta, effort_feats, r)
        if pred > max_eff:
            max_eff = pred
            proj_idx = idx
    proj_row = rows[proj_idx-1]
    pred_before = round(max_eff, 4)

    # Find row with minimal actual Effort- to derive plausible changes
    min_eff = float('inf')
    min_row = None
    for r in rows:
        eff = to_float(r["Effort-"])
        if eff is not None and eff < min_eff:
            min_eff = eff
            min_row = r
    # Determine plausible changes (numeric adjustable columns)
    changes = {}
    for f in effort_feats:
        if f == "Kloc+":
            continue
        curr = to_float(proj_row[f])
        new = to_float(min_row[f])
        if curr is None or new is None or curr == new:
            continue
        # Evaluate effect
        temp_row = proj_row.copy()
        temp_row[f] = str(new)
        pred_after_val = predict_with_beta(effort_beta, effort_feats, temp_row)
        if pred_after_val < pred_before:
            changes[f] = [curr, new]
    pred_after = round(pred_before, 4)  # default same
    if changes:
        # compute new predicted effort using all changes
        temp_row = proj_row.copy()
        for f, (c, n) in changes.items():
            temp_row[f] = str(n)
        pred_after_val = predict_with_beta(effort_beta, effort_feats, temp_row)
        pred_after = round(pred_after_val, 4)

    # Textual output
    print("== NASA-15 ==")
    print(f"Project row: {proj_idx}")
    print(f"Predicted effort before: {pred_before}")
    if changes:
        print(f"Changes suggested to reduce effort:")
        for k, v in changes.items():
            print(f"  {k}: {v[0]} -> {v[1]}")
    else:
        print("No changes found that reduce effort.")
    print(f"Predicted effort after: {pred_after}")

    # Machine-checked answer
    ans = {
        "project": proj_idx,
        "changes": changes,
        "predicted_effort": {"before": pred_before, "after": pred_after}
    }
    print("ANSWER NASA-15", json.dumps(ans, separators=(',', ':')))

# --------------------------------------------------------------------
# NASA-16 implementation
# --------------------------------------------------------------------
def nasa_16(rows, effort_beta, effort_feats, defect_beta):
    # Find row with highest predicted defect count
    max_def = -float('inf')
    proj_idx = None
    for idx, r in enumerate(rows, start=1):
        pred_def = predict_defects(defect_beta, r)
        if pred_def > max_def:
            max_def = pred_def
            proj_idx = idx
    proj_row = rows[proj_idx-1]
    pred_def_before = round(max_def, 4)
    pred_eff_before = round(predict_with_beta(effort_beta, effort_feats, proj_row), 4)

    # Find row with minimal actual Defects- to derive plausible changes
    min_def = float('inf')
    min_row = None
    for r in rows:
        defc = to_float(r["Defects-"])
        if defc is not None and defc < min_def:
            min_def = defc
            min_row = r

    # Determine changes (numeric adjustable columns) that reduce defect prediction
    changes = {}
    for f in effort_feats:
        if f == "Kloc+":
            continue
        curr = to_float(proj_row[f])
        new = to_float(min_row[f])
        if curr is None or new is None or curr == new:
            continue
        # Apply change and recalc effort and defects
        temp_row = proj_row.copy()
        temp_row[f] = str(new)
        pred_eff_new = predict_with_beta(effort_beta, effort_feats, temp_row)
        pred_def_new = predict_defects(defect_beta, temp_row)
        if pred_def_new < pred_def_before:
            changes[f] = [curr, new]
    # Limit to at most 3 changes
    if len(changes) > 3:
        # pick top 3 that give largest defect reduction
        deltas = []
        for f, (c, n) in changes.items():
            temp_row = proj_row.copy()
            temp_row[f] = str(n)
            pred_def_new = predict_defects(defect_beta, temp_row)
            delta = pred_def_before - pred_def_new
            deltas.append((delta, f, (c, n)))
        deltas.sort(reverse=True)
        changes = {f: v for _, f, v in deltas[:3]}

    # Compute final predictions after changes
    temp_row = proj_row.copy()
    for f, (c, n) in changes.items():
        temp_row[f] = str(n)
    pred_eff_after = round(predict_with_beta(effort_beta, effort_feats, temp_row), 4)
    pred_def_after = round(predict_defects(defect_beta, temp_row), 4)

    # Textual output
    print("== NASA-16 ==")
    print(f"Project row: {proj_idx}")
    print(f"Predicted defects before: {pred_def_before}")
    print(f"Predicted effort before: {pred_eff_before}")
    if changes:
        print(f"Changes suggested to improve quality (at most 3):")
        for k, v in changes.items():
            print(f"  {k}: {v[0]} -> {v[1]}")
    else:
        print("No changes found that reduce defects.")
    print(f"Predicted defects after: {pred_def_after}")
    print(f"Predicted effort after: {pred_eff_after}")

    # Machine-checked answer
    ans = {
        "project": proj_idx,
        "changes": changes,
        "predicted_defects": {"before": pred_def_before, "after": pred_def_after},
        "predicted_effort": {"before": pred_eff_before, "after": pred_eff_after}
    }
    print("ANSWER NASA-16", json.dumps(ans, separators=(',', ':')))

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_08.py <csv_path>\n")
        sys.exit(1)
    path = sys.argv[1]
    rows = parse_csv(path)

    # Build effort regression
    effort_feats = numeric_features(rows)   # includes Kloc+
    X_eff, Y_eff = build_design_matrix(rows, effort_feats)
    effort_beta = least_squares_solution(X_eff, Y_eff)

    # Build defect regression
    X_def, Y_def = build_defect_design(rows)
    defect_beta = defect_beta_solution(X_def, Y_def)

    nasa_15(rows, effort_beta, effort_feats)
    print()  # blank line between sections
    nasa_16(rows, effort_beta, effort_feats, defect_beta)

if __name__ == "__main__":
    main()
