#!/usr/bin/env python3
"""
skill_09.py

Implements APACHE-17 and APACHE-18 analyses on the provided CSV file.

Usage:
    python skill_09.py <csv_path>
"""

import sys
import csv
import json
import math
from itertools import combinations

def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, reader.fieldnames

def to_float(val):
    if val is None or val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def is_goal(col):
    return col.endswith('+') or col.endswith('-')

def is_setting(col, goal_col):
    return col != goal_col and not col.endswith('X') and not is_goal(col)

# --- Linear regression utilities -------------------------------------------

def solve_linear(A, b):
    """
    Solve Ax = b using Gaussian elimination with partial pivoting.
    Return x or None if singular.
    """
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[max_row][i]) < 1e-12:
            return None
        M[i], M[max_row] = M[max_row], M[i]
        pivot = M[i][i]
        for j in range(i, n+1):
            M[i][j] /= pivot
        for r in range(i+1, n):
            factor = M[r][i]
            for j in range(i, n+1):
                M[r][j] -= factor * M[i][j]
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        x[i] = M[i][n]
        for j in range(i+1, n):
            x[i] -= M[i][j] * x[j]
    return x

def fit_linear_regression(rows, goal_col, settings):
    """
    Build X and y from rows that have complete numeric goal and all settings.
    Return intercept and coefficient list [beta1, beta2, ...].
    """
    X = []
    y = []
    for row in rows:
        g = to_float(row[goal_col])
        if g is None:
            continue
        vals = [1.0]  # intercept term
        ok = True
        for s in settings:
            v = to_float(row[s])
            if v is None:
                ok = False
                break
            vals.append(v)
        if not ok:
            continue
        X.append(vals)
        y.append(g)
    if not X:
        return None, None
    p = len(X[0])  # including intercept
    XtX = [[0.0]*p for _ in range(p)]
    Xty = [0.0]*p
    for i in range(len(X)):
        xi = X[i]
        yi = y[i]
        for a in range(p):
            Xty[a] += xi[a] * yi
            for b in range(a, p):
                XtX[a][b] += xi[a] * xi[b]
    for a in range(p):
        for b in range(a+1, p):
            XtX[b][a] = XtX[a][b]
    coeffs = solve_linear(XtX, Xty)
    if coeffs is None:
        return None, None
    intercept = coeffs[0]
    betas = coeffs[1:]
    return intercept, betas

def predict(row, goal_col, settings, intercept, betas):
    if intercept is None or betas is None:
        return None
    val = intercept
    for i, s in enumerate(settings):
        v = to_float(row[s])
        if v is None:
            return None
        val += betas[i] * v
    return val

# --- APACHE-17 --------------------------------------------------------------

def apache_17(rows, goal_col, settings):
    intercept, betas = fit_linear_regression(rows, goal_col, settings)
    if intercept is None:
        # fallback: use mean
        all_vals = [to_float(r[goal_col]) for r in rows if to_float(r[goal_col]) is not None]
        mean_val = sum(all_vals)/len(all_vals) if all_vals else 0.0
        intercept = mean_val
        betas = [0.0]*len(settings)

    # Compute predictions and residuals
    predictions = {}
    residuals = []
    for idx, row in enumerate(rows, 1):
        actual = to_float(row[goal_col])
        pred = predict(row, goal_col, settings, intercept, betas)
        if actual is None or pred is None:
            continue
        predictions[idx] = pred
        residuals.append((idx, actual, pred, actual - pred))

    if not residuals:
        mean_res, std_res = 0.0, 0.0
    else:
        res_vals = [r[3] for r in residuals]
        mean_res = sum(res_vals)/len(res_vals)
        if len(res_vals) > 1:
            var = sum((v-mean_res)**2 for v in res_vals)/(len(res_vals)-1)
            std_res = math.sqrt(var)
        else:
            std_res = 0.0

    threshold = 2 * std_res

    flagged = []
    for idx, actual, pred, res in residuals:
        if abs(res) > threshold:
            flagged.append({
                "row": idx,
                "column": goal_col,
                "actual": round(actual, 4),
                "expected": round(pred, 4)
            })
    # keep up to 20, sorted by abs residual descending
    flagged.sort(key=lambda x: abs(x["expected"] - x["actual"]), reverse=True)
    flagged = flagged[:20]

    # Prepare JSON
    answer = {"flagged": flagged}
    return answer, (len(rows), len(predictions), mean_res, std_res, threshold, len(flagged))

# --- APACHE-18 --------------------------------------------------------------

def apache_18(rows, settings):
    # Compute Hamming distance to nearest neighbor for rows with complete settings
    valid_rows = []
    for idx, row in enumerate(rows, 1):
        if any(row[s] in ('?', '') for s in settings):
            continue
        valid_rows.append((idx, row))

    distances = {}
    for i, (idx_i, row_i) in enumerate(valid_rows):
        best = None
        for j, (idx_j, row_j) in enumerate(valid_rows):
            if idx_i == idx_j:
                continue
            diff = 0
            for s in settings:
                if row_i[s] != row_j[s]:
                    diff += 1
            if best is None or diff < best:
                best = diff
        if best is not None:
            distances[idx_i] = best

    if not distances:
        return {"flagged": [], "scores": {}}, (0, 0)

    vals = list(distances.values())
    mean_d = sum(vals)/len(vals)
    if len(vals) > 1:
        var = sum((v-mean_d)**2 for v in vals)/(len(vals)-1)
        std_d = math.sqrt(var)
    else:
        std_d = 0.0

    threshold = mean_d + 2 * std_d

    flagged_rows = [idx for idx, d in distances.items() if d > threshold]
    flagged_rows.sort(key=lambda x: distances[x], reverse=True)
    flagged_rows = flagged_rows[:20]

    scores = {str(idx): distances[idx] for idx in flagged_rows}

    answer = {"flagged": flagged_rows, "scores": scores}
    return answer, (len(valid_rows), mean_d, std_d, threshold, len(flagged_rows))

# ------------------------------------------------------------------------------

def main(csv_path):
    rows, columns = read_csv(csv_path)
    if not rows:
        print("No data rows found.")
        sys.exit(1)

    goal_col = None
    for c in columns:
        if c.endswith('+') or c.endswith('-'):
            goal_col = c
            break
    if goal_col is None:
        print("No goal column found.")
        sys.exit(1)

    settings = [c for c in columns if not c.endswith('X') and not is_goal(c)]

    # --- APACHE-17 ---
    answer17, stats17 = apache_17(rows, goal_col, settings)
    n_rows, n_pred, mean_res, std_res, thresh_res, n_flagged17 = stats17
    print("== APACHE-17 ==")
    print(f"Total rows: {n_rows}")
    print(f"Rows with complete data used for regression: {n_pred}")
    print(f"Residual mean: {round(mean_res,4)}, std: {round(std_res,4)}, threshold: {round(thresh_res,4)}")
    print(f"Flagged rows: {n_flagged17}")
    print("ANSWER APACHE-17 " + json.dumps(answer17, separators=(',', ':')))

    # --- APACHE-18 ---
    answer18, stats18 = apache_18(rows, settings)
    n_valid, mean_d, std_d, thresh_d, n_flagged18 = stats18
    print("\n== APACHE-18 ==")
    print(f"Valid rows (complete settings): {n_valid}")
    print(f"Average nearest-neighbor distance: {round(mean_d,4)}, std: {round(std_d,4)}, threshold: {round(thresh_d,4)}")
    print(f"Flagged rows: {n_flagged18}")
    print("ANSWER APACHE-18 " + json.dumps(answer18, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
