#!/usr/bin/env python3
"""
skill_04.py <csv_path>

Implements NASA-07 (Arrival assessment with alerts) and NASA-08 (Historical summaries & shortlist).
"""

import sys
import csv
import json
import math
from statistics import mean

# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
def parse_csv(path):
    """Read CSV into list of dict rows."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def to_float(val):
    """Convert a cell value to float, or None for missing."""
    if val == '?' or val == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

# ------------------------------------------------------------------
# Statistics helpers
# ------------------------------------------------------------------
def compute_goal_stats(rows, col):
    """Return stats dict for a goal column: n, mean, min, max."""
    vals = [to_float(r[col]) for r in rows if to_float(r[col]) is not None]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {"n": n, "mean": mean(vals), "min": min(vals), "max": max(vals)}

def std_of(values):
    """Sample standard deviation; returns 0 if insufficient data."""
    n = len(values)
    if n < 2:
        return 0.0
    m = mean(values)
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    return math.sqrt(var)

# ------------------------------------------------------------------
# Linear regression helpers
# ------------------------------------------------------------------
def fit_simple_regression(rows, xcol, ycol):
    """Fit y = a * x. Returns slope a."""
    X = [to_float(r[xcol]) for r in rows if to_float(r[xcol]) is not None and to_float(r[ycol]) is not None]
    Y = [to_float(r[ycol]) for r in rows if to_float(r[xcol]) is not None and to_float(r[ycol]) is not None]
    n = len(X)
    if n == 0:
        return 0.0
    x_mean = sum(X) / n
    y_mean = sum(Y) / n
    num = sum((X[i] - x_mean) * (Y[i] - y_mean) for i in range(n))
    den = sum((X[i] - x_mean) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0

# ------------------------------------------------------------------
# Multiple regression (3x3)
# ------------------------------------------------------------------
def transpose(A):
    return [list(col) for col in zip(*A)]

def matrix_multiply(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]

def invert_3x3(M):
    a, b, c = M[0]
    d, e, f = M[1]
    g, h, i = M[2]
    det = (a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g))
    if abs(det) < 1e-12:
        return None
    inv_det = 1.0 / det
    inv = [
        [(e*i - f*h) * inv_det, (c*h - b*i) * inv_det, (b*f - c*e) * inv_det],
        [(f*g - d*i) * inv_det, (a*i - c*g) * inv_det, (c*d - a*f) * inv_det],
        [(d*h - e*g) * inv_det, (b*g - a*h) * inv_det, (a*e - b*d) * inv_det],
    ]
    return inv

def fit_multiple_regression(rows, xcols, ycol):
    """
    Fit y = beta0 + beta1*x1 + beta2*x2
    xcols: list of columns for x1,x2
    Returns beta list [beta0, beta1, beta2]
    """
    X = []
    Y = []
    for r in rows:
        y = to_float(r[ycol])
        if y is None:
            continue
        xi = [1.0]
        skip = False
        for c in xcols:
            x = to_float(r[c])
            if x is None:
                skip = True
                break
            xi.append(x)
        if skip:
            continue
        X.append(xi)
        Y.append(y)
    n = len(X)
    if n == 0:
        return [0.0, 0.0, 0.0]
    Xt = transpose(X)
    XtX = matrix_multiply(Xt, X)
    XtY = [sum(Xt[i][j] * Y[j] for j in range(n)) for i in range(len(Xt))]
    inv_XtX = invert_3x3(XtX)
    if inv_XtX is None:
        return [0.0, 0.0, 0.0]
    beta = [sum(inv_XtX[i][j] * XtY[j] for j in range(3)) for i in range(3)]
    return beta

# ------------------------------------------------------------------
# Arrival assessment
# ------------------------------------------------------------------
def assess_arrivals(rows, arrivals, history_len):
    """Return list of assessment dicts for each arrival."""
    assessments = []
    # Pre-compute cumulative goal lists for thresholds
    cumulative_effort_vals = []
    cumulative_defect_vals = []
    for idx, r in enumerate(rows[:history_len], start=1):
        e = to_float(r["Effort-"])
        d = to_float(r["Defects-"])
        if e is not None:
            cumulative_effort_vals.append(e)
        if d is not None:
            cumulative_defect_vals.append(d)
    # Process each arrival in order
    for arrival_idx in arrivals:
        # Training rows: all rows before current arrival
        training_rows = rows[:arrival_idx-1]
        # Update cumulative lists with training rows
        for r in training_rows[len(cumulative_effort_vals)+len(cumulative_defect_vals):]:
            # Actually cumulative lists already contain training rows; skip this part
            pass

        # Simple regression for effort
        a = fit_simple_regression(training_rows, "Kloc+", "Effort-")
        # Predict effort
        kloc = to_float(rows[arrival_idx-1]["Kloc+"])
        pred_effort = a * kloc if kloc is not None else 0.0

        # Multiple regression for defects (using predicted effort)
        beta = fit_multiple_regression(training_rows, ["Kloc+", "Effort-"], "Defects-")
        pred_defects = beta[0] + beta[1]*kloc + beta[2]*pred_effort if kloc is not None else 0.0

        # Compute thresholds from training data only
        train_eff_vals = [to_float(r["Effort-"]) for r in training_rows if to_float(r["Effort-"]) is not None]
        train_def_vals = [to_float(r["Defects-"]) for r in training_rows if to_float(r["Defects-"]) is not None]
        eff_mean = mean(train_eff_vals) if train_eff_vals else 0.0
        def_mean = mean(train_def_vals) if train_def_vals else 0.0
        eff_std = std_of(train_eff_vals)
        def_std = std_of(train_def_vals)
        eff_threshold = eff_mean + 2*eff_std if eff_std > 0 else eff_mean
        def_threshold = def_mean + 2*def_std if def_std > 0 else def_mean

        alert = (pred_effort > eff_threshold) or (pred_defects > def_threshold)

        assessments.append({
            "row": arrival_idx,
            "predicted_effort": pred_effort,
            "predicted_defects": pred_defects,
            "alert": alert
        })
    return assessments

# ------------------------------------------------------------------
# Historical summaries and shortlist
# ------------------------------------------------------------------
def build_checkpoints(rows, history_len, arrival_counts):
    """Return list of checkpoint dicts."""
    checkpoints = []
    goal_cols = ["Kloc+", "Effort-", "Defects-"]
    for m in arrival_counts:
        upto = history_len + m
        subset = rows[:upto]
        summary = {}
        for g in goal_cols:
            stats = compute_goal_stats(subset, g)
            summary[g] = {"n": stats["n"], "mean": stats["mean"], "min": stats["min"], "max": stats["max"]}
        # Shortlist: best 5 rows seen so far by composite score
        candidates = []
        for r in subset:
            k = to_float(r["Kloc+"])
            e = to_float(r["Effort-"])
            d = to_float(r["Defects-"])
            if k is None or e is None or d is None:
                continue
            score = k - e - d
            # Keep row number (1-based)
            candidates.append((score, r))
        candidates.sort(key=lambda x: (-x[0],))  # descending
        shortlist = [rows.index(c[1]) + 1 for c in candidates[:5]]
        checkpoints.append({"after": m, "summary": summary, "shortlist": shortlist})
    return checkpoints

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_04.py <csv_path>\n")
        sys.exit(1)
    path = sys.argv[1]
    rows = parse_csv(path)
    n_rows = len(rows)

    # Determine arrivals
    arrivals_count = round(0.2 * n_rows)
    history_len = n_rows - arrivals_count
    arrivals = list(range(history_len + 1, n_rows + 1))  # 1-based indices

    # NASA-07
    print("== NASA-07 ==")
    assessments = assess_arrivals(rows, arrivals, history_len)
    print(f"Total arrivals processed: {len(assessments)}")
    for a in assessments:
        print(f"Row {a['row']} predicted Effort {a['predicted_effort']:.4f} Predicted Defects {a['predicted_defects']:.4f} alert={a['alert']}")
    print("ANSWER NASA-07", json.dumps({"assessments": assessments}, separators=(',', ':')))

    # NASA-08
    print("\n== NASA-08 ==")
    checkpoints = build_checkpoints(rows, history_len, [0, 9, 19])
    print(f"Checkpoints processed: {len(checkpoints)}")
    for cp in checkpoints:
        print(f"After {cp['after']} arrivals: {cp['after']} rows processed")
        print(f"Shortlist: {cp['shortlist']}")
    print("ANSWER NASA-08", json.dumps({"checkpoints": checkpoints}, separators=(',', ':')))

if __name__ == "__main__":
    main()
