#!/usr/bin/env python3
import csv
import json
import sys
import statistics
from collections import defaultdict

# ----------------------------------------------------------------------
# Helper functions (mostly from skill_01)
# ----------------------------------------------------------------------
def load_csv(path):
    """
    Load the CSV file and return:
      - data: list of dicts (each row)
      - headers: list of column names
    Missing values represented by '?'
    """
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        data = []
        for row in reader:
            # Convert '?' to None, other values stay as strings
            cleaned = {k: (None if v == '?' else v) for k, v in row.items()}
            data.append(cleaned)
    return data, headers

def to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

def compute_stats(values):
    """Return mean, median, min, max for a list of floats"""
    vals = [v for v in values if v is not None]
    if not vals:
        return (None, None, None, None)
    mean = sum(vals) / len(vals)
    median = statistics.median(vals)
    _min = min(vals)
    _max = max(vals)
    return (mean, median, _min, _max)

def best_worst(values, direction):
    """direction: '+' to maximize, '-' to minimize"""
    vals = [v for v in values if v is not None]
    if not vals:
        return (None, None)
    if direction == '+':
        return (max(vals), min(vals))
    else:
        return (min(vals), max(vals))

# ----------------------------------------------------------------------
# Linear regression helpers
# ----------------------------------------------------------------------
def _transpose(matrix):
    return [list(row) for row in zip(*matrix)]

def _matmul(A, B):
    """Multiply matrices A (m x n) and B (n x p)"""
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    result = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for k in range(n):
            aik = A[i][k]
            if aik == 0:
                continue
            for j in range(p):
                result[i][j] += aik * B[k][j]
    return result

def _solve_linear_system(A, b):
    """
    Solve Ax = b using Gaussian elimination with partial pivoting.
    A is square matrix (n x n), b is vector (n).
    Returns x as list of floats.
    """
    n = len(A)
    # Build augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for col in range(n):
        # Pivot
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot_row][col]) < 1e-12:
            raise ValueError("Matrix is singular")
        # Swap
        if pivot_row != col:
            M[col], M[pivot_row] = M[pivot_row], M[col]
        # Normalize
        pivot = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pivot
        # Eliminate
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if factor == 0:
                continue
            for j in range(col, n + 1):
                M[r][j] -= factor * M[col][j]
    # Solution
    return [M[i][-1] for i in range(n)]

def build_regression(data, setting_cols, goal_col):
    """
    Build linear regression model using rows with complete data.
    Returns coefficients list (including intercept first).
    """
    X = []
    Y = []
    for row in data:
        # Skip rows with missing goal or missing any setting
        goal_val = to_float(row.get(goal_col))
        if goal_val is None:
            continue
        vals = []
        skip = False
        for col in setting_cols:
            v = row.get(col)
            if v is None:
                skip = True
                break
            if v not in ('0', '1'):
                try:
                    int(v)
                except ValueError:
                    skip = True
                    break
            vals.append(1 if v == '1' else 0)
        if skip:
            continue
        # Add intercept
        X.append([1] + vals)
        Y.append([goal_val])
    if not X:
        return None  # no data to train
    Xt = _transpose(X)
    XtX = _matmul(Xt, X)
    XtY = _matmul(Xt, Y)
    try:
        coeffs = _solve_linear_system(XtX, [row[0] for row in XtY])
    except Exception:
        return None
    return coeffs

def predict(row, coeffs, setting_cols):
    """Predict goal value for a single row."""
    features = [1]
    for col in setting_cols:
        v = row.get(col)
        if v is None:
            features.append(0)
        else:
            features.append(1 if v == '1' else 0)
    return sum(c * f for c, f in zip(coeffs, features))

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_02.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    data, headers = load_csv(csv_path)

    # Identify goal columns (ending with '+' or '-')
    goal_cols = [h for h in headers if h.endswith('+') or h.endswith('-')]
    # Identify setting columns (all except goal columns and columns ending in 'X')
    setting_cols = [h for h in headers if h not in goal_cols and not h.endswith('X')]
    # For our dataset the main goal is Performance-
    perf_col = 'Performance-'

    # --------------------------------------------------------------
    # APACHE-01
    # --------------------------------------------------------------
    perf_values = [to_float(row.get(perf_col)) for row in data]
    mean, median, _min, _max = compute_stats(perf_values)
    best, worst = best_worst(perf_values, '-')  # '-' to minimize

    print("== APACHE-01 ==")
    print(f"Performance- mean: {mean:.2f}")
    print(f"Performance- median: {median:.2f}")
    print(f"Performance- min: {_min:.2f}")
    print(f"Performance- max: {_max:.2f}")
    print(f"Performance- best: {best:.2f}")
    print(f"Performance- worst: {worst:.2f}")

    apache01_json = {
        perf_col: {
            "mean": mean,
            "median": median,
            "min": _min,
            "max": _max,
            "best": best,
            "worst": worst
        }
    }
    print("ANSWER APACHE-01 " + json.dumps(apache01_json, separators=(',', ':')))

    # --------------------------------------------------------------
    # APACHE-02
    # --------------------------------------------------------------
    valid_perf = [(idx, to_float(row.get(perf_col))) for idx, row in enumerate(data)
                  if to_float(row.get(perf_col)) is not None]
    if not valid_perf:
        top_indices = []
        threshold = None
    else:
        sorted_vals = sorted([v for _, v in valid_perf])
        q1_index = len(sorted_vals) // 4
        threshold = sorted_vals[q1_index] if q1_index < len(sorted_vals) else sorted_vals[-1]
        top_indices = [idx for idx, v in valid_perf if v <= threshold]
        if len(top_indices) < 5:
            top_indices = [idx for _, idx in sorted(valid_perf, key=lambda x: x[1])[:5]]

    top_rows = [idx + 1 for idx in top_indices]  # 1-based row numbers

    setting_stats = []
    for col in setting_cols:
        top_ones = top_non_missing = 0
        rest_ones = rest_non_missing = 0
        for idx, row in enumerate(data):
            val = row.get(col)
            if val is None:
                continue
            if val not in ('0', '1'):
                try:
                    val_int = int(val)
                except ValueError:
                    continue
                val = str(val_int)
            is_one = val == '1'
            if idx in top_indices:
                top_non_missing += 1
                if is_one:
                    top_ones += 1
            else:
                rest_non_missing += 1
                if is_one:
                    rest_ones += 1
        top_share = (top_ones / top_non_missing) if top_non_missing else None
        rest_share = (rest_ones / rest_non_missing) if rest_non_missing else None
        if top_share is not None and rest_share is not None:
            diff = abs(top_share - rest_share)
            setting_stats.append((diff, col, top_share, rest_share))

    setting_stats.sort(reverse=True)
    top_settings = setting_stats[:max(2, len(setting_stats))] if setting_stats else []

    settings_output = []
    for _, col, top_share, rest_share in top_settings:
        settings_output.append({
            "setting": col,
            "top_share": round(top_share, 4),
            "rest_share": round(rest_share, 4)
        })

    print("== APACHE-02 ==")
    print(f"Top configurations (row numbers): {top_rows}")
    print("Distinguishing settings:")
    for s in settings_output:
        print(f"{s['setting']}: top {s['top_share']*100:.1f}%, rest {s['rest_share']*100:.1f}%")

    apache02_json = {
        "top": top_rows,
        "settings": settings_output
    }
    print("ANSWER APACHE-02 " + json.dumps(apache02_json, separators=(',', ':')))

    # --------------------------------------------------------------
    # APACHE-03: Predict performance for held‑out rows
    # --------------------------------------------------------------
    # Identify held‑out rows: numbers divisible by 5
    held_out_indices = [i for i, _ in enumerate(data) if (i + 1) % 5 == 0]
    # Build regression model on non held‑out rows
    training_rows = [row for i, row in enumerate(data) if i not in held_out_indices]
    coeffs = build_regression(training_rows, setting_cols, perf_col)

    predictions = {}
    errors = []
    if coeffs is not None:
        for i in held_out_indices:
            row = data[i]
            true_val = to_float(row.get(perf_col))
            if true_val is None:
                continue
            pred_val = predict(row, coeffs, setting_cols)
            predictions[str(i + 1)] = pred_val
            errors.append(abs(pred_val - true_val))
    mae = sum(errors) / len(errors) if errors else None

    print("== APACHE-03 ==")
    for row_no, pred in sorted(predictions.items(), key=lambda x: int(x[0])):
        print(f"Row {row_no}: predicted Performance- = {pred:.2f}")
    print(f"Mean absolute error on held‑out rows: {mae:.2f}" if mae is not None else "No predictions")

    apache03_json = {
        "predictions": predictions,
        "mae": mae
    }
    print("ANSWER APACHE-03 " + json.dumps(apache03_json, separators=(',', ':')))

    # --------------------------------------------------------------
    # APACHE-04: Compare pairs differing by one setting
    # --------------------------------------------------------------
    # Use rows with complete data (same as training set)
    if coeffs is None:
        pairs_list = []
    else:
        pairs_list = []
        # Map row index to row dict for quick access
        complete_rows = [ (i, r) for i, r in enumerate(training_rows) ]
        n = len(complete_rows)
        found = 0
        # Find at least 5 pairs
        for a_idx in range(n):
            if found >= 5:
                break
            for b_idx in range(a_idx + 1, n):
                if found >= 5:
                    break
                a_row = complete_rows[a_idx][1]
                b_row = complete_rows[b_idx][1]
                # Count differing settings
                diffs = []
                for col in setting_cols:
                    va = a_row.get(col)
                    vb = b_row.get(col)
                    if va is None or vb is None:
                        continue
                    if (va == '1') != (vb == '1'):
                        diffs.append(col)
                if len(diffs) == 1:
                    setting = diffs[0]
                    # Predictions
                    pred_a = predict(a_row, coeffs, setting_cols)
                    pred_b = predict(b_row, coeffs, setting_cols)
                    diff_est = pred_b - pred_a
                    low = diff_est - mae if mae is not None else diff_est
                    high = diff_est + mae if mae is not None else diff_est
                    pairs_list.append({
                        "a": complete_rows[a_idx][0] + 1,   # convert to 1‑based
                        "b": complete_rows[b_idx][0] + 1,
                        "setting": setting,
                        "difference": diff_est,
                        "low": low,
                        "high": high
                    })
                    found += 1

    print("== APACHE-04 ==")
    for p in pairs_list:
        print(f"Row {p['a']} vs Row {p['b']} differ in {p['setting']}: diff ≈ {p['difference']:.2f} "
              f"({p['low']:.2f} – {p['high']:.2f})")
    if not pairs_list:
        print("No pairs found")

    apache04_json = {"pairs": pairs_list}
    print("ANSWER APACHE-04 " + json.dumps(apache04_json, separators=(',', ':')))

if __name__ == "__main__":
    main()
