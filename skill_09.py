#!/usr/bin/env python3
"""
skill_09.py

Implements APACHE-17 and APACHE-18.

APACHE-17: Identify configurations whose recorded performance values
unusually differ from what a regression model would predict.

APACHE-18: Flag configurations whose combination of settings is unusually
different from the most similar configuration in the data set, regardless of performance.
"""

import csv
import json
import sys
import math

# ----------------------------------------------------------------------
# Reuse helpers from earlier skills
# ----------------------------------------------------------------------
try:
    # Prefer importing the full regression helpers
    from skill_02 import build_regression, predict, to_float
except Exception:
    # Minimal fallbacks if import fails
    def load_csv(path):
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            data = []
            for row in reader:
                cleaned = {k: (None if v == '?' else v) for k, v in row.items()}
                data.append(cleaned)
        return data, headers

    def to_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def compute_stats(values):
        vals = [v for v in values if v is not None]
        if not vals:
            return (None, None, None, None)
        mean = sum(vals) / len(vals)
        import statistics
        median = statistics.median(vals)
        _min = min(vals)
        _max = max(vals)
        return (mean, median, _min, _max)

    def _transpose(matrix):
        return [list(row) for row in zip(*matrix)]

    def _matmul(A, B):
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
        n = len(A)
        M = [row[:] + [b[i]] for i, row in enumerate(A)]
        for col in range(n):
            pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
            if abs(M[pivot_row][col]) < 1e-12:
                raise ValueError("Matrix is singular")
            if pivot_row != col:
                M[col], M[pivot_row] = M[pivot_row], M[col]
            pivot = M[col][col]
            for j in range(col, n + 1):
                M[col][j] /= pivot
            for r in range(n):
                if r == col:
                    continue
                factor = M[r][col]
                if factor == 0:
                    continue
                for j in range(col, n + 1):
                    M[r][j] -= factor * M[col][j]
        return [M[i][-1] for i in range(n)]

    def build_regression(data, setting_cols, goal_col):
        X = []
        Y = []
        for row in data:
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
            X.append([1] + vals)
            Y.append([goal_val])
        if not X:
            return None
        Xt = _transpose(X)
        XtX = _matmul(Xt, X)
        XtY = _matmul(Xt, Y)
        try:
            coeffs = _solve_linear_system(XtX, [row[0] for row in XtY])
        except Exception:
            return None
        return coeffs

    def predict(row, coeffs, setting_cols):
        features = [1]
        for col in setting_cols:
            v = row.get(col)
            if v is None:
                features.append(0)
            else:
                features.append(1 if v == '1' else 0)
        return sum(c * f for c, f in zip(coeffs, features))

# ----------------------------------------------------------------------
# Parse setting helpers
# ----------------------------------------------------------------------
try:
    from skill_06 import parse_setting, hamming_distance
except Exception:
    def parse_setting(val):
        if val is None:
            return None
        if val == '1':
            return 1
        if val == '0':
            return 0
        try:
            return 1 if int(val) != 0 else 0
        except ValueError:
            return None

    def hamming_distance(row_a, row_b, setting_cols):
        dist = 0
        for col in setting_cols:
            va = parse_setting(row_a.get(col))
            vb = parse_setting(row_b.get(col))
            if va is None or vb is None:
                continue
            if va != vb:
                dist += 1
        return dist

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    data, headers = load_csv(csv_path)

    # Identify goal and setting columns
    goal_cols = [h for h in headers if h.endswith('+') or h.endswith('-')]
    if not goal_cols:
        print("No goal column found.")
        sys.exit(1)
    goal_col = goal_cols[0]  # assume single goal
    setting_cols = [h for h in headers if h not in goal_cols and not h.endswith('X')]

    # ------------------------------------------------------------------
    # APACHE-17: performance anomaly detection
    # ------------------------------------------------------------------
    # Build regression model on all rows with complete data
    coeffs = build_regression(data, setting_cols, goal_col)

    flagged17 = []
    residuals = []

    if coeffs is not None:
        for idx, row in enumerate(data):
            actual_val = to_float(row.get(goal_col))
            if actual_val is None:
                continue
            pred_val = predict(row, coeffs, setting_cols)
            residual = actual_val - pred_val
            residuals.append(residual)
            flagged17.append((abs(residual), idx + 1, actual_val, pred_val, residual))
        # Compute threshold: mean + 1.5*std
        mean_res = sum(residuals) / len(residuals)
        std_res = math.sqrt(sum((r - mean_res) ** 2 for r in residuals) / len(residuals))
        threshold = mean_res + 1.5 * std_res if std_res > 0 else mean_res
        # Flag those with abs difference > threshold
        flagged17 = [f for f in flagged17 if f[0] > threshold]
        flagged17.sort(reverse=True, key=lambda x: x[0])
        flagged17 = flagged17[:20]
    else:
        flagged17 = []

    print("== APACHE-17 ==")
    for f in flagged17:
        row, act, pred, resid = f[1], f[2], f[3], f[4]
        print(f"Row {row}: actual={act:.2f}, predicted={pred:.2f}, diff={resid:.2f}")
    # JSON
    apache17_json = {
        "flagged": [
            {
                "row": f[1],
                "column": goal_col,
                "actual": round(f[2], 2),
                "expected": round(f[3], 2)
            }
            for f in flagged17
        ]
    }
    print("ANSWER APACHE-17 " + json.dumps(apache17_json, separators=(',', ':')))

    # ------------------------------------------------------------------
    # APACHE-18: unusual settings combination detection
    # ------------------------------------------------------------------
    distances = []
    for idx, row in enumerate(data):
        # Find distance to nearest neighbor (excluding self)
        min_dist = None
        for jdx, other in enumerate(data):
            if idx == jdx:
                continue
            dist = hamming_distance(row, other, setting_cols)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                # If distance 0, cannot be smaller, break early
                if min_dist == 0:
                    break
        if min_dist is None:
            min_dist = 0
        distances.append((min_dist, idx + 1))

    # Compute threshold as median + 1.5*IQR
    dist_vals = [d[0] for d in distances]
    dist_vals_sorted = sorted(dist_vals)
    n = len(dist_vals_sorted)
    median = dist_vals_sorted[n // 2]
    # IQR
    q1 = dist_vals_sorted[n // 4]
    q3 = dist_vals_sorted[3 * n // 4]
    iqr = q3 - q1
    threshold_dist = median + 1.5 * iqr if iqr > 0 else median

    flagged18 = [d for d in distances if d[0] > threshold_dist]
    flagged18.sort(reverse=True, key=lambda x: x[0])
    flagged18 = flagged18[:20]
    scores = {str(row): dist for dist, row in flagged18}

    print("== APACHE-18 ==")
    for d in flagged18:
        print(f"Row {d[1]} has unusual settings distance {d[0]}")
    # JSON
    apache18_json = {
        "flagged": [d[1] for d in flagged18],
        "scores": scores
    }
    print("ANSWER APACHE-18 " + json.dumps(apache18_json, separators=(',', ':')))

if __name__ == "__main__":
    main()
