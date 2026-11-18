#!/usr/bin/env python3
"""
skill_07.py

Implements APACHE-13 and APACHE-14: strategies for efficient benchmarking.
"""

import csv
import json
import sys
import statistics

# ----------------------------------------------------------------------
# Import helper functions from skill_02
# ----------------------------------------------------------------------
try:
    from skill_02 import build_regression, predict, to_float
except Exception:
    # Minimal re‑implementations if skill_02 cannot be imported
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
        median = statistics.median(vals)
        _min = min(vals)
        _max = max(vals)
        return (mean, median, _min, _max)

    # Simple linear regression (least squares) using only complete rows
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
# Utility functions
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    # Load data
    try:
        data, headers = load_csv(csv_path)
    except Exception:
        # If load_csv not imported, define locally
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            data = []
            for row in reader:
                cleaned = {k: (None if v == '?' else v) for k, v in row.items()}
                data.append(cleaned)

    # Identify goal and setting columns
    goal_cols = [h for h in headers if h.endswith('+') or h.endswith('-')]
    if not goal_cols:
        print("No goal column found.")
        sys.exit(1)
    goal_col = goal_cols[0]  # Assume single goal for this dataset

    setting_cols = [h for h in headers if h not in goal_cols and not h.endswith('X')]

    # Budget (max rows to evaluate)
    budget = min(30, len(data) // 5)

    # ------------------------------------------------------------------
    # APACHE-13
    # ------------------------------------------------------------------
    evaluated = []
    evaluated_set = set()

    for _ in range(budget):
        if not evaluated:
            # first choice: first row
            next_idx = 0
        else:
            # Build regression on evaluated rows
            training_rows = [data[i] for i in evaluated]
            coeffs = build_regression(training_rows, setting_cols, goal_col)
            if coeffs is None:
                # fallback: pick first remaining
                remaining = [i for i in range(len(data)) if i not in evaluated_set]
                next_idx = remaining[0] if remaining else None
            else:
                remaining = [i for i in range(len(data)) if i not in evaluated_set]
                preds = [(i, predict(data[i], coeffs, setting_cols)) for i in remaining]
                # choose lowest predicted performance
                next_idx = min(preds, key=lambda x: x[1])[0]
        if next_idx is None:
            break
        evaluated.append(next_idx)
        evaluated_set.add(next_idx)

    # Determine best row among evaluated
    if evaluated:
        best_idx = min(evaluated, key=lambda i: to_float(data[i].get(goal_col)))
        best_row_num = best_idx + 1
    else:
        best_row_num = None

    # Output results
    print("== APACHE-13 ==")
    print(f"Evaluated rows: {[i+1 for i in evaluated]}")
    print(f"Best row: {best_row_num}")
    print(f"Budget used: {len(evaluated)}")
    apache13_json = {
        "evaluated": [i + 1 for i in evaluated],
        "best": best_row_num
    }
    print("ANSWER APACHE-13 " + json.dumps(apache13_json, separators=(',', ':')))

    # ------------------------------------------------------------------
    # APACHE-14
    # ------------------------------------------------------------------
    order = []
    order_set = set()

    for _ in range(budget):
        if not order:
            # first pick: first row
            next_idx = 0
        else:
            training_rows = [data[i] for i in order]
            coeffs = build_regression(training_rows, setting_cols, goal_col)
            if coeffs is None:
                remaining = [i for i in range(len(data)) if i not in order_set]
                next_idx = remaining[0] if remaining else None
            else:
                remaining = [i for i in range(len(data)) if i not in order_set]
                preds = [(i, predict(data[i], coeffs, setting_cols)) for i in remaining]
                next_idx = min(preds, key=lambda x: x[1])[0]
        if next_idx is None:
            break
        order.append(next_idx)
        order_set.add(next_idx)

    print("== APACHE-14 ==")
    print(f"Proposed order (top {len(order)} rows): {[i+1 for i in order]}")
    apache14_json = {
        "order": [i + 1 for i in order]
    }
    print("ANSWER APACHE-14 " + json.dumps(apache14_json, separators=(',', ':')))

if __name__ == "__main__":
    main()
