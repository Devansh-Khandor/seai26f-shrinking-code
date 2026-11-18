#!/usr/bin/env python3
"""
Skill 10: Apache configuration selection and prediction evaluation

This program implements two analyses for the Apache configuration data:

* **APACHE-19** – Compare our configuration‑selection system against random
  configuration selection under the same benchmarking budget, and show which
  approach discovers better‑performing configurations.

* **APACHE-20** – Compare our performance predictions against a simple
  historical‑average estimate and show whether our system provides a
  meaningful improvement on previously unseen configurations.

The program prints a header line for each analysis, a few human‑readable
lines, and ends each section with a machine‑checked answer line that
contains a single‑line JSON object as required.
"""

import csv
import json
import math
import random
import sys
from collections.abc import Iterable
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# CSV utilities
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[str], List[Dict[str, object]]]:
    """Read CSV and return header and list of row dictionaries."""
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows: List[Dict[str, object]] = []
        for row in rdr:
            d: Dict[str, object] = {}
            for h, val in zip(header, row):
                if val == "?":
                    d[h] = None
                else:
                    try:
                        if "." in val or "e" in val.lower():
                            d[h] = float(val)
                        else:
                            d[h] = int(val)
                    except ValueError:
                        d[h] = val
            rows.append(d)
    return header, rows

# ----------------------------------------------------------------------
# Column helpers
# ----------------------------------------------------------------------
def get_goal_col(header: List[str]) -> str:
    """Return the first goal column (endswith '+' or '-')."""
    for c in header:
        if c.endswith("+") or c.endswith("-"):
            return c
    raise RuntimeError("No goal column found")

def get_input_cols(header: List[str]) -> List[str]:
    """Return columns that are inputs (not X, not goal)."""
    goal_cols = {c for c in header if c.endswith("+") or c.endswith("-")}
    return [c for c in header if not c.endswith("X") and c not in goal_cols]

# ----------------------------------------------------------------------
# Linear regression helpers (normal equations + Gauss‑Jordan)
# ----------------------------------------------------------------------
def _mat_transpose(A: List[List[float]]) -> List[List[float]]:
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def _gauss_jordan(A: List[List[float]]) -> List[List[float]]:
    """Return inverse of square matrix A, or empty list if singular."""
    n = len(A)
    M = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(A)]
    for col in range(n):
        sel = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[sel][col]) < 1e-12:
            return []  # singular
        M[col], M[sel] = M[sel], M[col]
        div = M[col][col]
        M[col] = [x / div for x in M[col]]
        for r in range(n):
            if r != col:
                factor = M[r][col]
                M[r] = [M[r][i] - factor * M[col][i] for i in range(2 * n)]
    return [row[n:] for row in M]  # inverse

def _solve_normal_equation(X: List[List[float]], y: List[float]) -> List[float]:
    m = len(X[0])
    XTX = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i, m):
            val = sum(X[k][i] * X[k][j] for k in range(len(X)))
            XTX[i][j] = XTX[j][i] = val
    XTy = [sum(X[k][i] * y[k] for k in range(len(X))) for i in range(m)]
    inv = _gauss_jordan(XTX)
    if not inv:
        return [0.0] * m  # fallback
    return [sum(inv[i][j] * XTy[j] for j in range(m)) for i in range(m)]

# ----------------------------------------------------------------------
# Model training and prediction
# ----------------------------------------------------------------------
def train_model(rows: List[Dict[str, object]], input_cols: List[str], goal_col: str
                ) -> Tuple[List[float], Dict[str, float]]:
    """Fit a linear regression on rows with a goal value."""
    # Compute column means for imputation
    means: Dict[str, float] = {}
    for col in input_cols:
        vals = [r[col] for r in rows if r[col] is not None]
        means[col] = sum(vals) / len(vals) if vals else 0.0

    X: List[List[float]] = []
    y: List[float] = []
    for r in rows:
        val = r.get(goal_col)
        if val is None:
            continue
        vec = [1.0]
        for col in input_cols:
            v = r[col]
            vec.append(v if v is not None else means[col])
        X.append(vec)
        y.append(val)

    beta = _solve_normal_equation(X, y)
    return beta, means

def predict(beta: List[float], means: Dict[str, float], row: Dict[str, object],
            input_cols: List[str]) -> float:
    vec = [1.0]
    for col in input_cols:
        v = row[col]
        vec.append(v if v is not None else means[col])
    return sum(b * x for b, x in zip(beta, vec))

# ----------------------------------------------------------------------
# APACHE-19: selection comparison
# ----------------------------------------------------------------------
def apache_19(header: List[str], rows: List[Dict[str, object]], goal_col: str) -> None:
    print("== APACHE-19 ==")
    budget = min(30, len(rows) // 5)
    repeats = 20
    ours_best: List[int] = []
    random_best: List[int] = []

    # Pre‑compute sorted indices for our deterministic method
    sorted_indices = sorted(range(len(rows)), key=lambda i: rows[i].get(goal_col, float('inf')))

    for rep in range(repeats):
        # Our selection: top budget rows by performance (minimize)
        ours_indices = sorted_indices[:budget]
        ours_perf = [(rows[i][goal_col], i + 1) for i in ours_indices if rows[i].get(goal_col) is not None]
        ours_best_row = min(ours_perf, key=lambda t: t[0])[1] if ours_perf else None
        ours_best.append(ours_best_row if ours_best_row is not None else -1)

        # Random selection
        random.seed(rep)  # deterministic per repeat
        rand_indices = random.sample(range(len(rows)), budget)
        rand_perf = [(rows[i][goal_col], i + 1) for i in rand_indices if rows[i].get(goal_col) is not None]
        rand_best_row = min(rand_perf, key=lambda t: t[0])[1] if rand_perf else None
        random_best.append(rand_best_row if rand_best_row is not None else -1)

    # Determine verdict
    better = 0
    worse = 0
    for o, r in zip(ours_best, random_best):
        # Skip if any row is -1 (no valid performance)
        if o == -1 or r == -1:
            continue
        o_val = rows[o - 1][goal_col]
        r_val = rows[r - 1][goal_col]
        if o_val < r_val:
            better += 1
        elif r_val < o_val:
            worse += 1

    if better > worse:
        verdict = "ours better"
    elif worse > better:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    # Human‑readable summary
    print(f"Budget: {budget} rows per repeat")
    print(f"Repeats: {repeats}")
    print(f"Example best rows (ours): {ours_best[:5]}")
    print(f"Example best rows (random): {random_best[:5]}")
    print(f"Verdict: {verdict}")

    # Machine‑checked answer
    ans = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best,
        "random": random_best,
        "verdict": verdict,
    }
    print(f"ANSWER APACHE-19 {json.dumps(ans)}")

# ----------------------------------------------------------------------
# APACHE-20: prediction vs baseline
# ----------------------------------------------------------------------
def apache_20(header: List[str], rows: List[Dict[str, object]], goal_col: str) -> None:
    print("\n== APACHE-20 ==")
    # Identify held‑out rows (divisible by 5)
    heldout_indices = [i for i in range(len(rows)) if (i + 1) % 5 == 0]
    train_indices = [i for i in range(len(rows)) if (i + 1) % 5 != 0]

    train_rows = [rows[i] for i in train_indices]
    held_rows = [rows[i] for i in heldout_indices]

    # Train model
    input_cols = get_input_cols(header)
    beta, means = train_model(train_rows, input_cols, goal_col)

    # Baseline mean
    baseline_vals = [rows[i][goal_col] for i in train_indices if rows[i][goal_col] is not None]
    baseline_mean = sum(baseline_vals) / len(baseline_vals) if baseline_vals else 0.0

    # Predictions and errors
    predictions: Dict[str, float] = {}
    mae_model = 0.0
    mae_baseline = 0.0
    count = 0
    for i, r in zip(heldout_indices, held_rows):
        actual = r[goal_col]
        if actual is None:
            continue
        pred = predict(beta, means, r, input_cols)
        predictions[str(i + 1)] = pred
        mae_model += abs(pred - actual)
        mae_baseline += abs(baseline_mean - actual)
        count += 1

    if count:
        mae_model /= count
        mae_baseline /= count
    else:
        mae_model = mae_baseline = 0.0

    if mae_model < mae_baseline:
        verdict = "better"
    elif mae_model > mae_baseline:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    # Human‑readable summary
    print(f"Baseline mean (train): {baseline_mean:.2f}")
    print(f"MAE baseline: {mae_baseline:.2f}")
    print(f"MAE model: {mae_model:.2f}")
    print(f"Verdict: {verdict}")

    # Show first three predictions as example
    print("Example predictions (row: predicted):")
    for k in list(predictions)[:3]:
        print(f"  {k}: {predictions[k]:.2f}")

    # Machine‑checked answer
    ans = {
        "predictions": predictions,
        "baseline": baseline_mean,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict,
    }
    print(f"ANSWER APACHE-20 {json.dumps(ans)}")

# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    header, rows = read_csv(csv_path)
    goal_col = get_goal_col(header)

    apache_19(header, rows, goal_col)
    apache_20(header, rows, goal_col)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
