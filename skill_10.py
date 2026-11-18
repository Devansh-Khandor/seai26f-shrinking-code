#!/usr/bin/env python3
"""
skill_10.py: CAR-19 and CAR-20 – comparison of selection strategy and
             performance prediction vs historical average.

The script is a self‑contained solution that follows the specification
exactly.  It reads the supplied CSV, parses all cells, identifies goal
columns, and computes the required metrics.  No third‑party packages are
used – only the Python standard library.
"""

import csv
import json
import math
import random
import sys
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def parse_cell(s: str):
    """Coerce a CSV cell to int, float, or keep string; '?' -> None."""
    if s == "?":
        return None
    for fn in (int, float):
        try:
            return fn(s)
        except ValueError:
            continue
    return s

def read_csv(path: str) -> Tuple[List[str], List[Tuple[int, List[Any]]]]:
    """Return header and rows with 1‑based row numbers."""
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows: List[Tuple[int, List[Any]]] = []
        for i, line in enumerate(reader, 1):
            rows.append((i, [parse_cell(v) for v in line]))
    return header, rows

def is_goal(col: str) -> bool:
    return col.endswith(("+", "-"))

def goal_indices(header: List[str]) -> List[int]:
    return [i for i, c in enumerate(header) if is_goal(c)]

def goal_names(header: List[str]) -> List[str]:
    return [c for c in header if is_goal(c)]

# --------------------------------------------------------------------------- #
# Composite score
# --------------------------------------------------------------------------- #

def compute_goal_bounds(header: List[str], rows: List[Tuple[int, List[Any]]]
                       ) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
    """Return min and max for each goal column (numeric only)."""
    bounds: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
    for idx in goal_indices(header):
        vals = [row[idx] for _, row in rows if isinstance(row[idx], (int, float))]
        if vals:
            bounds[idx] = (min(vals), max(vals))
        else:
            bounds[idx] = (None, None)
    return bounds

def composite_score(row: List[Any], header: List[str],
                    bounds: Dict[int, Tuple[Optional[float], Optional[float]]]
                    ) -> Optional[float]:
    """Average normalized goal score; higher better."""
    scores: List[float] = []
    for idx in goal_indices(header):
        val = row[idx]
        if val is None:
            return None
        lo, hi = bounds[idx]
        if lo is None or hi is None or hi == lo:
            norm = 0.5
        else:
            if header[idx].endswith("-"):
                norm = (hi - val) / (hi - lo)
            else:
                norm = (val - lo) / (hi - lo)
        scores.append(norm)
    return sum(scores) / len(scores) if scores else None

# --------------------------------------------------------------------------- #
# Linear regression helpers (Gaussian elimination)
# --------------------------------------------------------------------------- #

def mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Multiply matrices A (m x n) and B (n x p)."""
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    assert n == n2, "Matrix size mismatch"
    result = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for k in range(n):
            aik = A[i][k]
            if aik:
                for j in range(p):
                    result[i][j] += aik * B[k][j]
    return result

def mat_transpose(A: List[List[float]]) -> List[List[float]]:
    """Transpose matrix A."""
    return [list(row) for row in zip(*A)]

def solve_linear_system(A: List[List[float]], B: List[float]) -> List[float]:
    """Solve Ax = B using Gaussian elimination with partial pivoting."""
    m = len(A)
    # Augmented matrix
    aug = [row[:] + [B[i]] for i, row in enumerate(A)]
    # Forward elimination
    for k in range(m):
        max_row = max(range(k, m), key=lambda i: abs(aug[i][k]))
        if abs(aug[max_row][k]) < 1e-12:
            # Singular: return zeros
            return [0.0] * m
        if max_row != k:
            aug[k], aug[max_row] = aug[max_row], aug[k]
        for i in range(k + 1, m):
            factor = aug[i][k] / aug[k][k]
            for j in range(k, m + 1):
                aug[i][j] -= factor * aug[k][j]
    # Back substitution
    x = [0.0] * m
    for i in reversed(range(m)):
        s = aug[i][m]
        for j in range(i + 1, m):
            s -= aug[i][j] * x[j]
        x[i] = s / aug[i][i]
    return x

def linear_regression(X: List[List[float]], Y: List[float]) -> Tuple[List[float], float]:
    """Return beta coefficients and residual std dev sigma."""
    if not X:
        return [], 0.0
    XTX = mat_mul(mat_transpose(X), X)
    XTY = [sum(X[i][j] * Y[i] for i in range(len(X))) for j in range(len(XTX[0]))]
    beta = solve_linear_system(XTX, XTY)
    # Residuals
    residuals = [Y[i] - sum(beta[j] * X[i][j] for j in range(len(beta))) for i in range(len(X))]
    n = len(X)
    p = len(beta)
    sigma = math.sqrt(sum(r * r for r in residuals) / max(1, n - p))
    return beta, sigma

# --------------------------------------------------------------------------- #
# CAR-19 – selection comparison
# --------------------------------------------------------------------------- #

def car19(header: List[str], rows: List[Tuple[int, List[Any]]]) -> None:
    # Goal bounds for normalization
    bounds = compute_goal_bounds(header, rows)

    # Pre‑compute composite scores for all rows with complete goals
    scored_rows: List[Tuple[int, float]] = []
    for rnum, row in rows:
        score = composite_score(row, header, bounds)
        if score is not None:
            scored_rows.append((rnum, score))

    # Budget
    budget = min(30, len(rows) // 5)
    repeats = 20

    # Our method: take top 'budget' rows by composite score (deterministic)
    scored_rows_sorted = sorted(scored_rows, key=lambda x: x[1], reverse=True)
    our_best_row_num = scored_rows_sorted[0][0] if scored_rows_sorted else None
    our_best_score = scored_rows_sorted[0][1] if scored_rows_sorted else None

    # Random method repeats
    random_best_rows: List[int] = []
    random_best_scores: List[float] = []
    random.seed(0)  # deterministic
    for _ in range(repeats):
        sample = random.sample(rows, budget)
        best_row = None
        best_score = None
        for rnum, row in sample:
            score = composite_score(row, header, bounds)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_row = rnum
        if best_row is None:
            best_row = 0
            best_score = -1
        random_best_rows.append(best_row)
        random_best_scores.append(best_score)

    # Compare
    ours_better = 0
    random_better = 0
    for rb, rs in zip(random_best_scores, [our_best_score]*repeats):
        if our_best_score is None:
            continue
        if our_best_score > rs:
            ours_better += 1
        elif our_best_score < rs:
            random_better += 1
        # ties ignored

    if ours_better > random_better:
        verdict = "ours better"
    elif random_better > ours_better:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    # Output
    print("== CAR-19 ==")
    print(f"Evaluation budget: {budget}")
    print(f"Best row by our method: {our_best_row_num}")
    print(f"Best rows by random selection (one per repeat): {', '.join(map(str, random_best_rows))}")
    print(f"Verdict: {verdict}")

    # JSON answer
    ans = {
        "budget": budget,
        "repeats": repeats,
        "ours": [our_best_row_num] * repeats,
        "random": random_best_rows,
        "verdict": verdict
    }
    print(f"ANSWER CAR-19 {json.dumps(ans, separators=(',', ':'))}")

# --------------------------------------------------------------------------- #
# CAR-20 – performance prediction vs baseline
# --------------------------------------------------------------------------- #

def car20(header: List[str], rows: List[Tuple[int, List[Any]]]) -> None:
    # Identify Mpg+ column
    mpg_col = None
    for idx, c in enumerate(header):
        if c == "Mpg+":
            mpg_col = idx
            break
    if mpg_col is None:
        # No Mpg+ column – cannot compute
        print("== CAR-20 ==")
        print("Mpg+ column missing; cannot compute.")
        print("ANSWER CAR-20 {}")
        return

    # Separate held‑out and training rows
    held_out = [(rnum, r) for rnum, r in rows if rnum % 5 == 0]
    train = [(rnum, r) for rnum, r in rows if rnum % 5 != 0]

    # Baseline: mean Mpg+ of training rows
    mpg_vals_train = [r[mpg_col] for _, r in train if isinstance(r[mpg_col], (int, float))]
    baseline_mean = sum(mpg_vals_train) / len(mpg_vals_train) if mpg_vals_train else 0.0

    # Prepare linear regression:
    # Features: all goal columns except Mpg+
    feature_indices = [i for i in goal_indices(header) if i != mpg_col]
    # Build X and Y for training rows that have complete features and target
    X_train: List[List[float]] = []
    Y_train: List[float] = []
    for _, r in train:
        if any(r[i] is None for i in feature_indices) or r[mpg_col] is None:
            continue
        X_train.append([1.0] + [float(r[i]) for i in feature_indices])
        Y_train.append(float(r[mpg_col]))
    beta, sigma = linear_regression(X_train, Y_train)

    # Predict for held‑out rows
    predictions: Dict[int, float] = {}
    mae_model = 0.0
    mae_baseline = 0.0
    count = 0
    for rnum, r in held_out:
        # Actual Mpg+
        actual = r[mpg_col]
        if not isinstance(actual, (int, float)):
            continue
        # Model prediction
        if any(r[i] is None for i in feature_indices):
            pred = baseline_mean
        else:
            pred = beta[0] + sum(beta[j+1] * float(r[i]) for j, i in enumerate(feature_indices))
        predictions[rnum] = round(pred, 4)
        mae_model += abs(pred - actual)
        mae_baseline += abs(baseline_mean - actual)
        count += 1

    mae_model = mae_model / count if count else 0.0
    mae_baseline = mae_baseline / count if count else 0.0

    if mae_model < mae_baseline:
        verdict = "better"
    elif mae_model > mae_baseline:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    # Output
    print("== CAR-20 ==")
    print(f"Number of held‑out rows: {len(held_out)}")
    print(f"Baseline mean Mpg+: {round(baseline_mean, 4)}")
    print(f"Model MAE: {round(mae_model, 4)}, Baseline MAE: {round(mae_baseline, 4)}")
    print(f"Verdict: {verdict}")

    # JSON answer
    ans = {
        "predictions": predictions,
        "baseline": round(baseline_mean, 4),
        "mae_model": round(mae_model, 4),
        "mae_baseline": round(mae_baseline, 4),
        "verdict": verdict
    }
    print(f"ANSWER CAR-20 {json.dumps(ans, separators=(',', ':'))}")

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv path>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    header, rows = read_csv(path)

    car19(header, rows)
    car20(header, rows)

if __name__ == "__main__":
    main()
