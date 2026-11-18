#!/usr/bin/env python3
"""
CAR-19: Compare our vehicle-selection system against random selection with same budget.
CAR-20: Evaluate our Mpg+ predictions versus historical-average baseline on held‑out rows.
"""

import csv
import json
import math
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

# ---------- CSV parsing -------------------------------------------------
def read_csv(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return list of rows (dict of column->value) and header list. Missing cells become None."""
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for rnum, row in enumerate(reader, start=1):
            parsed: Dict[str, Any] = {}
            for k, v in row.items():
                if v is None or v.strip() == "" or v.strip() == "?":
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            parsed["_rownum"] = rnum
            rows.append(parsed)
    return rows, headers


# ---------- helpers -------------------------------------------------------
def goal_columns(headers: List[str]) -> List[str]:
    return [c for c in headers if c.endswith(("+", "-"))]


def x_columns(headers: List[str]) -> List[str]:
    return [c for c in headers if c.endswith("X")]


def numeric_columns(headers: List[str]) -> List[str]:
    nums = []
    for h in headers:
        if h in ("_rownum",):
            continue
        try:
            float(h)
            nums.append(h)
        except ValueError:
            pass
    # We'll detect numeric by checking sample values later
    return nums


def compute_bounds(rows: List[Dict[str, Any]], cols: List[str]) -> Dict[str, Tuple[float, float]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    for c in cols:
        vals = [r[c] for r in rows if isinstance(r.get(c), (int, float))]
        if vals:
            bounds[c] = (min(vals), max(vals))
        else:
            bounds[c] = (0.0, 1.0)
    return bounds


def overall_score(row: Dict[str, Any], goal_cols: List[str], bounds: Dict[str, Tuple[float, float]]) -> float:
    """Higher is better."""
    s = 0.0
    for g in goal_cols:
        v = row.get(g)
        if v is None or not isinstance(v, (int, float)):
            # ignore rows with missing goals for scoring
            return -float("inf")
        lo, hi = bounds[g]
        if lo == hi:
            norm = 0.0
        else:
            if g.endswith("+"):
                norm = (v - lo) / (hi - lo)
            else:
                norm = (hi - v) / (hi - lo)
        s += norm
    return s


# ---------- linear regression helpers ------------------------------------
def solve_linear(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """Solve Ax = b with partial pivoting. Returns None if singular."""
    n = len(A)
    # Augment
    aug = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        # Pivot
        max_row = max(range(i, n), key=lambda r: abs(aug[r][i]))
        if abs(aug[max_row][i]) < 1e-12:
            return None
        aug[i], aug[max_row] = aug[max_row], aug[i]
        # Normalize
        pivot = aug[i][i]
        for j in range(i, n + 1):
            aug[i][j] /= pivot
        # Eliminate
        for r in range(n):
            if r == i:
                continue
            factor = aug[r][i]
            for j in range(i, n + 1):
                aug[r][j] -= factor * aug[i][j]
    return [aug[i][-1] for i in range(n)]


# ---------- CAR-19 implementation ---------------------------------------
def car19(rows: List[Dict[str, Any]], goal_cols: List[str]) -> Dict[str, Any]:
    num_rows = len(rows)
    budget = min(30, num_rows // 5)
    repeats = 20
    bounds = compute_bounds(rows, goal_cols)

    # Precompute scores for all rows
    scores = []
    for r in rows:
        sc = overall_score(r, goal_cols, bounds)
        if sc > -float("inf"):
            scores.append((sc, r["_rownum"], r))
    # Our selection: top budget rows by score (deterministic)
    scores.sort(key=lambda x: (-x[0], x[1]))
    ours_selection = [row for _, _, row in scores[:budget]]

    ours_best_scores = []
    random_best_scores = []

    ours_best_rows = []
    random_best_rows = []

    for _ in range(repeats):
        # Ours
        best_ours = max(ours_selection, key=lambda r: overall_score(r, goal_cols, bounds))
        ours_best_scores.append(overall_score(best_ours, goal_cols, bounds))
        ours_best_rows.append(best_ours["_rownum"])

        # Random
        rand_sel = random.sample(rows, budget)
        best_rand = max(rand_sel, key=lambda r: overall_score(r, goal_cols, bounds))
        random_best_scores.append(overall_score(best_rand, goal_cols, bounds))
        random_best_rows.append(best_rand["_rownum"])

    # Count comparisons
    ours_better = sum(1 for o, r in zip(ours_best_scores, random_best_scores) if o > r)
    random_better = sum(1 for o, r in zip(ours_best_scores, random_best_scores) if r > o)

    if ours_better > random_better:
        verdict = "ours better"
    elif random_better > ours_better:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    result = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best_rows,
        "random": random_best_rows,
        "verdict": verdict
    }
    return result


# ---------- CAR-20 implementation ---------------------------------------
def car20(rows: List[Dict[str, Any]], goal_cols: List[str]) -> Dict[str, Any]:
    # Identify held-out rows (divisible by 5)
    held_out = [r for r in rows if r["_rownum"] % 5 == 0]
    # Training rows: non-held-out
    train_rows = [r for r in rows if r["_rownum"] % 5 != 0]

    mpg_col = "Mpg+"
    # Filter training rows that have MPG
    train_rows = [r for r in train_rows if isinstance(r.get(mpg_col), (int, float))]
    # Feature columns: all numeric columns that are not goals and not X
    headers = list(rows[0].keys())
    x_cols = x_columns(headers)
    all_numeric = [c for c in headers if isinstance(r.get(c), (int, float)) for r in rows]
    feature_cols = [c for c in all_numeric if c not in goal_cols and c not in x_cols and c != "_rownum"]

    # Compute feature means for imputation
    feat_means: Dict[str, float] = {}
    for f in feature_cols:
        vals = [r[f] for r in train_rows if isinstance(r.get(f), (int, float))]
        if vals:
            feat_means[f] = sum(vals) / len(vals)
        else:
            feat_means[f] = 0.0

    # Build X matrix and y vector
    X = []
    y = []
    for r in train_rows:
        row_feats = []
        skip = False
        for f in feature_cols:
            v = r.get(f)
            if v is None or not isinstance(v, (int, float)):
                skip = True
                break
            row_feats.append(v)
        if skip:
            continue
        X.append([1.0] + row_feats)  # bias term
        y.append(r[mpg_col])

    if not X:
        # No training data: fallback to baseline predictions
        baseline = 0.0
        mae_baseline = 0.0
        mae_model = 0.0
        predictions = {}
        verdict = "no clear difference"
    else:
        # Solve normal equations: (X^T X) w = X^T y
        XT = list(map(list, zip(*X)))
        A = [[sum(XT[i][k] * XT[j][k] for k in range(len(XT[0]))) for j in range(len(TX := XT))] for i in range(len(TX))]
        b = [sum(XT[i][k] * y[k] for k in range(len(y))) for i in range(len(TX))]
        weights = solve_linear(A, b)
        if weights is None:
            weights = [0.0] * len(A)

        # Baseline mean of training MPG
        baseline = sum(y) / len(y)

        predictions = {}
        errors_model = []
        errors_baseline = []

        for r in held_out:
            # skip if actual MPG missing
            actual = r.get(mpg_col)
            if actual is None or not isinstance(actual, (int, float)):
                continue
            # Prepare feature vector
            vec = [1.0]
            for f in feature_cols:
                v = r.get(f)
                if v is None or not isinstance(v, (int, float)):
                    v = feat_means[f]
                vec.append(v)
            # Predict
            est = sum(w * v for w, v in zip(weights, vec))
            predictions[r["_rownum"]] = round(est, 6)
            errors_model.append(abs(est - actual))
            errors_baseline.append(abs(baseline - actual))

        mae_model = sum(errors_model) / len(errors_model) if errors_model else 0.0
        mae_baseline = sum(errors_baseline) / len(errors_baseline) if errors_baseline else 0.0

        # Verdict logic
        if mae_model < mae_baseline - 0.01:
            verdict = "better"
        elif mae_model > mae_baseline + 0.01:
            verdict = "worse"
        else:
            verdict = "no clear difference"

    result = {
        "predictions": {str(k): v for k, v in predictions.items()},
        "baseline": round(baseline, 6) if isinstance(baseline, float) else baseline,
        "mae_model": round(mae_model, 6),
        "mae_baseline": round(mae_baseline, 6),
        "verdict": verdict
    }
    return result


# ---------- main ----------------------------------------------------------
def main(csv_path: str) -> None:
    rows, headers = read_csv(csv_path)
    if not rows:
        print("No data")
        return

    goal_cols = goal_columns(headers)

    # CAR-19
    print("== CAR-19 ==")
    res19 = car19(rows, goal_cols)
    # Print some details
    print(f"Budget: {res19['budget']}")
    print(f"Repeats: {res19['repeats']}")
    print(f"Ours best rows: {res19['ours'][:5]} ...")
    print(f"Random best rows: {res19['random'][:5]} ...")
    print(f"Verdict: {res19['verdict']}")
    print(f"ANSWER CAR-19 {json.dumps(res19, separators=(',', ':'))}")

    # CAR-20
    print("\n== CAR-20 ==")
    res20 = car20(rows, goal_cols)
    # Print some details
    print(f"Baseline MPG: {res20['baseline']}")
    print(f"MAE (model): {res20['mae_model']}")
    print(f"MAE (baseline): {res20['mae_baseline']}")
    print(f"Verdict: {res20['verdict']}")
    print(f"ANSWER CAR-20 {json.dumps(res20, separators=(',', ':'))}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
