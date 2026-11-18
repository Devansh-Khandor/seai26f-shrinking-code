#!/usr/bin/env python3
"""
skill_10.py: Implements CAR-19 and CAR-20 for used‑car dealership data.

Usage:
    python skill_10.py <csv_path>
"""

import csv
import json
import math
import random
import sys
from typing import Any, Dict, List, Tuple

# ------------------------------------------------------------------
# CSV utilities
# ------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[str], List[List[Any]]]:
    """Read CSV, convert numeric cells, keep '?' as None."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows: List[List[Any]] = []
        for line in reader:
            conv: List[Any] = []
            for v in line:
                if v == "?":
                    conv.append(None)
                else:
                    try:
                        if "." in v or "e" in v.lower():
                            conv.append(float(v))
                        else:
                            conv.append(int(v))
                    except ValueError:
                        conv.append(v)
            rows.append(conv)
    return header, rows


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def is_goal(col: str) -> bool:
    return col.endswith(("+", "-"))


def goal_indices(header: List[str]) -> List[int]:
    return [i for i, h in enumerate(header) if is_goal(h)]


def feature_indices(header: List[str]) -> List[int]:
    return [
        i
        for i, h in enumerate(header)
        if not h.endswith(("X", "+", "-")) and not h.endswith("X")
    ]


def compute_min_max(rows: List[List[Any]], indices: List[int]) -> Tuple[Dict[int, float], Dict[int, float]]:
    mins: Dict[int, float] = {}
    maxs: Dict[int, float] = {}
    for idx in indices:
        vals = [r[idx] for r in rows if r[idx] is not None]
        if vals:
            mins[idx] = min(vals)
            maxs[idx] = max(vals)
        else:
            mins[idx] = maxs[idx] = None
    return mins, maxs


def normalize_value(v: Any, mn: float, mx: float) -> float:
    if v is None:
        return 0.5
    if mx == mn:
        return 0.5
    return (v - mn) / (mx - mn)


def overall_score(
    row: List[Any],
    goal_idx: List[int],
    mins: Dict[int, float],
    maxs: Dict[int, float],
    ismax: Dict[int, bool],
) -> float:
    score = 0.0
    for idx in goal_idx:
        v = row[idx]
        if v is None:
            return float("nan")  # incomplete row
        mn, mx = mins[idx], maxs[idx]
        if mn is None or mx is None:
            return float("nan")
        if ismax[idx]:
            norm = (v - mn) / (mx - mn) if mx != mn else 0.5
        else:
            norm = (mx - v) / (mx - mn) if mx != mn else 0.5
        score += norm
    return score


# ------------------------------------------------------------------
# Linear regression (closed form)
# ------------------------------------------------------------------
def matrix_inverse(A: List[List[float]]) -> List[List[float]]:
    n = len(A)
    M = [row[:] + [float(i == j) for i in range(n)] for j, row in enumerate(A)]
    for i in range(n):
        pivot = i
        while pivot < n and abs(M[pivot][i]) < 1e-12:
            pivot += 1
        if pivot == n:
            raise ValueError("Singular matrix")
        if pivot != i:
            M[i], M[pivot] = M[pivot], M[i]
        div = M[i][i]
        M[i] = [v / div for v in M[i]]
        for r in range(n):
            if r != i:
                factor = M[r][i]
                M[r] = [rv - factor * iv for rv, iv in zip(M[r], M[i])]
    return [row[n:] for row in M]


def mat_vec_mul(A: List[List[float]], v: List[float]) -> List[float]:
    res = [0.0 for _ in range(len(A))]
    for i, row in enumerate(A):
        s = 0.0
        for a, b in zip(row, v):
            s += a * b
        res[i] = s
    return res


def train_linear_regression(X: List[List[float]], y: List[float]) -> List[float]:
    n_features = len(X[0])
    XTX = [[0.0] * n_features for _ in range(n_features)]
    for xi in X:
        for i in range(n_features):
            for j in range(n_features):
                XTX[i][j] += xi[i] * xi[j]
    inv_XTX = matrix_inverse(XTX)
    XTy = [0.0] * n_features
    for xi, yi in zip(X, y):
        for i in range(n_features):
            XTy[i] += xi[i] * yi
    beta = mat_vec_mul(inv_XTX, XTy)
    return beta


def predict(beta: List[float], features: List[float]) -> float:
    return sum(b * f for b, f in zip(beta, features))


# ------------------------------------------------------------------
# CAR-19 implementation
# ------------------------------------------------------------------
def car19(header: List[str], rows: List[List[Any]]) -> Dict[str, Any]:
    n_rows = len(rows)
    budget = min(30, n_rows // 5)
    repeats = 20

    g_idx = goal_indices(header)
    ismax_goal = {i: header[i].endswith("+") for i in g_idx}

    # Compute goal min/max for normalization
    goal_mins, goal_maxs = compute_min_max(rows, g_idx)

    # Pre‑compute best overall row for reference
    best_overall = -float("inf")
    best_overall_row = None
    for i, r in enumerate(rows):
        sc = overall_score(r, g_idx, goal_mins, goal_maxs, ismax_goal)
        if math.isnan(sc):
            continue
        if sc > best_overall:
            best_overall = sc
            best_overall_row = i + 1

    # Feature statistics
    feat_idx = feature_indices(header)
    feat_mins, feat_maxs = compute_min_max(rows, feat_idx)

    def heuristic_score(r: List[Any]) -> float:
        s = 0.0
        for idx in feat_idx:
            v = r[idx]
            mn, mx = feat_mins[idx], feat_maxs[idx]
            if mn is None or mx is None:
                continue
            s += normalize_value(v, mn, mx)
        return s

    ours_best: List[int] = []
    random_best: List[int] = []

    # Prepare base random generator for reproducibility
    base_seed = 123456
    for rep in range(repeats):
        rng = random.Random(base_seed + rep)
        all_indices = list(range(n_rows))
        rng.shuffle(all_indices)

        # OURS: heuristic ranking
        ranked = sorted(all_indices, key=lambda i: heuristic_score(rows[i]), reverse=True)
        eval_indices_ours = ranked[:budget]
        # Determine best among evaluated
        best_sc = -float("inf")
        best_row = None
        for idx in eval_indices_ours:
            sc = overall_score(rows[idx], g_idx, goal_mins, goal_maxs, ismax_goal)
            if math.isnan(sc):
                continue
            if sc > best_sc:
                best_sc = sc
                best_row = idx + 1
        if best_row is None:
            best_row = eval_indices_ours[0] + 1
        ours_best.append(best_row)

        # RANDOM: first budget of shuffled
        eval_indices_rand = all_indices[:budget]
        best_sc = -float("inf")
        best_row = None
        for idx in eval_indices_rand:
            sc = overall_score(rows[idx], g_idx, goal_mins, goal_maxs, ismax_goal)
            if math.isnan(sc):
                continue
            if sc > best_sc:
                best_sc = sc
                best_row = idx + 1
        if best_row is None:
            best_row = eval_indices_rand[0] + 1
        random_best.append(best_row)

    # Determine verdict
    ours_wins = sum(1 for o, r in zip(ours_best, random_best) if o == best_overall_row)
    random_wins = sum(1 for o, r in zip(ours_best, random_best) if r == best_overall_row)
    ties = repeats - ours_wins - random_wins
    if ours_wins > random_wins:
        verdict = "ours better"
    elif random_wins > ours_wins:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    return {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best,
        "random": random_best,
        "verdict": verdict,
    }


# ------------------------------------------------------------------
# CAR-20 implementation
# ------------------------------------------------------------------
def car20(header: List[str], rows: List[List[Any]]) -> Dict[str, Any]:
    # Identify goal columns and specifically Mpg+
    g_idx = goal_indices(header)
    mpg_idx = None
    for i in g_idx:
        if header[i] == "Mpg+":
            mpg_idx = i
            break
    if mpg_idx is None:
        raise ValueError("No Mpg+ column")

    # Held‑out rows: 1‑based divisible by 5
    held_out = [i for i, r in enumerate(rows) if (i + 1) % 5 == 0]
    training_rows = [r for i, r in enumerate(rows) if i not in held_out]

    # Features: all non‑goal, non‑X columns
    feat_idx = feature_indices(header)

    # Build training X and y
    X: List[List[float]] = []
    y: List[float] = []
    for r in training_rows:
        if r[mpg_idx] is None:
            continue
        # check for missing feature
        skip = False
        vec = [1.0]
        for idx in feat_idx:
            v = r[idx]
            if v is None:
                skip = True
                break
            vec.append(float(v))
        if skip:
            continue
        X.append(vec)
        y.append(float(r[mpg_idx]))

    # Train linear regression
    beta = train_linear_regression(X, y) if X else [0.0]

    # Predict held‑out
    preds: Dict[str, float] = {}
    mae_model = 0.0
    mae_baseline = 0.0
    n_test = 0
    baseline_sum = 0.0
    baseline_cnt = 0
    for i in held_out:
        r = rows[i]
        # compute prediction
        vec = [1.0]
        for idx in feat_idx:
            v = r[idx]
            vec.append(float(v) if v is not None else 0.5)
        pred = predict(beta, vec)
        preds[str(i + 1)] = pred
        # real value
        real = float(r[mpg_idx]) if r[mpg_idx] is not None else 0.0
        mae_model += abs(pred - real)
        n_test += 1

    # baseline mean on training mpg
    for r in training_rows:
        if r[mpg_idx] is not None:
            baseline_sum += float(r[mpg_idx])
            baseline_cnt += 1
    baseline_mean = baseline_sum / baseline_cnt if baseline_cnt else 0.0

    # baseline mae
    for i in held_out:
        r = rows[i]
        real = float(r[mpg_idx]) if r[mpg_idx] is not None else 0.0
        mae_baseline += abs(baseline_mean - real)

    mae_model /= n_test if n_test else 1
    mae_baseline /= n_test if n_test else 1

    # Verdict
    if mae_model < mae_baseline:
        verdict = "better"
    elif mae_model > mae_baseline:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    return {
        "predictions": preds,
        "baseline": baseline_mean,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict,
    }


# ------------------------------------------------------------------
# Main driver
# ------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)

    path = sys.argv[1]
    header, rows = read_csv(path)

    # CAR-19
    print("== CAR-19 ==")
    res19 = car19(header, rows)
    print(f"Budget: {res19['budget']}")
    print(f"Repeats: {res19['repeats']}")
    print(f"Ours best rows: {res19['ours'][:5]}...")  # show first 5
    print(f"Random best rows: {res19['random'][:5]}...")
    ours_wins = sum(1 for o, r in zip(res19['ours'], res19['random']) if o == res19['budget'])
    random_wins = sum(1 for o, r in zip(res19['ours'], res19['random']) if r == res19['budget'])
    print(f"Verdict: {res19['verdict']}")
    print(f"ANSWER CAR-19 {json.dumps(res19)}")

    # CAR-20
    print("\n== CAR-20 ==")
    res20 = car20(header, rows)
    print(f"Baseline mean Mpg+: {res20['baseline']:.4f}")
    print(f"MAE Model: {res20['mae_model']:.4f}")
    print(f"MAE Baseline: {res20['mae_baseline']:.4f}")
    print(f"Predictions for first 5 held‑out rows: {list(res20['predictions'].items())[:5]}")
    print(f"Verdict: {res20['verdict']}")
    print(f"ANSWER CAR-20 {json.dumps(res20)}")


if __name__ == "__main__":
    main()
