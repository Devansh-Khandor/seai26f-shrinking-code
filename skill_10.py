#!/usr/bin/env python3
"""
skill_10.py

Usage: python skill_10.py <csv path>
"""

import json
import random
import sys
from typing import Any, Dict, List, Tuple

import ezr

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def numeric(v: Any) -> float | None:
    """Return numeric value, or None if missing."""
    return None if v == "?" else v

def minmax(values: List[float]) -> Tuple[float, float]:
    """Return (min, max).  If all same, return (m, m+1)."""
    if not values:
        return 0.0, 1.0
    mn, mx = min(values), max(values)
    if mn == mx:
        mx = mn + 1.0
    return mn, mx

def normalize(v: float, mn: float, mx: float) -> float:
    """Map value to 0..1."""
    return 0.0 if mx == mn else (v - mn) / (mx - mn)

# ----------------------------------------------------------------------
# Composite score over all goals
# ----------------------------------------------------------------------
def compute_goal_minmax(header: List[str], rows: List[List[Any]]) -> Dict[str, Tuple[float, float]]:
    """Min/max for each goal column over all rows."""
    mins_maxs: Dict[str, Tuple[float, float]] = {}
    for i, name in enumerate(header):
        if name[-1] not in "+-":
            continue
        vals = [numeric(r[i]) for r in rows if numeric(r[i]) is not None]
        mn, mx = minmax(vals)
        mins_maxs[name] = (mn, mx)
    return mins_maxs

def composite_score(row: List[Any], header: List[str], goal_minmax: Dict[str, Tuple[float, float]]) -> float:
    """Mean of normalized goal values available in row."""
    norms: List[float] = []
    for i, name in enumerate(header):
        if name[-1] not in "+-":
            continue
        v = numeric(row[i])
        if v is None:
            continue
        mn, mx = goal_minmax[name]
        norm = (v - mn) / (mx - mn) if mx > mn else 0.0
        if name.endswith("-"):
            norm = 1.0 - norm
        norms.append(norm)
    return sum(norms) / len(norms) if norms else 0.0

# ----------------------------------------------------------------------
# Heuristic score for selecting rows to evaluate
# ----------------------------------------------------------------------
def heuristic_row_score(row: List[Any], header: List[str]) -> float:
    """Simple heuristic based on horsepower, volume and cylinders."""
    # Map column names to indices
    try:
        hp_idx = header.index("HpX")
        vol_idx = header.index("Volume")
        cyl_idx = header.index("Clndrs")
    except ValueError:
        # Missing columns: fallback to 0 score
        return 0.0

    hp = numeric(row[hp_idx])
    vol = numeric(row[vol_idx])
    cyl = numeric(row[cyl_idx])

    # Normalization helpers
    def norm(v: float, mn: float, mx: float) -> float:
        return normalize(v, mn, mx) if v is not None else 0.0

    # We'll use global min/max for these columns over all data
    # (will be set later when called)
    return (hp, vol, cyl)

# ----------------------------------------------------------------------
# KNN prediction for Mpg+
# ----------------------------------------------------------------------
def knn_predict_mpg(held_out_rows: List[Tuple[int, List[Any]]],
                    train_rows: List[Tuple[int, List[Any]]],
                    k: int = 3) -> Dict[int, float]:
    """Return predicted Mpg+ for each held-out row id."""
    header = held_out_rows[0][1] if held_out_rows else []
    # Build table for distance calculation (using all rows)
    tbl = ezr.Tbl([header] + [r for _, r in train_rows + held_out_rows])

    # Precompute goal index
    try:
        mpg_idx = header.index("Mpg+")
    except ValueError:
        return {}

    predictions: Dict[int, float] = {}
    # For each held-out
    for idx, row in held_out_rows:
        # compute distances to all training rows
        dists = []
        for tidx, tr in train_rows:
            d = ezr.xdist(tbl, row, tr)
            dists.append((d, tr))
        dists.sort(key=lambda x: x[0])
        neighbors = dists[:k]
        vals = [numeric(n[1][mpg_idx]) for n in neighbors if numeric(n[1][mpg_idx]) is not None]
        if vals:
            predictions[idx] = sum(vals) / len(vals)
    return predictions

# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv path>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Load data
    rows_iter = ezr.csv(csv_path)
    header = next(rows_iter)
    data_rows: List[List[Any]] = list(rows_iter)
    n_rows = len(data_rows)

    # Determine budget
    budget = min(30, n_rows // 5)

    # Compute goal min/max over all rows
    goal_minmax = compute_goal_minmax(header, data_rows)

    # ------------------------------------------------------------------
    # CAR-19: Selection vs Random
    # ------------------------------------------------------------------
    print("== CAR-19 ==")

    repeats = 20
    ours_best: List[int] = []
    random_best: List[int] = []

    # Precompute global min/max for heuristic columns
    hp_vals = [numeric(r[header.index("HpX")]) for r in data_rows if numeric(r[header.index("HpX")]) is not None]
    vol_vals = [numeric(r[header.index("Volume")]) for r in data_rows if numeric(r[header.index("Volume")]) is not None]
    cyl_vals = [numeric(r[header.index("Clndrs")]) for r in data_rows if numeric(r[header.index("Clndrs")]) is not None]
    hp_mn, hp_mx = minmax(hp_vals)
    vol_mn, vol_mx = minmax(vol_vals)
    cyl_mn, cyl_mx = minmax(cyl_vals)

    def row_heuristic(row: List[Any]) -> float:
        hp = numeric(row[header.index("HpX")])
        vol = numeric(row[header.index("Volume")])
        cyl = numeric(row[header.index("Clndrs")])
        hp_norm = normalize(hp, hp_mn, hp_mx) if hp is not None else 0.0
        vol_norm = normalize(vol, vol_mn, vol_mx) if vol is not None else 0.0
        cyl_norm = normalize(cyl, cyl_mn, cyl_mx) if cyl is not None else 0.0
        # prefer high hp, low vol, low cyl
        return 0.4 * hp_norm + 0.3 * (1 - vol_norm) + 0.3 * (1 - cyl_norm)

    # Run repeats
    rng = random.Random(ezr.the.Seed)
    for _ in range(repeats):
        # Ours: heuristic pick budget rows
        scored = [(row_heuristic(r), i) for i, r in enumerate(data_rows)]
        scored.sort(key=lambda x: -x[0])
        ours_indices = [idx for _, idx in scored[:budget]]
        # Evaluate them: get composite scores
        ours_scores = [composite_score(data_rows[i], header, goal_minmax) for i in ours_indices]
        best_i = ours_indices[ours_scores.index(max(ours_scores))]
        ours_best.append(best_i + 1)  # 1-based

        # Random: sample budget rows
        rnd_idx = rng.sample(range(n_rows), budget)
        rnd_scores = [composite_score(data_rows[i], header, goal_minmax) for i in rnd_idx]
        best_r = rnd_idx[rnd_scores.index(max(rnd_scores))]
        random_best.append(best_r + 1)

    # Verdict
    better_count = sum(1 for a, b in zip(ours_best, random_best)
                       if composite_score(data_rows[a-1], header, goal_minmax) >
                          composite_score(data_rows[b-1], header, goal_minmax))
    random_better = sum(1 for a, b in zip(ours_best, random_best)
                        if composite_score(data_rows[a-1], header, goal_minmax) <
                           composite_score(data_rows[b-1], header, goal_minmax))
    if better_count > random_better:
        verdict = "ours better"
    elif random_better > better_count:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    # Output machine‑checked answer
    ans_19 = {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best,
        "random": random_best,
        "verdict": verdict
    }
    print(f"ANSWER CAR-19 {json.dumps(ans_19, separators=(',', ':'))}")

    # ------------------------------------------------------------------
    # CAR-20: Prediction vs baseline
    # ------------------------------------------------------------------
    print("\n== CAR-20 ==")

    # Identify held‑out rows: numbers divisible by 5 (1‑based)
    held_out_indices = [i for i in range(n_rows) if (i+1) % 5 == 0]
    train_indices = [i for i in range(n_rows) if (i+1) % 5 != 0]

    held_out_rows = [(i+1, data_rows[i]) for i in held_out_indices]
    train_rows = [(i+1, data_rows[i]) for i in train_indices]

    # KNN predictions
    predictions = knn_predict_mpg(held_out_rows, train_rows, k=3)

    # Baseline mean of Mpg+ on training set
    mpg_vals_train = [numeric(data_rows[i][header.index("Mpg+")])
                      for i in train_indices
                      if numeric(data_rows[i][header.index("Mpg+")]) is not None]
    baseline = sum(mpg_vals_train) / len(mpg_vals_train) if mpg_vals_train else 0.0

    # Compute MAE for model and baseline
    mae_model = 0.0
    mae_baseline = 0.0
    n_eval = 0
    for idx, row in held_out_rows:
        actual = numeric(row[header.index("Mpg+")])
        if actual is None:
            continue
        n_eval += 1
        pred = predictions.get(idx, baseline)
        mae_model += abs(pred - actual)
        mae_baseline += abs(baseline - actual)
    mae_model = mae_model / n_eval if n_eval else 0.0
    mae_baseline = mae_baseline / n_eval if n_eval else 0.0

    # Verdict
    if mae_model < mae_baseline:
        verdict2 = "better"
    elif mae_model > mae_baseline:
        verdict2 = "worse"
    else:
        verdict2 = "no clear difference"

    # Output machine‑checked answer
    ans_20 = {
        "predictions": predictions,
        "baseline": baseline,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict2
    }
    print(f"ANSWER CAR-20 {json.dumps(ans_20, separators=(',', ':'))}")

if __name__ == "__main__":
    main()
