#!/usr/bin/env python3
"""
Skill 10: NASA-19 and NASA-20

NASA-19 – Compare the custom effort estimation system (k‑NN) against a
historical‑average baseline on held‑out rows.

NASA-20 – Compare a ranking‑based proposal selection (using predicted
effort and defects) against random selection under the same evaluation
budget.

The script prints a human‑readable section for each request followed by a
machine‑checked JSON line.
"""

import csv
import json
import math
import random
import sys
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# Basic utilities
# ----------------------------------------------------------------------
def atom(s: str) -> Any:
    """Coerce a CSV cell to a Python value."""
    s = s.strip()
    if s == "?":
        return s
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return {"True": True, "False": False}.get(s, s)

# ----------------------------------------------------------------------
# Load CSV
# ----------------------------------------------------------------------
def load_csv(path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [dict(zip(header, [atom(v) for v in r])) for r in reader]
    return header, rows

# ----------------------------------------------------------------------
# Column classification
# ----------------------------------------------------------------------
def is_goal(name: str) -> bool:
    return name.endswith("+") or name.endswith("-")

def is_x(name: str) -> bool:
    return name.endswith("X")

# ----------------------------------------------------------------------
# Feature columns for distance calculation
# ----------------------------------------------------------------------
def feature_names(header: List[str]) -> List[str]:
    return [c for c in header if not is_goal(c) and not is_x(c)]

# ----------------------------------------------------------------------
# Distances
# ----------------------------------------------------------------------
def numeric_column(col: str, rows: List[Dict[str, Any]]) -> bool:
    return any(isinstance(r[col], (int, float)) for r in rows if r[col] != "?")

def distance(r1: Dict[str, Any], r2: Dict[str, Any],
             feats: List[str]) -> float:
    s = 0.0
    cnt = 0
    for f in feats:
        v1, v2 = r1[f], r2[f]
        if v1 == "?" or v2 == "?" or (isinstance(v1, str) and isinstance(v2, str) and v1 == v2):
            continue
        if isinstance(v1, str) or isinstance(v2, str):
            s += 1.0
            cnt += 1
        else:
            try:
                diff = float(v1) - float(v2)
                s += diff * diff
                cnt += 1
            except Exception:
                s += 1.0
                cnt += 1
    return math.sqrt(s) if cnt > 0 else float("inf")

# ----------------------------------------------------------------------
# k‑NN prediction
# ----------------------------------------------------------------------
def knn_predict(target_row: Dict[str, Any], rows: List[Dict[str, Any]],
                exclude: set, target: str, k: int = 3, feats: List[str] = None) -> float:
    if feats is None:
        feats = feature_names([c for c in target_row])
    dists = []
    for r in rows:
        if id(r) in exclude:
            continue
        d = distance(target_row, r, feats)
        dists.append((d, r))
    dists.sort(key=lambda x: x[0])
    neigh_vals = []
    for d, r in dists[:k]:
        val = r[target]
        if val != "?":
            neigh_vals.append(val)
    if neigh_vals:
        return sum(neigh_vals) / len(neigh_vals)
    # Fallback to global mean of target
    vals = [r[target] for r in rows if r[target] != "?"]
    return sum(vals) / len(vals) if vals else 0.0

# ----------------------------------------------------------------------
# Composite objective (higher better)
# ----------------------------------------------------------------------
def composite_score(row: Dict[str, Any]) -> float:
    k = row["Kloc+"]
    e = row["Effort-"]
    d = row["Defects-"]
    return float(k) - float(e) - float(d)

# ----------------------------------------------------------------------
# NASA‑19
# ----------------------------------------------------------------------
def nasa19(header: List[str], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Identify held‑out rows (index divisible by 5)
    held_out_ids = [i + 1 for i in range(len(rows)) if (i + 1) % 5 == 0]
    held_out = [rows[i] for i in range(len(rows)) if (i + 1) % 5 == 0]
    training = [rows[i] for i in range(len(rows)) if (i + 1) % 5 != 0]

    # Train: use kNN predictions on held‑out rows
    feats = feature_names(header)
    predictions = {}
    mae_sum = 0.0
    n = 0
    for idx, r in zip(held_out_ids, held_out):
        pred = knn_predict(r, training, set(), "Effort-", k=3, feats=feats)
        predictions[str(idx)] = pred
        mae_sum += abs(pred - r["Effort-"])
        n += 1
    mae_model = mae_sum / n if n else 0.0

    # Baseline: mean Effort- of training rows
    baseline_mean = sum(r["Effort-"] for r in training) / len(training)
    mae_baseline = sum(abs(baseline_mean - r["Effort-"]) for r in held_out) / n if n else 0.0

    # Verdict
    if mae_model < mae_baseline - 1e-8:
        verdict = "better"
    elif mae_model > mae_baseline + 1e-8:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    return {
        "predictions": predictions,
        "baseline": baseline_mean,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict
    }

# ----------------------------------------------------------------------
# NASA‑20
# ----------------------------------------------------------------------
def nasa20(header: List[str], rows: List[Dict[str, Any]],
           repeats: int = 20, budget: int = 18) -> Dict[str, Any]:
    feats = feature_names(header)
    # Pre‑compute predicted effort and defects for each row using leave‑one‑out
    preds_effort = {}
    preds_defects = {}
    for idx, r in enumerate(rows):
        exclude = {id(r)}
        pred_e = knn_predict(r, rows, exclude, "Effort-", k=3, feats=feats)
        pred_d = knn_predict(r, rows, exclude, "Defects-", k=3, feats=feats)
        preds_effort[id(r)] = pred_e
        preds_defects[id(r)] = pred_d

    # Ranking score based on predictions
    def ranking_score(r):
        return float(r["Kloc+"]) - preds_effort[id(r)] - preds_defects[id(r)]

    # Ours selection: top budget rows by ranking_score
    sorted_rows = sorted(rows, key=ranking_score, reverse=True)
    ours_select = sorted_rows[:budget]
    ours_best = max(ours_select, key=composite_score)
    ours_best_id = rows.index(ours_best) + 1

    # Random selection
    random_best_ids = []
    ours_best_ids = []

    rng = random.Random(12345)
    for _ in range(repeats):
        rand_sel = rng.sample(rows, budget)
        rand_best = max(rand_sel, key=composite_score)
        random_best_ids.append(rows.index(rand_best) + 1)

        ours_best_ids.append(ours_best_id)  # same each repeat

    # Compare counts
    ours_wins = sum(1 for rb, ob in zip(random_best_ids, ours_best_ids) if composite_score(rows[ob-1]) > composite_score(rows[rb-1]))
    random_wins = repeats - ours_wins
    if ours_wins > random_wins:
        verdict = "ours better"
    elif random_wins > ours_wins:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    return {
        "budget": budget,
        "repeats": repeats,
        "ours": ours_best_ids,
        "random": random_best_ids,
        "verdict": verdict
    }

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = load_csv(csv_path)

    # NASA-19
    res19 = nasa19(header, rows)
    print("== NASA-19 ==")
    print(f"Baseline mean Effort-: {res19['baseline']:.6f}")
    print(f"Model MAE: {res19['mae_model']:.6f}")
    print(f"Baseline MAE: {res19['mae_baseline']:.6f}")
    print(f"Verdict: {res19['verdict']}")
    print(f"ANSWER NASA-19 {json.dumps(res19)}")

    # NASA-20
    res20 = nasa20(header, rows, repeats=20, budget=18)
    print("\n== NASA-20 ==")
    print(f"Budget: {res20['budget']}")
    print(f"Repeats: {res20['repeats']}")
    print(f"Ours best rows each repeat: {res20['ours']}")
    print(f"Random best rows each repeat: {res20['random']}")
    print(f"Verdict: {res20['verdict']}")
    print(f"ANSWER NASA-20 {json.dumps(res20)}")

if __name__ == "__main__":
    main()
