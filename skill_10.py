#!/usr/bin/env python3
"""
skill_10.py: Evaluate estimation and selection performance on NASA data.
Usage: python skill_10.py <csv-file>
"""

import json
import random
from typing import Dict, List, Tuple

import ezr


# ------------------------------------------------------------------
# Helper: nearest‑neighbour effort predictor (used for NASA‑19)
# ------------------------------------------------------------------
def predict_effort_knn(
    tbl: ezr.TBL, target_row: ezr.ROW, training_rows: List[ezr.ROW], k: int = 3
) -> float:
    """Predict Effort- of target_row using k nearest neighbors in training_rows."""
    effort_col = next(c for c in tbl.cols.values() if c.txt == "Effort-")
    distances = []
    for row in training_rows:
        d = ezr.xdist(tbl, target_row, row)
        distances.append((d, row))
    distances.sort(key=lambda x: x[0])
    total = 0.0
    count = 0
    for _, row in distances:
        val = row[effort_col.at]
        if val == "?":
            continue
        total += float(val)
        count += 1
        if count >= k:
            break
    return total / count if count > 0 else 0.0


# ------------------------------------------------------------------
# NASA-19: Effort estimation comparison
# ------------------------------------------------------------------
def nasa_19(tbl: ezr.TBL) -> Tuple[Dict[str, float], float, float, float, float, str]:
    """Return predictions, baseline mean, mae_model, mae_baseline, and verdict."""
    # Held‑out indices (1‑based)
    held_out = [i + 1 for i, r in enumerate(tbl.rows) if (i + 1) % 5 == 0]
    # Build training set: all rows that are not held‑out
    training_rows = [r for i, r in enumerate(tbl.rows) if (i + 1) not in held_out]

    # Compute baseline mean Effort- over training set (ignore missing)
    effort_col = next(c for c in tbl.cols.values() if c.txt == "Effort-")
    training_efforts = [float(r[effort_col.at]) for r in training_rows if r[effort_col.at] != "?"]
    baseline_mean = sum(training_efforts) / len(training_efforts) if training_efforts else 0.0

    predictions: Dict[str, float] = {}
    mae_model = 0.0
    mae_baseline = 0.0
    count = 0
    for idx in held_out:
        target = tbl.rows[idx - 1]
        # Prediction via k‑NN on training rows
        pred = predict_effort_knn(tbl, target, training_rows, k=3)
        predictions[str(idx)] = pred
        actual = target[effort_col.at]
        if actual != "?":
            actual_f = float(actual)
            mae_model += abs(actual_f - pred)
            mae_baseline += abs(actual_f - baseline_mean)
            count += 1

    if count > 0:
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

    return predictions, baseline_mean, mae_model, mae_baseline, mae_model, verdict


# ------------------------------------------------------------------
# NASA-20: Selection comparison
# ------------------------------------------------------------------
def nasa_20(tbl: ezr.TBL, budget: int, repeats: int = 20) -> Tuple[List[int], List[int], str]:
    """Return best row numbers for ours and random in each repeat and verdict."""
    # Function to get best (min ydist) row from a list of indices
    def best_row(indices: List[int]) -> Tuple[int, float]:
        best_idx = indices[0]
        best_val = ezr.ydist(tbl, tbl.rows[best_idx])
        for i in indices[1:]:
            val = ezr.ydist(tbl, tbl.rows[i])
            if val < best_val:
                best_val = val
                best_idx = i
        return best_idx, best_val

    our_best_rows: List[int] = []
    our_best_ys: List[float] = []
    random_best_rows: List[int] = []
    random_best_ys: List[float] = []

    for rep in range(repeats):
        # Random selection
        rng = random.Random(rep)  # deterministic per repeat
        random_indices = rng.sample(range(len(tbl.rows)), budget)

        # Our selection: use ranker to rank all rows and pick top budget
        ranker_fn = ezr.ranker(tbl)
        sorted_indices = sorted(range(len(tbl.rows)), key=lambda i: ranker_fn(tbl.rows[i]))
        our_indices = sorted_indices[:budget]

        # Determine best rows
        our_idx, our_y = best_row(our_indices)
        rand_idx, rand_y = best_row(random_indices)

        our_best_rows.append(our_idx + 1)      # 1‑based
        our_best_ys.append(our_y)
        random_best_rows.append(rand_idx + 1)
        random_best_ys.append(rand_y)

    # Compute average best ydist
    avg_our = sum(our_best_ys) / len(our_best_ys) if our_best_ys else 0.0
    avg_rand = sum(random_best_ys) / len(random_best_ys) if random_best_ys else 0.0

    if avg_our < avg_rand:
        verdict = "ours better"
    elif avg_our > avg_rand:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    return our_best_rows, random_best_rows, verdict


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    import sys

    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv-file>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = list(ezr.csv(csv_path))
    tbl = ezr.Tbl(rows)

    # NASA-19
    print("== NASA-19 ==")
    preds, baseline, mae_model, mae_baseline, _, verdict19 = nasa_19(tbl)
    print(f"Baseline Effort- mean: {baseline:.3f}")
    print(f"MAE model: {mae_model:.3f}")
    print(f"MAE baseline: {mae_baseline:.3f}")
    print("Predictions for held‑out rows:")
    for r in sorted(preds, key=int)[:3]:
        print(f"  Row {r}: {preds[r]:.3f}")
    print(f"ANSWER NASA-19 {json.dumps({'predictions': preds, 'baseline': round(baseline,3), 'mae_model': round(mae_model,3), 'mae_baseline': round(mae_baseline,3), 'verdict': verdict19})}")

    # NASA-20
    budget = min(30, len(tbl.rows) // 5)
    repeats = 20
    ours_rows, rand_rows, verdict20 = nasa_20(tbl, budget, repeats)
    print("\n== NASA-20 ==")
    print(f"Budget: {budget}, Repeats: {repeats}")
    print(f"Best row per repeat (ours): {ours_rows[:3]}")
    print(f"Best row per repeat (random): {rand_rows[:3]}")
    print(f"ANSWER NASA-20 {json.dumps({'budget': budget, 'repeats': repeats, 'ours': ours_rows, 'random': rand_rows, 'verdict': verdict20})}")

if __name__ == "__main__":
    main()
