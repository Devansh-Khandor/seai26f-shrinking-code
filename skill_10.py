#!/usr/bin/env python3
"""
skill_10.py - compare estimation and selection systems

Usage:  python skill_10.py <csv path>
"""

import sys
import json
import random
import ezr

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def read_csv(path):
    """Return header and list of rows."""
    rows = list(ezr.csv(path))
    header = rows[0]
    data = rows[1:]
    return header, data


# Goal utilities (copied from skill_07)
def get_goal_ranges(header, rows):
    """Return dict of goal column -> (min, max)."""
    goal_cols = [c for c in header if c.endswith(('+', '-'))]
    ranges = {}
    for c in goal_cols:
        idx = header.index(c)
        vals = [row[idx] for row in rows if row[idx] != "?"]
        if vals:
            mn = min(vals)
            mx = max(vals)
            ranges[c] = (mn, mx)
    return ranges


def normalize_value(col, val, mn, mx):
    """Return normalized 0..1 for a goal value."""
    if mx == mn:
        return 0.5
    if col.endswith('+'):
        return (val - mn) / (mx - mn)
    else:  # '-'
        return (mx - val) / (mx - mn)


def composite_score(header, ranges, row):
    """Return composite score based on all goal columns."""
    score = 0.0
    cnt = 0
    for c, (mn, mx) in ranges.items():
        v = row[header.index(c)]
        if v == "?":
            return None
        try:
            v = float(v)
        except ValueError:
            return None
        score += normalize_value(c, v, mn, mx)
        cnt += 1
    return score / cnt if cnt else None


# ----------------------------------------------------------------------
# NASA-19: model vs baseline MAE
# ----------------------------------------------------------------------
def nasa_19(header, data):
    # Identify held‑out indices (1‑based)
    held_out = [i + 1 for i in range(len(data)) if (i + 1) % 5 == 0]
    non_held = [row for i, row in enumerate(data, start=1) if i not in held_out]

    # Baseline mean Effort- of non‑held rows
    eff_idx = header.index("Effort-")
    eff_vals = [float(r[eff_idx]) for r in non_held if r[eff_idx] != "?"]
    baseline_mean = sum(eff_vals) / len(eff_vals) if eff_vals else 0.0

    # Tree model trained on non‑held rows
    tbl = ezr.Tbl([header] + non_held)
    model_node = ezr.tree(tbl, non_held)

    # Predictions for held‑out rows
    predictions = {}
    mae_model = 0.0
    mae_baseline = 0.0
    count = 0
    for idx in held_out:
        row = data[idx - 1]
        # actual effort
        actual = row[eff_idx]
        if actual == "?":
            pred = None
            base = None
        else:
            actual_f = float(actual)
            # model prediction
            leaf = ezr.leaf(model_node, row)
            pred_val = sum(r[eff_idx] for r in leaf.rows if r[eff_idx] != "?")
            n_vals = sum(1 for r in leaf.rows if r[eff_idx] != "?")
            pred = pred_val / n_vals if n_vals else None
            # baseline prediction
            base = baseline_mean

            mae_model += abs(pred - actual_f)
            mae_baseline += abs(base - actual_f)
            count += 1
        predictions[str(idx)] = round(pred, 4) if pred is not None else None

    mae_model = mae_model / count if count else 0.0
    mae_baseline = mae_baseline / count if count else 0.0

    # Verdict
    eps = 0.01
    if mae_model < mae_baseline - eps:
        verdict = "better"
    elif mae_model > mae_baseline + eps:
        verdict = "worse"
    else:
        verdict = "no clear difference"

    # Output
    print("== NASA-19 ==")
    print(f"Held‑out rows: {len(held_out)}")
    print(f"Baseline mean Effort-: {baseline_mean:.3f}")
    print(f"MAE (model): {mae_model:.3f}, MAE (baseline): {mae_baseline:.3f}")
    print(f"Verdict: {verdict}")

    answer = {
        "predictions": predictions,
        "baseline": baseline_mean,
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "verdict": verdict
    }
    print(f"ANSWER NASA-19 {json.dumps(answer, separators=(',', ':'))}")


# ----------------------------------------------------------------------
# NASA-20: selection vs random
# ----------------------------------------------------------------------
def nasa_20(header, data):
    # Settings
    repeats = 20
    budget = min(30, len(data) // 5)

    # Compute goal ranges once
    ranges = get_goal_ranges(header, data)

    # Composite score for each row (skip rows with missing goals)
    scores = {}
    for idx, row in enumerate(data, start=1):
        sc = composite_score(header, ranges, row)
        if sc is not None:
            scores[idx] = sc

    # Our ordering: top rows by composite score
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ours_order = [idx for idx, _ in ordered[:budget]]

    # Best row in our ordering
    best_ours = max(ours_order, key=lambda i: scores[i])

    # Random repeats
    random.seed(42)
    random_best = []
    for _ in range(repeats):
        chosen = random.sample(list(scores.keys()), k=budget)
        best_row = max(chosen, key=lambda i: scores[i])
        random_best.append(best_row)

    # Best row for random across repeats (most frequent)
    # but spec expects list of best rows each repeat
    # compute average composite score of best rows
    avg_ours = scores[best_ours]
    avg_random = sum(scores[i] for i in random_best) / repeats

    # Verdict
    eps = 0.01
    if avg_ours > avg_random + eps:
        verdict = "ours better"
    elif avg_ours < avg_random - eps:
        verdict = "random better"
    else:
        verdict = "no clear difference"

    # Output
    print("== NASA-20 ==")
    print(f"Budget: {budget}")
    print(f"Repeats: {repeats}")
    print(f"Our best row in each repeat: {[best_ours]*repeats}")
    print(f"Random best rows each repeat: {random_best}")
    print(f"Verdict: {verdict}")

    answer = {
        "budget": budget,
        "repeats": repeats,
        "ours": [best_ours] * repeats,
        "random": random_best,
        "verdict": verdict
    }
    print(f"ANSWER NASA-20 {json.dumps(answer, separators=(',', ':'))}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    header, data = read_csv(path)

    nasa_19(header, data)
    nasa_20(header, data)


if __name__ == "__main__":
    main()
