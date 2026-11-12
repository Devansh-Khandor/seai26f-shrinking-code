#!/usr/bin/env python3
"""
Skill 10 – Comparative Evaluation of Vehicle Selection and Performance Estimation

This script implements two new diagnostic components for the
used‑car dealership data:

* CAR-19 – Compare the composite‑score‑based vehicle selection
  strategy against a random selection under the same evaluation budget.

* CAR-20 – Assess the accuracy of the weighted‑nearest‑neighbour
  weight estimation and the MPG‑by‑model estimation against simple
  historical‑average baselines, using leave‑one‑out cross‑validation.
"""

import csv
import math
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def read_csv(path: Path) -> List[Dict[str, str]]:
    """Read CSV file into a list of dictionaries."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def safe_float(val: str) -> float | None:
    """Return float or None if conversion fails."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ----------------------------------------------------------------------
# Composite score helpers (used for CAR‑19)
# ----------------------------------------------------------------------
def composite_bounds(rows: List[Dict[str, str]]) -> Tuple[List[float], List[float], List[float]]:
    """Return lists of weight, mpg, acc values (floats) for computing bounds."""
    weights = [safe_float(r["Lbs-"]) for r in rows if safe_float(r["Lbs-"]) is not None]
    mpgs = [safe_float(r["Mpg+"]) for r in rows if safe_float(r["Mpg+"]) is not None]
    accs = [safe_float(r["Acc+"]) for r in rows if safe_float(r["Acc+"]) is not None]
    return weights, mpgs, accs


def composite_score(row: Dict[str, str],
                    w_min: float, w_max: float,
                    mpg_min: float, mpg_max: float,
                    acc_min: float, acc_max: float) -> float | None:
    """Return the composite score for a single row."""
    w = safe_float(row["Lbs-"])
    m = safe_float(row["Mpg+"])
    a = safe_float(row["Acc+"])
    if None in (w, m, a):
        return None

    def norm(val: float, low: float, high: float,
             higher_is_better: bool = True) -> float:
        if high == low:
            return 0.5
        return (val - low) / (high - low) if higher_is_better else (high - val) / (high - low)

    w_score = norm(w, w_min, w_max, higher_is_better=False)
    m_score = norm(m, mpg_min, mpg_max)
    a_score = norm(a, acc_min, acc_max)
    return (w_score + m_score + a_score) / 3


def scores_for_rows(rows: List[Dict[str, str]]) -> List[float]:
    """Compute composite scores for all rows (None if any missing)."""
    weights, mpgs, accs = composite_bounds(rows)
    w_min, w_max = min(weights), max(weights)
    mpg_min, mpg_max = min(mpgs), max(mpgs)
    acc_min, acc_max = min(accs), max(accs)

    scores = []
    for r in rows:
        sc = composite_score(r, w_min, w_max, mpg_min, mpg_max, acc_min, acc_max)
        scores.append(sc)
    return scores


# ----------------------------------------------------------------------
# Weight estimation helpers (CAR‑20)
# ----------------------------------------------------------------------
def compute_feature_stats(rows: List[Dict[str, str]], features: List[str]) -> Dict[str, Tuple[float, float]]:
    """Compute mean and variance for each feature.  Variance is population var."""
    stats = {}
    for name in features:
        vals = [safe_float(r[name]) for r in rows if safe_float(r[name]) is not None]
        if vals:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            stats[name] = (mean, var if var > 0 else 1.0)
        else:
            stats[name] = (0.0, 1.0)
    return stats


def estimate_weight_single(idx: int,
                          rows: List[Dict[str, str]],
                          stats: Dict[str, Tuple[float, float]],
                          feature_names: List[str]) -> float | None:
    """Weighted‑nearest‑neighbour weight estimate for a single row (excluding it)."""
    feats_i = [safe_float(rows[idx][name]) for name in feature_names]
    if None in feats_i:
        return None

    w_total = 0.0
    w_weighted = 0.0

    for j, r in enumerate(rows):
        if j == idx:
            continue
        feats_j = [safe_float(r[name]) for name in feature_names]
        if None in feats_j:
            continue

        # Squared, variance‑normalised distance
        dist_sq = sum(((feats_i[k] - feats_j[k]) ** 2) / stats[feature_names[k]][1]
                      for k in range(len(feature_names)))
        dist = math.sqrt(dist_sq)

        # Weight (inverse distance)
        w = 1.0 / (dist + 1e-6)

        w_target = safe_float(r["Lbs-"])
        if w_target is None:
            continue

        w_total += w
        w_weighted += w * w_target

    if w_total == 0.0:
        return None
    return w_weighted / w_total


# ----------------------------------------------------------------------
# MPG estimation helpers (CAR‑20)
# ----------------------------------------------------------------------
def estimate_mpg_single(idx: int,
                        rows: List[Dict[str, str]],
                        model_to_mpgs: Dict[str, List[float]]) -> float | None:
    """Estimate MPG for a single row (excluding it)."""
    r = rows[idx]
    model = r["Model"]
    # Build model-specific list excluding this row
    mpg_list = [mpg for j, mpg in model_to_mpgs.get(model, []) if j != idx]
    if mpg_list:
        return sum(mpg_list) / len(mpg_list)
    # Fallback: overall mean
    all_mpgs = [mpg for lst in model_to_mpgs.values() for mpg in lst]
    if not all_mpgs:
        return None
    return sum(all_mpgs) / len(all_mpgs)


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: File '{csv_path}' does not exist.")
        sys.exit(1)

    rows = read_csv(csv_path)
    if not rows:
        print("Error: CSV file is empty or has no data rows.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # CAR‑19 – Selection vs random
    # ------------------------------------------------------------------
    print("== CAR-19 ==")

    # Compute all composite scores
    all_scores = scores_for_rows(rows)
    # Filter out rows with missing composite score
    valid_indices = [i for i, s in enumerate(all_scores) if s is not None]
    valid_scores = [all_scores[i] for i in valid_indices]
    if not valid_scores:
        print("  No valid rows with complete data to compute scores.")
    else:
        # Determine evaluation budget (top N)
        N = min(5, len(valid_scores))

        # Best‑score selection
        top_indices = sorted(valid_indices, key=lambda i: all_scores[i], reverse=True)[:N]
        best_avg = sum(all_scores[i] for i in top_indices) / N

        # Random selection (fixed seed for reproducibility)
        random.seed(0)
        random_indices = random.sample(valid_indices, N)
        random_avg = sum(all_scores[i] for i in random_indices) / N

        improvement = best_avg - random_avg
        improvement_pct = (improvement / random_avg * 100) if random_avg != 0 else float("inf")

        print(f"  Evaluation budget: {N} vehicles")
        print(f"  Best‑score average: {best_avg:7.4f}")
        print(f"  Random average:    {random_avg:7.4f}")
        print(f"  Improvement:       {improvement:7.4f} ({improvement_pct:.1f}%)")

    # ------------------------------------------------------------------
    # CAR‑20 – Estimation accuracy
    # ------------------------------------------------------------------
    print("\n== CAR-20 ==")

    n_rows = len(rows)

    # ------------------------------------------------------------------
    # Weight estimation
    # ------------------------------------------------------------------
    feature_names = ["HpX", "Volume", "Clndrs", "origin", "Acc+"]
    stats = compute_feature_stats(rows, feature_names)

    weight_errors = []
    baseline_errors = []

    for i in range(n_rows):
        actual = safe_float(rows[i]["Lbs-"])
        if actual is None:
            continue

        # Leave‑one‑out estimate
        est = estimate_weight_single(i, rows, stats, feature_names)
        if est is not None:
            weight_errors.append(abs(est - actual))

        # Baseline: overall mean weight
        baseline_errors.append(abs(stat_weight[0] - actual))

    def mean(lst: List[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    # Overall mean weight (used for baseline)
    stat_weight = compute_feature_stats(rows, ["Lbs-"])["Lbs-"]

    mae_weight_est = mean(weight_errors)
    mae_weight_base = mean(baseline_errors)

    print("  Weight estimation:")
    print(f"    MAE (weighted NN):   {mae_weight_est:6.2f} lbs")
    print(f"    MAE (historical mean): {mae_weight_base:6.2f} lbs")

    # ------------------------------------------------------------------
    # MPG estimation
    # ------------------------------------------------------------------
    # Build model→list of (row_index, mpg)
    model_to_mpgs: Dict[str, List[Tuple[int, float]]] = {}
    for i, r in enumerate(rows):
        mpg = safe_float(r["Mpg+"])
        if mpg is None:
            continue
        model_to_mpgs.setdefault(r["Model"], []).append((i, mpg))

    mpg_errors = []
    baseline_errors_mpg = []

    for i in range(n_rows):
        actual = safe_float(rows[i]["Mpg+"])
        if actual is None:
            continue

        # Leave‑one‑out estimate
        est = estimate_mpg_single(i, rows, model_to_mpgs)
        if est is not None:
            mpg_errors.append(abs(est - actual))

        # Baseline: overall mean MPG
        baseline_errors_mpg.append(abs(stat_mpg[0] - actual))

    stat_mpg = compute_feature_stats(rows, ["Mpg+"])["Mpg+"]
    mae_mpg_est = mean(mpg_errors)
    mae_mpg_base = mean(baseline_errors_mpg)

    print("  MPG estimation:")
    print(f"    MAE (model‑avg NN):   {mae_mpg_est:6.2f} mpg")
    print(f"    MAE (historical mean): {mae_mpg_base:6.2f} mpg")


if __name__ == "__main__":
    main()
