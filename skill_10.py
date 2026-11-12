#!/usr/bin/env python3
"""
Small data‑analysis skill for NASA project data – NASA‑19 and NASA‑20.

Usage:
    python skill_10.py <csv_path>
"""

import csv
import sys
import math
import random
from typing import List, Dict, Tuple

# --------------------------------------------------------------------------- #
# Re‑use helpers from previous skills
# --------------------------------------------------------------------------- #
from skill_01 import (
    load_data,
    parse_float,
    compute_stats,
)
from skill_02 import linear_regression, predict
from skill_04 import group_numeric, ci95, pairwise_no_overlap

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def _extract_numeric_fields(records: List[Dict[str, str]]) -> Dict[str, List[float]]:
    """Return mapping of numeric columns to list of values."""
    numeric = {
        "Kloc+": [],
        "Effort-": [],
        "Defects-": [],
        "MonthsX": [],
    }
    for r in records:
        numeric["Kloc+"].append(parse_float(r["Kloc+"]))
        numeric["Effort-"].append(parse_float(r["Effort-"]))
        numeric["Defects-"].append(parse_float(r["Defects-"]))
        numeric["MonthsX"].append(parse_float(r["MonthsX"]))
    return numeric


def _cross_validation_mae(records: List[Dict[str, str]]) -> Tuple[float, float]:
    """
    Perform leave‑one‑out cross‑validation on the Effort~Size regression.
    Returns (MAE_regression, MAE_average).
    """
    n = len(records)
    mae_reg = 0.0
    mae_avg = 0.0

    for i in range(n):
        # Train on all but record i
        train = records[:i] + records[i + 1 :]
        train_numeric = _extract_numeric_fields(train)
        slope, intercept, _ = linear_regression(
            train_numeric["Kloc+"], train_numeric["Effort-"]
        )
        # Prediction for left‑out
        size_i = parse_float(records[i]["Kloc+"])
        effort_i = parse_float(records[i]["Effort-"])
        if math.isnan(effort_i):
            continue  # skip missing
        pred_eff = predict(slope, intercept, size_i)
        mae_reg += abs(pred_eff - effort_i)

        # Average estimate from training set
        train_efforts = train_numeric["Effort-"]
        avg_eff = sum(train_efforts) / len(train_efforts)
        mae_avg += abs(avg_eff - effort_i)

    mae_reg /= n
    mae_avg /= n
    return mae_reg, mae_avg


def _select_by_score(
    records: List[Dict[str, str]],
    slope_ek: float,
    intercept_ek: float,
    slope_de: float,
    intercept_de: float,
    k: int,
) -> List[Dict[str, str]]:
    """
    Return the top‑k projects according to the scoring scheme:
    score = z_size - z_eff - z_def
    using predicted effort and defects.
    """
    predicted = []
    for r in records:
        size = parse_float(r["Kloc+"])
        pred_eff = predict(slope_ek, intercept_ek, size)
        pred_def = predict(slope_de, intercept_de, pred_eff)
        predicted.append(
            {
                "id": int(r["idX"]),
                "size": size,
                "eff_actual": parse_float(r["Effort-"]),
                "def_actual": parse_float(r["Defects-"]),
                "pred_eff": pred_eff,
                "pred_def": pred_def,
            }
        )

    # Compute z‑scores across predicted values
    eff_vals = [p["pred_eff"] for p in predicted]
    def_vals = [p["pred_def"] for p in predicted]
    size_vals = [p["size"] for p in predicted]

    eff_mean, eff_std = compute_stats(eff_vals)[:2]
    def_mean, def_std = compute_stats(def_vals)[:2]
    size_mean, size_std = compute_stats(size_vals)[:2]

    def z(v, m, s):
        return (v - m) / s if s > 0 else 0.0

    for p in predicted:
        p["z_size"] = z(p["size"], size_mean, size_std)
        p["z_eff"] = z(p["pred_eff"], eff_mean, eff_std)
        p["z_def"] = z(p["pred_def"], def_mean, def_std)
        p["score"] = p["z_size"] - p["z_eff"] - p["z_def"]

    predicted.sort(key=lambda p: p["score"], reverse=True)
    return predicted[:k]


def _random_selection_mean(
    records: List[Dict[str, str]],
    k: int,
    trials: int = 200,
    seed: int = 42,
) -> Tuple[float, float]:
    """Return average actual effort and defects for random selections."""
    random.seed(seed)
    efforts = []
    defects = []
    n = len(records)
    idxs = list(range(n))
    for _ in range(trials):
        sample = random.sample(idxs, k)
        eff_sum = sum(parse_float(records[i]["Effort-"]) for i in sample)
        def_sum = sum(parse_float(records[i]["Defects-"]) for i in sample)
        efforts.append(eff_sum / k)
        defects.append(def_sum / k)
    return sum(efforts) / trials, sum(defects) / trials


# --------------------------------------------------------------------------- #
# Main routine: NASA‑19 & NASA‑20
# --------------------------------------------------------------------------- #
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_10.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    records = load_data(csv_path)
    numeric = _extract_numeric_fields(records)

    # -----------------------------------------------------------------------
    # NASA‑19 – compare regression vs historical‑average estimate
    # -----------------------------------------------------------------------
    print("== NASA-19 ==")
    mae_reg, mae_avg = _cross_validation_mae(records)

    improvement = mae_avg - mae_reg
    print(f"Regression MAE on unseen projects: {mae_reg:.2f} hours")
    print(f"Historical‑average MAE:          {mae_avg:.2f} hours")
    print(
        f"Improvement (avg – reg):          {improvement:.2f} hours "
        f"({100 * improvement / mae_avg:.1f}% better)"
    )

    # -----------------------------------------------------------------------
    # NASA‑20 – compare selection approach vs random selection
    # -----------------------------------------------------------------------
    print("\n== NASA-20 ==")

    n = len(records)
    k = max(1, int(0.2 * n))  # evaluate 20% of projects

    # Fit regressions once
    slope_ek, intercept_ek, _ = linear_regression(
        numeric["Kloc+"], numeric["Effort-"]
    )
    slope_de, intercept_de, _ = linear_regression(
        numeric["Effort-"], numeric["Defects-"]
    )

    # Our systematic selection
    top_projects = _select_by_score(
        records, slope_ek, intercept_ek, slope_de, intercept_de, k
    )
    # Compute mean actual effort and defects for selected projects
    eff_sys = sum(p["eff_actual"] for p in top_projects) / k
    def_sys = sum(p["def_actual"] for p in top_projects) / k
    print(f"Systematic selection (top {k}):")
    print(f"  Mean actual effort:  {eff_sys:.2f} h")
    print(f"  Mean actual defects: {def_sys:.2f}")

    # Random selection baseline
    eff_rand, def_rand = _random_selection_mean(records, k)
    print(f"Random selection (average over trials):")
    print(f"  Mean actual effort:  {eff_rand:.2f} h")
    print(f"  Mean actual defects: {def_rand:.2f} h")

    # Summary
    eff_gain = eff_rand - eff_sys
    def_gain = def_rand - def_sys
    print(
        f"\nOur selection yields a reduction of {eff_gain:.2f} hours effort "
        f"({100 * eff_gain / eff_rand:.1f}% improvement) and "
        f"{def_gain:.2f} fewer defects "
        f"({100 * def_gain / def_rand:.1f}% improvement) compared to random."
    )


if __name__ == "__main__":
    main()
