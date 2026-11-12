#!/usr/bin/env python3
"""
Small data‑analysis skill for NASA project data – new NASA‑13 and NASA‑14.

Usage:
    python skill_07.py <csv_path>
"""

import csv
import sys
import statistics
import math
from typing import List, Dict, Tuple

# Import helpers from previous skills
from skill_01 import (
    load_data,
    parse_float,
    extract_numeric_fields,
    compute_stats,
)
from skill_02 import linear_regression, predict


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def numeric_fields() -> set:
    """Return the set of numeric field names used in the dataset."""
    return {"Kloc+", "Effort-", "Defects-", "MonthsX"}


def categorical_fields(records: List[Dict[str, str]]) -> List[str]:
    """Return a list of categorical field names (excluding idX and numeric fields)."""
    keys = records[0].keys()
    return [k for k in keys if k not in numeric_fields() and k != "idX"]


def group_by_combination(
    records: List[Dict[str, str]],
    cat_fields: List[str],
) -> Dict[Tuple[str, ...], Dict[str, List[float]]]:
    """
    Group records by the values of all categorical fields.
    Returns a mapping from a tuple of categorical values to a dict of numeric lists.
    """
    groups: Dict[Tuple[str, ...], Dict[str, List[float]]] = {}
    for r in records:
        key = tuple(r[f] for f in cat_fields)
        groups.setdefault(key, {"Kloc+": [], "Effort-": [], "Defects-": []})
        groups[key]["Kloc+"].append(parse_float(r["Kloc+"]))
        groups[key]["Effort-"].append(parse_float(r["Effort-"]))
        groups[key]["Defects-"].append(parse_float(r["Defects-"]))
    return groups


def z_score(val: float, mean: float, std: float) -> float:
    """Compute z‑score, guarding against zero standard deviation."""
    return (val - mean) / std if std > 0 else 0.0


# --------------------------------------------------------------------------- #
# Main routine: NASA‑13 & NASA‑14
# --------------------------------------------------------------------------- #
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    records = load_data(csv_path)

    # -----------------------------------------------------------------------
    # NASA‑13 – promising combinations of project characteristics
    # -----------------------------------------------------------------------
    print("== NASA-13 ==")

    cat_fields = categorical_fields(records)
    groups = group_by_combination(records, cat_fields)

    # Compute mean statistics for each group
    group_stats: List[Tuple[Tuple[str, ...], float, float, float]] = []
    for key, nums in groups.items():
        k_mean = statistics.mean(nums["Kloc+"])
        e_mean = statistics.mean(nums["Effort-"])
        d_mean = statistics.mean(nums["Defects-"])
        group_stats.append((key, k_mean, e_mean, d_mean))

    # Sort by: high Kloc+, low effort, low defects
    group_stats.sort(key=lambda x: (-x[1], x[2], x[3]))

    # Show top three combinations
    if group_stats:
        print("Top 3 characteristic combinations (by size, effort, defects):")
        for i, (key, k_mean, e_mean, d_mean) in enumerate(group_stats[:3], start=1):
            combo_str = ", ".join(f"{f}={v}" for f, v in zip(cat_fields, key))
            print(f"  {i}. {combo_str}")
            print(f"     mean Kloc+ = {k_mean:.1f}, mean Effort- = {e_mean:.1f}, mean Defects- = {d_mean:.1f}")
    else:
        print("No categorical combinations found.")

    # -----------------------------------------------------------------------
    # NASA‑14 – prioritize proposals for evaluation
    # -----------------------------------------------------------------------
    print("\n== NASA-14 ==")

    # Extract numeric lists for regression
    numeric = extract_numeric_fields(records)

    # Build regression models
    slope_ek, intercept_ek, _ = linear_regression(numeric["Kloc+"], numeric["Effort-"])
    slope_de, intercept_de, _ = linear_regression(numeric["Effort-"], numeric["Defects-"])

    # Predicted effort and defects for each project
    predicted: List[Dict[str, float]] = []
    for r in records:
        pid = int(r["idX"])
        size = parse_float(r["Kloc+"])
        pred_eff = predict(slope_ek, intercept_ek, size)
        pred_def = predict(slope_de, intercept_de, pred_eff)
        predicted.append(
            {
                "id": pid,
                "size": size,
                "pred_eff": pred_eff,
                "pred_def": pred_def,
            }
        )

    # Compute z‑scores across the predicted values
    eff_vals = [p["pred_eff"] for p in predicted]
    def_vals = [p["pred_def"] for p in predicted]
    size_vals = [p["size"] for p in predicted]

    eff_mean, eff_std = compute_stats(eff_vals)[:2]
    def_mean, def_std = compute_stats(def_vals)[:2]
    size_mean, size_std = compute_stats(size_vals)[:2]

    # Score = (size z) - (eff z) - (def z)
    for p in predicted:
        p["z_size"] = z_score(p["size"], size_mean, size_std)
        p["z_eff"] = z_score(p["pred_eff"], eff_mean, eff_std)
        p["z_def"] = z_score(p["pred_def"], def_mean, def_std)
        p["score"] = p["z_size"] - p["z_eff"] - p["z_def"]

    # Sort by score descending
    predicted.sort(key=lambda p: p["score"], reverse=True)

    # Show top 5 proposals to evaluate
    top_n = 5
    print(f"Top {top_n} proposals to evaluate first (promising size, low effort, low defects):")
    for i, p in enumerate(predicted[:top_n], start=1):
        print(
            f"  {i}. Project {p['id']}: size={p['size']:.1f} KLOC, "
            f"predicted effort={p['pred_eff']:.1f} h, predicted defects={p['pred_def']:.1f}, "
            f"score={p['score']:.3f}"
        )


if __name__ == "__main__":
    main()
