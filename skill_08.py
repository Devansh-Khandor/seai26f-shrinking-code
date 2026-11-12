#!/usr/bin/env python3
"""
Small data‑analysis skill for NASA project data – new NASA‑15 and NASA‑16.

Usage:
    python skill_08.py <csv_path>
"""

import csv
import sys
import statistics
import math
from typing import List, Dict, Tuple

# --------------------------------------------------------------------------- #
# Re‑use helpers from previous skills
# --------------------------------------------------------------------------- #
from skill_01 import (
    load_data,
    parse_float,
    extract_numeric_fields,
    compute_stats,
)
from skill_02 import linear_regression, predict
from skill_04 import group_numeric, ci95, pairwise_no_overlap


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def _predicted_values(
    records: List[Dict[str, str]],
    slope_ek: float,
    intercept_ek: float,
    slope_de: float,
    intercept_de: float,
) -> List[Dict[str, float]]:
    """Return list of dicts containing predicted effort and defects for each record."""
    out = []
    for r in records:
        size = parse_float(r["Kloc+"])
        pred_eff = predict(slope_ek, intercept_ek, size)
        pred_def = predict(slope_de, intercept_de, pred_eff)
        out.append(
            {
                "id": int(r["idX"]),
                "size": size,
                "eff_actual": parse_float(r["Effort-"]),
                "pred_eff": pred_eff,
                "pred_def": pred_def,
                "acap": r.get("acap", "N/A"),
            }
        )
    return out


def _choose_project(
    preds: List[Dict[str, float]],
    threshold: float,
    key: str,
) -> Dict[str, float]:
    """Return the first project above threshold, or the one with the maximum value."""
    above = [p for p in preds if p[key] > threshold]
    if above:
        return above[0]
    # fallback
    return max(preds, key=lambda p: p[key])


def _size_reduction(
    size: float, percent: float, slope_ek: float, intercept_ek: float, slope_de: float, intercept_de: float
) -> Tuple[float, float]:
    """Return predicted effort and defects after reducing size by percent."""
    new_size = size * (1.0 - percent / 100.0)
    new_eff = predict(slope_ek, intercept_ek, new_size)
    new_def = predict(slope_de, intercept_de, new_eff)
    return new_eff, new_def


def _acap_effect(
    current_acap: str,
    acap_stats: Dict[str, Dict[str, float]],
) -> Tuple[str, float]:
    """
    Return the target acap level (lowest mean effort) and the expected reduction in effort
    relative to the current group. If current group already has lowest effort, keep it.
    """
    # Find group with lowest mean effort
    target_acap = min(acap_stats, key=lambda k: acap_stats[k]["mean"])
    reduction = acap_stats[current_acap]["mean"] - acap_stats[target_acap]["mean"]
    return target_acap, reduction


# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_08.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    records = load_data(csv_path)
    numeric = extract_numeric_fields(records)

    # Regression models
    slope_ek, intercept_ek, _ = linear_regression(numeric["Kloc+"], numeric["Effort-"])
    slope_de, intercept_de, _ = linear_regression(numeric["Effort-"], numeric["Defects-"])

    # Predicted values
    preds = _predicted_values(records, slope_ek, intercept_ek, slope_de, intercept_de)

    # Thresholds
    eff_mean, _, eff_sd = compute_stats(numeric["Effort-"])
    def_mean, _, def_sd = compute_stats(numeric["Defects-"])
    eff_thresh = eff_mean + 2 * eff_sd
    def_thresh = def_mean + 2 * def_sd

    # -----------------------------------------------------------------------
    # NASA-15 – effort reduction proposals
    # -----------------------------------------------------------------------
    print("== NASA-15 ==")

    high_eff_project = _choose_project(preds, eff_thresh, "pred_eff")
    print(f"High‑effort proposal: Project {high_eff_project['id']}")
    print(f"  Predicted effort: {high_eff_project['pred_eff']:.1f} h")
    print(f"  Predicted defects: {high_eff_project['pred_def']:.1f}")

    # Size reduction
    size_red_pct = 15  # percent
    new_eff, new_def = _size_reduction(
        high_eff_project["size"],
        size_red_pct,
        slope_ek,
        intercept_ek,
        slope_de,
        intercept_de,
    )
    print(f"\n  Reducing size by {size_red_pct}%:")
    print(f"    New predicted effort: {new_eff:.1f} h")
    print(f"    New predicted defects: {new_def:.1f}")

    # Acap upgrade
    acap_stats = group_numeric(records, "acap", "Effort-")
    current_acap = high_eff_project["acap"]
    target_acap, reduction = _acap_effect(current_acap, acap_stats)
    if reduction > 0:
        new_eff_acap = high_eff_project["pred_eff"] - reduction
        new_def_acap = predict(slope_de, intercept_de, new_eff_acap)
        print(f"\n  Upgrading programmer capability from {current_acap} to {target_acap}:")
        print(f"    Expected effort reduction: {reduction:.1f} h")
        print(f"    New predicted effort: {new_eff_acap:.1f} h")
        print(f"    New predicted defects: {new_def_acap:.1f}")
    else:
        print("\n  Current acap level already has the lowest mean effort; no improvement from upgrade.")

    # -----------------------------------------------------------------------
    # NASA-16 – defect improvement proposals
    # -----------------------------------------------------------------------
    print("\n== NASA-16 ==")

    high_def_project = _choose_project(preds, def_thresh, "pred_def")
    print(f"High‑defect proposal: Project {high_def_project['id']}")
    print(f"  Predicted effort: {high_def_project['pred_eff']:.1f} h")
    print(f"  Predicted defects: {high_def_project['pred_def']:.1f}")

    # Smallest plausible change: 5% size reduction
    size_red_pct_def = 5
    eff_after, def_after = _size_reduction(
        high_def_project["size"],
        size_red_pct_def,
        slope_ek,
        intercept_ek,
        slope_de,
        intercept_de,
    )
    print(f"\n  Reducing size by {size_red_pct_def}%:")
    print(f"    New predicted effort: {eff_after:.1f} h")
    print(f"    New predicted defects: {def_after:.1f}")

    # If further improvement needed, suggest acap upgrade
    if def_after > def_thresh:
        # Re‑compute target acap
        acap_stats_def = group_numeric(records, "acap", "Effort-")
        current_acap_def = high_def_project["acap"]
        target_acap_def, reduction_def = _acap_effect(current_acap_def, acap_stats_def)
        if reduction_def > 0:
            eff_after_acap = eff_after - reduction_def
            def_after_acap = predict(slope_de, intercept_de, eff_after_acap)
            print(f"\n  Additionally upgrading acap from {current_acap_def} to {target_acap_def}:")
            print(f"    Effort reduction: {reduction_def:.1f} h")
            print(f"    New predicted effort: {eff_after_acap:.1f} h")
            print(f"    New predicted defects: {def_after_acap:.1f}")
        else:
            print("\n  Current acap level already has the lowest mean effort; no improvement from upgrade.")
    else:
        print("\n  Size reduction alone brings defects below the high‑defect threshold.")


if __name__ == "__main__":
    main()
