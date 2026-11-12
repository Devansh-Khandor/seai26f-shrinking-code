#!/usr/bin/env python3
"""
Small data‑analysis skill for NASA project data – new NASA‑17 and NASA‑18.

Usage:
    python skill_09.py <csv_path>
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
from skill_04 import group_numeric

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def categorical_fields(records: List[Dict[str, str]]) -> List[str]:
    """Return list of categorical field names (excluding idX and numeric fields)."""
    numeric = {"Kloc+", "Effort-", "Defects-", "MonthsX"}
    keys = records[0].keys()
    return [k for k in keys if k not in numeric and k != "idX"]

def group_by_combination(
    records: List[Dict[str, str]],
    cat_fields: List[str],
) -> Dict[Tuple[str, ...], List[int]]:
    """Group record ids by the values of all categorical fields."""
    groups: Dict[Tuple[str, ...], List[int]] = {}
    for r in records:
        key = tuple(r[f] for f in cat_fields)
        groups.setdefault(key, []).append(int(r["idX"]))
    return groups

def numeric_z_scores(record: Dict[str, str],
                     group_stats: Dict[Tuple[str, ...], Dict[str, float]],
                     cat_fields: List[str]) -> List[float]:
    """Compute z‑scores for numeric fields of a record relative to its group."""
    key = tuple(record[f] for f in cat_fields)
    stats = group_stats.get(key)
    if not stats:
        return []
    z = []
    for field in ("Kloc+", "Effort-", "Defects-", "MonthsX"):
        val = parse_float(record[field])
        if math.isnan(val):
            z.append(0.0)
            continue
        mean = stats[field]["mean"]
        std = stats[field]["std"]
        z.append((val - mean) / std if std > 0 else 0.0)
    return z

def compute_group_numeric_stats(records: List[Dict[str, str]],
                               cat_fields: List[str]) -> Dict[Tuple[str, ...], Dict[str, Dict[str, float]]]:
    """
    For each group defined by categorical fields, compute mean, std, n for numeric fields.
    Returns mapping: group_key -> numeric_field -> {mean, std, n}.
    """
    groups: Dict[Tuple[str, ...], Dict[str, List[float]]] = {}
    for r in records:
        key = tuple(r[f] for f in cat_fields)
        groups.setdefault(key, {"Kloc+": [], "Effort-": [], "Defects-": [], "MonthsX": []})
        for field in ("Kloc+", "Effort-", "Defects-", "MonthsX"):
            val = parse_float(r[field])
            if not math.isnan(val):
                groups[key][field].append(val)

    stats: Dict[Tuple[str, ...], Dict[str, Dict[str, float]]] = {}
    for key, nums in groups.items():
        stats[key] = {}
        for field, values in nums.items():
            n = len(values)
            mean = statistics.mean(values) if values else float('nan')
            std = statistics.stdev(values) if n > 1 else 0.0
            stats[key][field] = {"mean": mean, "std": std, "n": n}
    return stats

# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_09.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    records = load_data(csv_path)

    # -----------------------------------------------------------------------
    # NASA‑17 – Flag unusual characteristic combinations
    # -----------------------------------------------------------------------
    print("== NASA-17 ==")

    cat_fields = categorical_fields(records)
    comb_groups = group_by_combination(records, cat_fields)
    # Compute numeric stats per group
    group_num_stats = compute_group_numeric_stats(records, cat_fields)

    flagged_projects = []

    for r in records:
        pid = int(r["idX"])
        key = tuple(r[f] for f in cat_fields)
        group_size = len(comb_groups[key])
        reasons = []

        # Rare combination
        if group_size <= 2:
            reasons.append(f"rare combination (size={group_size})")

        # Numeric outlier in any field
        z_scores = numeric_z_scores(r, group_num_stats, cat_fields)
        for field, z in zip(("Kloc+", "Effort-", "Defects-", "MonthsX"), z_scores):
            if abs(z) > 2.0:
                reasons.append(f"{field} outlier (z={z:.2f})")

        if reasons:
            flagged_projects.append({"id": pid, "reasons": reasons})

    if flagged_projects:
        print(f"Found {len(flagged_projects)} project(s) with unusual characteristic combinations:")
        for p in flagged_projects[:10]:
            print(f"  - Project {p['id']}: {', '.join(p['reasons'])}")
    else:
        print("No projects with unusually different combinations found.")

    # -----------------------------------------------------------------------
    # NASA‑18 – Flag projects with unexpected effort or defect counts
    # -----------------------------------------------------------------------
    print("\n== NASA-18 ==")

    # Build regression models
    slope_ek, intercept_ek, resid_std_ek = linear_regression(
        extract_numeric_fields(records)["Kloc+"],
        extract_numeric_fields(records)["Effort-"]
    )
    slope_de, intercept_de, resid_std_de = linear_regression(
        extract_numeric_fields(records)["Effort-"],
        extract_numeric_fields(records)["Defects-"]
    )

    flagged_residuals = []

    for r in records:
        pid = int(r["idX"])
        size = parse_float(r["Kloc+"])
        effort = parse_float(r["Effort-"])
        defects = parse_float(r["Defects-"])

        pred_effort = predict(slope_ek, intercept_ek, size)
        effort_res = effort - pred_effort
        effort_outlier = False
        if not math.isnan(resid_std_ek) and resid_std_ek > 0:
            effort_outlier = abs(effort_res) > 2 * resid_std_ek

        pred_defects = predict(slope_de, intercept_de, effort)
        defect_res = defects - pred_defects
        defect_outlier = False
        if not math.isnan(resid_std_de) and resid_std_de > 0:
            defect_outlier = abs(defect_res) > 2 * resid_std_de

        if effort_outlier or defect_outlier:
            reasons = []
            if effort_outlier:
                reasons.append(f"effort residual {effort_res:.1f} > 2*sd")
            if defect_outlier:
                reasons.append(f"defects residual {defect_res:.1f} > 2*sd")
            flagged_residuals.append({"id": pid, "reasons": reasons})

    if flagged_residuals:
        print(f"Found {len(flagged_residuals)} project(s) with unexpected effort or defect counts:")
        for p in flagged_residuals[:10]:
            print(f"  - Project {p['id']}: {', '.join(p['reasons'])}")
    else:
        print("No projects with effort or defect counts beyond expected ranges.")


if __name__ == "__main__":
    main()
