#!/usr/bin/env python3
"""
Skill 03: Compare fuel efficiency by cylinders (CAR-05) and
overall performance by model-year ranges (CAR-06).

This script is self‑contained and does not depend on other skill modules.
It follows the same data handling conventions as the previous skills.
"""

import sys
import csv
import json
import math
import statistics
import itertools
from typing import List, Dict, Any, Tuple

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def read_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read CSV file and return header and list of rows (dicts)."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    return header, rows


def parse_value(v: str):
    """Parse numeric value or return None for missing data."""
    if v in ('', '?'):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def is_goal(col: str) -> bool:
    """Return True if column ends in '+' or '-'. """
    return col.endswith('+') or col.endswith('-')


# --------------------------------------------------------------------------- #
# CAR-05: Cylinder comparison
# --------------------------------------------------------------------------- #

def compute_group_stats(rows: List[Dict[str, str]]) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    For each Clndrs value compute:
        - n: total rows with this cylinder value
        - median: median Mpg+
        - mean: mean Mpg+
        - std: std deviation of Mpg+
    Return groups dict keyed by int Clndrs and list of comparison results.
    """
    # Gather groups
    groups_raw: Dict[int, List[float]] = {}
    groups_counts: Dict[int, int] = {}

    for row in rows:
        clndrs_val = parse_value(row.get('Clndrs'))
        if clndrs_val is None:
            continue
        cl = int(clndrs_val)
        groups_counts[cl] = groups_counts.get(cl, 0) + 1
        mpg_val = parse_value(row.get('Mpg+'))
        if mpg_val is not None:
            groups_raw.setdefault(cl, []).append(mpg_val)

    groups_stats: Dict[int, Dict[str, Any]] = {}
    for cl, mpg_list in groups_raw.items():
        n_total = groups_counts.get(cl, 0)
        if mpg_list:
            med = statistics.median(mpg_list)
            mean_val = statistics.mean(mpg_list)
            # std dev; if single point, stdev raises, set to 0
            try:
                std = statistics.stdev(mpg_list)
            except statistics.StatisticsError:
                std = 0.0
        else:
            # No MPG values
            med = mean_val = std = None
        groups_stats[cl] = {
            "n": n_total,
            "median": round(med, 4) if med is not None else None,
            "mean": round(mean_val, 4) if mean_val is not None else None,
            "std": std
        }

    # Comparisons
    comparisons: List[Dict[str, Any]] = []
    sorted_keys = sorted(groups_stats.keys())
    for a, b in itertools.combinations(sorted_keys, 2):
        a_stats = groups_stats[a]
        b_stats = groups_stats[b]
        # If either group lacks MPG values, cannot assess reliability
        if a_stats["mean"] is None or b_stats["mean"] is None:
            verdict = "same"
        else:
            # compute 95% CI for each mean
            n_a, n_b = a_stats["n"], b_stats["n"]
            se_a = a_stats["std"] / math.sqrt(n_a) if n_a > 1 else 0.0
            se_b = b_stats["std"] / math.sqrt(n_b) if n_b > 1 else 0.0
            low_a = a_stats["mean"] - 1.96 * se_a
            high_a = a_stats["mean"] + 1.96 * se_a
            low_b = b_stats["mean"] - 1.96 * se_b
            high_b = b_stats["mean"] + 1.96 * se_b
            # Check overlap
            if low_a > high_b or low_b > high_a:
                verdict = "different"
            else:
                verdict = "same"
        comparisons.append({"a": a, "b": b, "verdict": verdict})
    return groups_stats, comparisons


# --------------------------------------------------------------------------- #
# CAR-06: Model‑year range comparison
# --------------------------------------------------------------------------- #

def compute_model_groups(header: List[str], rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Split data into two non‑overlapping inclusive ranges based on 'Model'.
    Return list of two group dicts and verdict.
    """
    # Extract numeric Model values
    model_vals = []
    for row in rows:
        m = parse_value(row.get('Model'))
        if m is not None:
            model_vals.append(int(m))

    if not model_vals:
        # No data, return empty
        return [], "same"

    # Determine cutoff
    model_vals_sorted = sorted(model_vals)
    median_val = statistics.median(model_vals_sorted)
    cutoff = int(math.floor(median_val))

    # Build groups
    group_a_indices = [i for i, v in enumerate(model_vals) if v <= cutoff]
    group_b_indices = [i for i, v in enumerate(model_vals) if v > cutoff]

    group_a_rows = [rows[i] for i in group_a_indices]
    group_b_rows = [rows[i] for i in group_b_indices]

    # Group metadata
    def group_meta(rows_subset: List[Dict[str, str]]) -> Dict[str, Any]:
        if not rows_subset:
            return {"from": None, "to": None, "n": 0}
        models = [parse_value(r.get('Model')) for r in rows_subset if parse_value(r.get('Model')) is not None]
        return {"from": int(min(models)), "to": int(max(models)), "n": len(rows_subset)}

    meta_a = group_meta(group_a_rows)
    meta_b = group_meta(group_b_rows)

    # For each goal column compute mean and 95% CI per group
    goal_cols = [c for c in header if is_goal(c)]
    def compute_group_goal_stats(rows_subset: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
        stats = {}
        for col in goal_cols:
            vals = [parse_value(r[col]) for r in rows_subset if parse_value(r[col]) is not None]
            if vals:
                mean_val = statistics.mean(vals)
                try:
                    std = statistics.stdev(vals)
                except statistics.StatisticsError:
                    std = 0.0
                n = len(vals)
                se = std / math.sqrt(n) if n > 1 else 0.0
                low = mean_val - 1.96 * se
                high = mean_val + 1.96 * se
                stats[col] = {"mean": mean_val, "low": low, "high": high}
            else:
                # No data for this goal
                stats[col] = None
        return stats

    stats_a = compute_group_goal_stats(group_a_rows)
    stats_b = compute_group_goal_stats(group_b_rows)

    # Determine if any goal column shows non-overlapping CI
    verdict = "same"
    for col in goal_cols:
        a_stat = stats_a.get(col)
        b_stat = stats_b.get(col)
        if a_stat is None or b_stat is None:
            continue
        # Check overlap
        if a_stat["high"] < b_stat["low"] or b_stat["high"] < a_stat["low"]:
            verdict = "different"
            break

    # Assemble group list
    group_list = [
        {"from": meta_a["from"], "to": meta_a["to"], "n": meta_a["n"]},
        {"from": meta_b["from"], "to": meta_b["to"], "n": meta_b["n"]}
    ]
    return group_list, verdict


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    # ================== CAR-05 ==================
    print("== CAR-05 ==")
    groups_stats, comparisons = compute_group_stats(rows)

    # Print at least three lines
    for cl in sorted(groups_stats.keys()):
        g = groups_stats[cl]
        print(f"Group {cl}: n={g['n']}, median={g['median']:.4f}, mean={g['mean']:.4f}")

    for comp in comparisons:
        print(f"Comparison {comp['a']} vs {comp['b']}: verdict={comp['verdict']}")

    # Answer
    answer_car05 = {
        "groups": {
            str(cl): {"n": groups_stats[cl]["n"],
                      "median": groups_stats[cl]["median"],
                      "mean": groups_stats[cl]["mean"]}
            for cl in sorted(groups_stats.keys())
        },
        "comparisons": [{"a": comp["a"], "b": comp["b"], "verdict": comp["verdict"]} for comp in comparisons]
    }
    print(f"ANSWER CAR-05 {json.dumps(answer_car05)}")

    # ================== CAR-06 ==================
    print("\n== CAR-06 ==")
    group_list, verdict = compute_model_groups(header, rows)

    # Print at least three lines
    for idx, g in enumerate(group_list, start=1):
        print(f"Group {idx}: from={g['from']} to={g['to']} n={g['n']}")

    print(f"Overall performance verdict: {verdict}")

    # Answer
    answer_car06 = {
        "groups": group_list,
        "verdict": verdict
    }
    print(f"ANSWER CAR-06 {json.dumps(answer_car06)}")


if __name__ == "__main__":
    main()
