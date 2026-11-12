#!/usr/bin/env python3
"""
skill_07.py

A self‑contained script that extends the NASA project dataset with two new analyses:

- NASA-13: Identify promising combinations of project characteristics that
           historically delivered more software with lower development effort
           and fewer defects.
- NASA-14: Recommend which project proposals should be evaluated first
           to discover promising alternatives within a limited evaluation budget.

Run as:
    python skill_07.py path/to/nasa93dem.csv
"""

import csv
import sys
import statistics
from pathlib import Path
from itertools import combinations

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def _read_csv(path: Path):
    """Read CSV and return list of dicts with numeric conversion."""
    data = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {}
            for k, v in row.items():
                if v is None:
                    cleaned[k] = None
                    continue
                val = v.strip()
                try:
                    cleaned[k] = float(val)
                except ValueError:
                    cleaned[k] = val
            data.append(cleaned)
    return data


def _categorical_columns(data):
    """Return list of column names that are non‑numeric in all rows."""
    cat_cols = set()
    for row in data:
        for k, v in row.items():
            if isinstance(v, str):
                cat_cols.add(k)
    return list(cat_cols)


def _numeric_columns(data):
    """Return list of column names that are numeric in at least one row."""
    num_cols = set()
    for row in data:
        for k, v in row.items():
            if isinstance(v, (int, float)):
                num_cols.add(k)
    return list(num_cols)


def _group_stats(data, group_keys, num_key):
    """
    Compute statistics for each group defined by group_keys.
    Returns a dict mapping tuple of group values to (n, mean_num, std_num, min_num, max_num).
    """
    groups = {}
    for row in data:
        # Skip rows missing numeric value
        num_val = row.get(num_key)
        if not isinstance(num_val, (int, float)):
            continue
        # Build group key tuple
        key = tuple(row.get(g) for g in group_keys)
        if key not in groups:
            groups[key] = {"vals": []}
        groups[key]["vals"].append(num_val)

    stats = {}
    for key, info in groups.items():
        vals = info["vals"]
        n = len(vals)
        if n == 0:
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if n > 1 else 0.0
        stats[key] = {"n": n, "mean": mean, "std": std,
                      "min": min(vals), "max": max(vals)}
    return stats


def _linear_regression(x_vals, y_vals):
    """
    Return (intercept, slope, se, mean_x, ssx) for ordinary least squares.
    """
    n = len(x_vals)
    if n < 2:
        raise ValueError("Need at least two data points for regression.")
    mean_x = statistics.mean(x_vals)
    mean_y = statistics.mean(y_vals)
    ssx = sum((x - mean_x) ** 2 for x in x_vals)
    if ssx == 0:
        slope = 0.0
    else:
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals)) / ssx
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(x_vals, y_vals)]
    sse = sum(r ** 2 for r in residuals)
    se = (sse / (n - 2)) ** 0.5 if n > 2 else 0.0
    return intercept, slope, se, mean_x, ssx


def _predict(x, intercept, slope):
    """Simple linear prediction."""
    return intercept + slope * x


def _normalize(val, min_val, max_val):
    """Return a value between 0 and 1 for ranking (0 best)."""
    if max_val == min_val:
        return 0.0
    return (val - min_val) / (max_val - min_val)


# --------------------------------------------------------------------------- #
# NASA-13: promising combinations
# --------------------------------------------------------------------------- #

def _nasa_13(data):
    print("== NASA-13 ==")

    # Use only completed projects (with numeric Effort- and Defects- and Kloc+)
    completed = [
        r for r in data
        if isinstance(r.get("Effort-"), (int, float))
        and isinstance(r.get("Defects-"), (int, float))
        and isinstance(r.get("Kloc+"), (int, float))
    ]

    if not completed:
        print("No completed projects to analyze.")
        return

    cat_cols = _categorical_columns(completed)
    # Pick two categorical columns to analyze together
    pair_keys = []
    for a, b in combinations(cat_cols, 2):
        pair_keys.append((a, b))

    # For each pair, compute stats for Effort- and Defects-
    results = []
    for a, b in pair_keys:
        # Stats for Effort
        effort_stats = _group_stats(completed, [a, b], "Effort-")
        # Stats for Defects
        defect_stats = _group_stats(completed, [a, b], "Defects-")
        # Only keep groups present in both
        common_keys = set(effort_stats) & set(defect_stats)
        for key in common_keys:
            eff = effort_stats[key]
            defc = defect_stats[key]
            if eff["n"] < 3:
                continue  # need at least 3 samples
            # Composite score: weighted sum of effort and defects per size
            size_stats = _group_stats(completed, [a, b], "Kloc+")
            size = size_stats.get(key)
            if not size or size["n"] < 3:
                continue
            # Normalize by size to compare groups fairly
            score = (eff["mean"] / size["mean"]) + (defc["mean"] / size["mean"])
            results.append({
                "group": key,
                "n": eff["n"],
                "size_mean": size["mean"],
                "eff_mean": eff["mean"],
                "def_mean": defc["mean"],
                "score": score
            })

    if not results:
        print("No sufficient group data to produce promising combinations.")
        return

    # Sort by score ascending (lower better)
    results.sort(key=lambda r: r["score"])
    top_n = 5
    print(f"Top {min(top_n, len(results))} promising characteristic combinations:")
    for r in results[:top_n]:
        group_desc = ", ".join(f"{k}='{v}'" for k, v in zip(pair_keys[0], r["group"]))
        print(f"  {group_desc} -> Avg Kloc+={r['size_mean']:.1f}, "
              f"Avg Effort-={r['eff_mean']:.1f}, Avg Defects-={r['def_mean']:.1f}")
    print()


# --------------------------------------------------------------------------- #
# NASA-14: proposal prioritization
# --------------------------------------------------------------------------- #

def _nasa_14(data):
    print("== NASA-14 ==")

    # Separate proposals (missing Effort- or Defects-)
    proposals = [
        r for r in data
        if not isinstance(r.get("Effort-"), (int, float))
        or not isinstance(r.get("Defects-"), (int, float))
    ]

    if not proposals:
        print("No proposals detected.")
        return

    # Completed projects for regression
    completed = [
        r for r in data
        if isinstance(r.get("Effort-"), (int, float))
        and isinstance(r.get("Defects-"), (int, float))
        and isinstance(r.get("Kloc+"), (int, float))
    ]

    if not completed:
        print("Insufficient completed projects to build predictive models.")
        return

    # Build regression models
    try:
        e_int, e_slope, e_se, e_mean_x, e_ssx = _linear_regression(
            [r["Kloc+"] for r in completed],
            [r["Effort-"] for r in completed]
        )
        d_int, d_slope, d_se, d_mean_x, d_ssx = _linear_regression(
            [r["Effort-"] for r in completed],
            [r["Defects-"] for r in completed]
        )
    except Exception:
        print("Regression failed; cannot predict proposals.")
        return

    # Evaluate each proposal
    evals = []
    for r in proposals:
        proj_id = r.get("idX", "N/A")
        k = r.get("Kloc+")
        if not isinstance(k, (int, float)):
            continue
        eff_pred = _predict(k, e_int, e_slope)
        def_pred = _predict(eff_pred, d_int, d_slope)
        # Lower effort and defects are better
        score = eff_pred + def_pred  # simple sum; lower is better
        evals.append({
            "id": proj_id,
            "k": k,
            "eff_pred": eff_pred,
            "def_pred": def_pred,
            "score": score
        })

    if not evals:
        print("No proposals with usable Kloc+ for prediction.")
        return

    # Rank by score (ascending)
    evals.sort(key=lambda x: x["score"])
    top_n = 5
    print(f"Top {min(top_n, len(evals))} proposals to evaluate first:")
    for e in evals[:top_n]:
        print(f"  ID {e['id']}: Kloc+={e['k']:.1f} -> Effort≈{e['eff_pred']:.1f}, "
              f"Defects≈{e['def_pred']:.1f}, Score={e['score']:.1f}")
    print()


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_07.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    data = _read_csv(csv_path)

    _nasa_13(data)
    _nasa_14(data)


if __name__ == "__main__":
    main()
