#!/usr/bin/env python3
"""
Small data‑analysis skill for the auto93 dataset (CAR‑05 and CAR‑06).

Usage:
    python skill_03.py <path_to_csv>
"""

import sys
import json
import numpy as np
from skill_01 import load_data

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
def group_by_clndrs(data):
    """
    Return a dict mapping Clndrs value -> list of Mpg+ (floats) where present.
    """
    groups = {}
    for row in data:
        cl = row["values"]["Clndrs"]
        mpg = row["values"]["Mpg+"]
        if cl is None or mpg is None:
            continue
        groups.setdefault(cl, []).append(mpg)
    return groups

def compute_stats_for_groups(groups):
    """
    For each group compute n, mean, median, std.
    """
    stats = {}
    for cl, vals in groups.items():
        arr = np.array(vals)
        stats[cl] = {
            "n": int(len(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        }
    return stats

def compare_groups(stats):
    """
    For every unordered pair of Clndrs groups, decide if the
    mean Mpg+ differs reliably (using 95% CI of the difference).
    Return list of comparison dicts.
    """
    comps = []
    keys = sorted(stats.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            sa, sb = stats[a], stats[b]
            diff = abs(sa["mean"] - sb["mean"])
            se = np.sqrt(sa["std"] ** 2 / sa["n"] + sb["std"] ** 2 / sb["n"])
            # If SE is zero (very small groups), treat as same
            if se == 0:
                verdict = "same"
            else:
                verdict = "different" if diff > 1.96 * se else "same"
            comps.append({"a": a, "b": b, "verdict": verdict})
    return comps

def group_by_model_ranges(data):
    """
    Split rows into two non‑overlapping inclusive ranges based on Model value.
    Lower half -> first group, upper half -> second group.
    Return list of dicts with 'from', 'to', 'n'.
    """
    # All rows, including those with missing goal values
    models = sorted(set(row["values"]["Model"] for row in data if row["values"]["Model"] is not None))
    n_models = len(models)
    mid = n_models // 2
    group1_vals = models[:mid]
    group2_vals = models[mid:]
    groups = []
    for gvals in [group1_vals, group2_vals]:
        if not gvals:
            continue
        g_from = gvals[0]
        g_to = gvals[-1]
        # Count rows falling into this range
        n_rows = sum(1 for row in data
                     if row["values"]["Model"] is not None and g_from <= row["values"]["Model"] <= g_to)
        groups.append({"from": int(g_from), "to": int(g_to), "n": int(n_rows)})
    return groups

def overall_performance_stats(groups, data):
    """
    For each group, compute mean and std of all goal columns.
    Returns dict: group_idx -> { 'mean': [...], 'std': [...] }
    """
    goal_cols = ["Lbs-", "Acc+", "Mpg+"]
    stats = {}
    for idx, grp in enumerate(groups):
        g_from, g_to = grp["from"], grp["to"]
        rows = [row for row in data
                if row["values"]["Model"] is not None and g_from <= row["values"]["Model"] <= g_to
                and all(row["values"][c] is not None for c in goal_cols)]
        if not rows:
            stats[idx] = {"mean": [0.0]*len(goal_cols), "std": [0.0]*len(goal_cols), "n": 0}
            continue
        arr = np.array([[row["values"][c] for c in goal_cols] for row in rows], dtype=float)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
        stats[idx] = {"mean": mean.tolist(), "std": std.tolist(), "n": arr.shape[0]}
    return stats

def compare_model_groups(group_stats):
    """
    Determine if overall performance differs between the two groups.
    Use 95% CI on each goal. If any goal shows a significant difference,
    verdict is 'different', else 'same'.
    """
    # Must have exactly two groups
    if len(group_stats) < 2:
        return "same"
    # For each goal compare
    goal_cols = ["Lbs-", "Acc+", "Mpg+"]
    for g in range(2):
        pass
    stats1 = group_stats[0]
    stats2 = group_stats[1]
    for i, col in enumerate(goal_cols):
        m1, m2 = stats1["mean"][i], stats2["mean"][i]
        s1, s2 = stats1["std"][i], stats2["std"][i]
        n1, n2 = stats1["n"], stats2["n"]
        diff = abs(m1 - m2)
        se = np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2) if n1 > 0 and n2 > 0 else 0.0
        if se == 0:
            continue
        if diff > 1.96 * se:
            return "different"
    return "same"

# --------------------------------------------------------------------
# Main logic
# --------------------------------------------------------------------
def main(csv_path):
    data = load_data(csv_path)

    # === CAR-05 ===
    print("== CAR-05 ==")
    cl_groups = group_by_clndrs(data)
    cl_stats = compute_stats_for_groups(cl_groups)

    # Print group summaries
    for cl in sorted(cl_stats):
        st = cl_stats[cl]
        print(f"Clndrs {cl}: n={st['n']}, mean={st['mean']:.2f}, median={st['median']:.2f}")

    # Comparisons
    comps = compare_groups(cl_stats)
    for comp in comps:
        print(f"Compare {comp['a']} vs {comp['b']}: {comp['verdict']}")

    # Machine‑checked answer
    groups_json = {str(k): {"n": v["n"], "median": v["median"], "mean": v["mean"]} for k, v in cl_stats.items()}
    answer_json = json.dumps({"groups": groups_json,
                              "comparisons": comps},
                             separators=(",", ":"))
    print(f"ANSWER CAR-05 {answer_json}")

    # === CAR-06 ===
    print("\n== CAR-06 ==")
    model_groups = group_by_model_ranges(data)
    if len(model_groups) < 2:
        print("Insufficient model ranges for comparison.")
        verdict = "same"
    else:
        # Compute performance stats
        perf_stats = overall_performance_stats(model_groups, data)
        verdict = compare_model_groups(perf_stats)

        # Print group info
        for i, grp in enumerate(model_groups):
            print(f"Group {i+1} (Model {grp['from']}–{grp['to']}): n={grp['n']}")

        print(f"Verdict: overall performance is {verdict}")

    # Machine‑checked answer
    answer_json = json.dumps({"groups": model_groups,
                              "verdict": verdict},
                             separators=(",", ":"))
    print(f"ANSWER CAR-06 {answer_json}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
