#!/usr/bin/env python3
"""
Skill 03: Comparative analysis of fuel efficiency (CAR‑05) and overall performance (CAR‑06)
for an used‑car inventory dataset.

The script accepts a single CSV path as its command‑line argument and prints the
required output for CAR‑05 and CAR‑06 as specified in the assignment.
"""

import sys
import csv
import json
import math
import statistics

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def read_csv(path):
    """
    Read CSV and return a list of rows (dicts).  Missing values '?' are converted to None.
    The header is discarded.  Row numbering starts at 1.
    """
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for lineno, line in enumerate(reader, start=1):
            parsed = {}
            for k, v in line.items():
                if v == '?' or v is None:
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = None
            rows.append(parsed)
    return rows

# ----------------------------------------------------------------------
# CAR‑05 utilities
# ----------------------------------------------------------------------
def group_stats(rows, group_col, target_col):
    """
    Return a mapping from group value to dict with keys 'n', 'mean', 'median', 'std'.
    Missing target values are ignored for that group's statistics.
    """
    groups = {}
    for r in rows:
        g = r.get(group_col)
        if g is None:
            continue
        g = str(int(g)) if isinstance(g, float) else str(g)
        val = r.get(target_col)
        if val is None:
            continue
        groups.setdefault(g, []).append(val)

    stats = {}
    for g, vals in groups.items():
        n = len(vals)
        mean = statistics.mean(vals) if n > 0 else None
        median = statistics.median(vals) if n > 0 else None
        std = statistics.stdev(vals) if n > 1 else 0.0
        stats[g] = {"n": n, "mean": mean, "median": median, "std": std}
    return stats

# ----------------------------------------------------------------------
# T‑test helpers
# ----------------------------------------------------------------------
# Critical t values for 95% two‑sided confidence (df 1‑30)
CRIT_T_VALUES = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
    23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
    30: 2.042
}

def t_test(mean1, var1, n1, mean2, var2, n2):
    """
    Welch's t‑test.  Return verdict 'different' if 95% CI for mean difference excludes 0,
    otherwise 'same'.  Handles small sample sizes by treating variance as 0.
    """
    # Insufficient data
    if n1 < 2 or n2 < 2:
        return "same"

    se = math.sqrt(var1 / n1 + var2 / n2)
    if se == 0:
        return "same"
    t_stat = abs((mean1 - mean2) / se)

    # Welch–Satterthwaite df
    df_num = (var1 / n1 + var2 / n2) ** 2
    df_den = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    df = df_num / df_den if df_den != 0 else 1
    df_int = int(max(1, math.floor(df)))

    crit_t = CRIT_T_VALUES.get(df_int, 1.96)
    return "different" if t_stat > crit_t else "same"

# ----------------------------------------------------------------------
# CAR‑06 utilities
# ----------------------------------------------------------------------
def compute_composite_scores(rows, goal_cols):
    """
    Compute a performance score for each row as the sum of normalized goal values.
    Normalization uses global min/max for each goal.  Rows with missing goals receive None.
    Returns a list of tuples (row_index, score) where row_index is 1‑based.
    """
    # Find global min and max per goal
    mins = {}
    maxs = {}
    for col in goal_cols:
        vals = [r[col] for r in rows if r.get(col) is not None]
        if vals:
            mins[col] = min(vals)
            maxs[col] = max(vals)
        else:
            mins[col] = maxs[col] = 0.0

    scores = []
    for idx, r in enumerate(rows, start=1):
        if any(r.get(col) is None for col in goal_cols):
            scores.append((idx, None))
            continue
        # Normalization
        norm_weight = ((maxs["Lbs-"] - r["Lbs-"]) / (maxs["Lbs-"] - mins["Lbs-"])) if maxs["Lbs-"] != mins["Lbs-"] else 0.0
        norm_acc = ((r["Acc+"] - mins["Acc+"]) / (maxs["Acc+"] - mins["Acc+"])) if maxs["Acc+"] != mins["Acc+"] else 0.0
        norm_mpg = ((r["Mpg+"] - mins["Mpg+"]) / (maxs["Mpg+"] - mins["Mpg+"])) if maxs["Mpg+"] != mins["Mpg+"] else 0.0
        scores.append((idx, norm_weight + norm_acc + norm_mpg))
    return scores

def split_model_groups(rows):
    """
    Split rows into two non‑overlapping inclusive Model ranges:
    group1: <= median Model value
    group2: > median Model value
    Returns list of dicts: {'from': int, 'to': int, 'indices': list of row indices}
    """
    # Extract model values
    models = [r["Model"] for r in rows if r["Model"] is not None]
    if not models:
        return []
    sorted_models = sorted(models)
    mid_idx = (len(sorted_models) - 1) // 2
    median_value = sorted_models[mid_idx]

    group1 = []
    group2 = []
    for idx, r in enumerate(rows, start=1):
        if r["Model"] is None:
            continue
        if r["Model"] <= median_value:
            group1.append(idx)
        else:
            group2.append(idx)

    return [
        {"from": int(min(sorted_models)), "to": int(median_value), "indices": group1},
        {"from": int(median_value + 1), "to": int(max(sorted_models)), "indices": group2}
    ]

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_path):
    rows = read_csv(csv_path)

    # ---- CAR‑05 -------------------------------------------------------
    print("== CAR-05 ==")
    mpg_stats = group_stats(rows, "Clndrs", "Mpg+")

    # Print group summaries
    for g in sorted(mpg_stats.keys(), key=lambda x: int(x)):
        s = mpg_stats[g]
        print(f"Group Clndrs={g}: n={s['n']}, mean={s['mean']:.4f}, median={s['median']:.4f}")

    # Compare every pair of groups
    comparisons = []
    group_keys = sorted(mpg_stats.keys(), key=lambda x: int(x))
    for i in range(len(group_keys)):
        for j in range(i + 1, len(group_keys)):
            a = group_keys[i]
            b = group_keys[j]
            sa = mpg_stats[a]
            sb = mpg_stats[b]
            verdict = t_test(sa["mean"], sa["std"]**2, sa["n"],
                             sb["mean"], sb["std"]**2, sb["n"])
            comparisons.append({"a": int(a), "b": int(b), "verdict": verdict})
            print(f"Comparison {a} vs {b}: {verdict}")

    car05_answer = {
        "groups": {
            g: {
                "n": mpg_stats[g]["n"],
                "median": mpg_stats[g]["median"],
                "mean": mpg_stats[g]["mean"]
            } for g in mpg_stats
        },
        "comparisons": comparisons
    }
    print("ANSWER CAR-05 " + json.dumps(car05_answer, separators=(',', ':')))

    # ---- CAR‑06 -------------------------------------------------------
    print("== CAR-06 ==")
    goal_cols = ["Lbs-", "Acc+", "Mpg+"]
    scores = compute_composite_scores(rows, goal_cols)

    # Map row index to score
    score_dict = {idx: sc for idx, sc in scores if sc is not None}

    # Group rows by model ranges
    model_groups = split_model_groups(rows)
    if not model_groups:
        # Edge: no models present
        print("No model data available.")
        sys.exit(1)

    # Prepare stats for each group
    group_stats_list = []
    group_means = []
    for grp in model_groups:
        indices = grp["indices"]
        group_scores = [score_dict[i] for i in indices if i in score_dict]
        n = len(indices)
        mean_score = statistics.mean(group_scores) if group_scores else None
        group_stats_list.append({
            "from": grp["from"],
            "to": grp["to"],
            "n": n
        })
        group_means.append((mean_score, len(group_scores)))

    # Compare performance between the two groups
    if len(group_means) == 2 and group_means[0][0] is not None and group_means[1][0] is not None:
        mean1, n1 = group_means[0]
        mean2, n2 = group_means[1]
        # Estimate variance from group_scores lengths
        # Compute standard deviation of each group
        stds = []
        for grp in model_groups:
            indices = grp["indices"]
            group_scores = [score_dict[i] for i in indices if i in score_dict]
            if len(group_scores) > 1:
                stds.append(statistics.stdev(group_scores) ** 2)
            else:
                stds.append(0.0)
        var1 = stds[0]
        var2 = stds[1]
        verdict = t_test(mean1, var1, n1, mean2, var2, n2)
    else:
        verdict = "same"

    for grp in group_stats_list:
        print(f"Group Model {grp['from']} to {grp['to']}: n={grp['n']}")

    car06_answer = {
        "groups": group_stats_list,
        "verdict": verdict
    }
    print("ANSWER CAR-06 " + json.dumps(car06_answer, separators=(',', ':')))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
