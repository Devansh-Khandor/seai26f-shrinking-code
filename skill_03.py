#!/usr/bin/env python3
"""
Skill 03: Statistical comparison of vehicle groups

Usage:
    python skill_03.py <path to auto93.csv>
"""

import csv
import sys
import json
from skill_01 import read_csv

# ----------------------------------------------------------------------
# Basic statistics helpers

def mean(values):
    return sum(values) / len(values)

def median(values):
    v = sorted(values)
    n = len(v)
    if n % 2:
        return v[n // 2]
    return (v[n // 2 - 1] + v[n // 2]) / 2.0

def std_sample(values):
    n = len(values)
    if n <= 1:
        return 0.0
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / (n - 1)) ** 0.5

# ----------------------------------------------------------------------
# CAR-05: compare fuel efficiency by cylinder count

def mpg_stats_by_cyl(rows):
    groups = {}
    for idx, row in enumerate(rows, start=1):
        cyl = row.get("Clndrs")
        mpg = row.get("Mpg+")
        if mpg is None:
            continue
        key = str(cyl)
        groups.setdefault(key, []).append(mpg)
    # compute stats
    result = {}
    for key, vals in groups.items():
        result[key] = {
            "n": len(vals),
            "median": median(vals),
            "mean": mean(vals)
        }
    return result

def compare_groups(group_stats):
    keys = sorted(group_stats.keys(), key=lambda x: float(x))
    comparisons = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            vals_a = group_stats[a]["mean"]
            vals_b = group_stats[b]["mean"]
            # estimate std from the raw values? we need raw lists, recompute
            # Instead we compute difference significance using
            # the sample std from the raw data for each group
    return comparisons

# We need raw values for each group to compute std and se
def mpg_comparisons(rows):
    groups_vals = {}
    for row in rows:
        cyl = row.get("Clndrs")
        mpg = row.get("Mpg+")
        if mpg is None:
            continue
        groups_vals.setdefault(cyl, []).append(mpg)

    groups_stats = {}
    for cyl, vals in groups_vals.items():
        groups_stats[str(cyl)] = {
            "n": len(vals),
            "median": median(vals),
            "mean": mean(vals),
            "std": std_sample(vals)
        }

    comparisons = []
    cyl_keys = sorted(groups_stats.keys(), key=lambda x: float(x))
    for i in range(len(cyl_keys)):
        for j in range(i + 1, len(cyl_keys)):
            a, b = cyl_keys[i], cyl_keys[j]
            m1 = groups_stats[a]["mean"]
            m2 = groups_stats[b]["mean"]
            n1 = groups_stats[a]["n"]
            n2 = groups_stats[b]["n"]
            s1 = groups_stats[a]["std"]
            s2 = groups_stats[b]["std"]
            se = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 0.5 if (n1 + n2) > 0 else 0.0
            diff = abs(m1 - m2)
            verdict = "different" if diff > 1.96 * se else "same"
            comparisons.append({"a": a, "b": b, "verdict": verdict})
    return groups_stats, comparisons

# ----------------------------------------------------------------------
# CAR-06: compare performance across two model year ranges

def composite_score(row, goal_cols):
    score = 0.0
    for col in goal_cols:
        val = row.get(col)
        if val is None:
            return None
        if col.endswith("-"):
            score -= val
        else:
            score += val
    return score

def model_range_split(rows):
    models = sorted(set(row.get("Model") for row in rows if row.get("Model") is not None))
    if not models:
        return (None, None)
    mid = (len(models) - 1) // 2
    split_val = models[mid]
    return split_val

def performance_groups(rows):
    goal_cols = ["Lbs-", "Acc+", "Mpg+"]
    split = model_range_split(rows)
    group1 = []
    group2 = []
    for row in rows:
        mod = row.get("Model")
        if mod is None:
            continue
        if mod <= split:
            group1.append(row)
        else:
            group2.append(row)
    return group1, group2, split

def composite_stats(rows):
    scores = [composite_score(r, ["Lbs-", "Acc+", "Mpg+"]) for r in rows]
    scores = [s for s in scores if s is not None]
    n = len(scores)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0}
    m = mean(scores)
    s = std_sample(scores)
    return {"n": n, "mean": m, "std": s}

def compare_performance(rows):
    g1, g2, split = performance_groups(rows)
    stats1 = composite_stats(g1)
    stats2 = composite_stats(g2)
    diff = abs(stats1["mean"] - stats2["mean"])
    se = (stats1["std"] ** 2 / stats1["n"] + stats2["std"] ** 2 / stats2["n"]) ** 0.5 if (stats1["n"] + stats2["n"]) > 0 else 0.0
    verdict = "different" if diff > 1.96 * se else "same"
    group1_desc = {"from": min(row.get("Model") for row in g1), "to": max(row.get("Model") for row in g1), "n": stats1["n"]}
    group2_desc = {"from": min(row.get("Model") for row in g2), "to": max(row.get("Model") for row in g2), "n": stats2["n"]}
    return {"groups": [group1_desc, group2_desc], "verdict": verdict}

# ----------------------------------------------------------------------
# Main execution

def main(csv_path):
    rows = read_csv(csv_path)

    # CAR-05
    groups_stats, comparisons = mpg_comparisons(rows)
    print("== CAR-05 ==")
    # at least three lines of results
    for cyl, stats in sorted(groups_stats.items(), key=lambda x: float(x[0])):
        print(f"Cyl {cyl}: n={stats['n']}, median={stats['median']:.3f}, mean={stats['mean']:.3f}")
    for comp in comparisons:
        print(f"Compare {comp['a']} vs {comp['b']}: verdict={comp['verdict']}")
    print("ANSWER CAR-05 " + json.dumps({"groups": groups_stats, "comparisons": comparisons}))

    # CAR-06
    perf = compare_performance(rows)
    print("\n== CAR-06 ==")
    for idx, g in enumerate(perf["groups"], start=1):
        print(f"Group {idx}: from={g['from']} to={g['to']} n={g['n']}")
    print(f"Verdict: {perf['verdict']}")
    print("ANSWER CAR-06 " + json.dumps(perf))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
