#!/usr/bin/env python3
"""
Skill 03 – Apache performance analytics (APACHE-05 & APACHE-06)

Usage:
    python skill_03.py <csv_path>
"""

import json
import math
import sys
from itertools import combinations

import ezr


def _median(lst: list[float]) -> float | None:
    """Return the median of a list of numbers. None if empty."""
    if not lst:
        return None
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    mid = n // 2
    if n % 2:
        return sorted_lst[mid]
    return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2.0


def _mean(lst: list[float]) -> float | None:
    """Return the mean of a list of numbers. None if empty."""
    return sum(lst) / len(lst) if lst else None


def _group_stats(rows, col, goal_at):
    """
    Return a dict mapping group values to their stats.
    Each value is a dict with keys 'n', 'median', 'mean'.
    """
    groups = {}
    # Count rows per group, and collect performance values per group
    perf_by_group = {}
    for row in rows:
        key = row[col.at]
        if key == "?":      # cannot group a missing key
            continue
        groups.setdefault(key, {"n": 0, "values": []})
        groups[key]["n"] += 1
        # Collect goal value if present
        g = row[goal_at]
        if g != "?":
            groups[key]["values"].append(float(g))

    # Finalise stats
    for key, data in groups.items():
        vals = data.pop("values")
        data["median"] = _median(vals)
        data["mean"] = _mean(vals)

    return groups


def _comparisons(groups):
    """
    Return list of comparison objects for every unordered pair of group keys.
    Verdict is 'different' if the distributions differ significantly
    (i.e. same() is False), otherwise 'same'.
    """
    comps = []
    keys = sorted(groups.keys(), key=lambda x: str(x))
    for a, b in combinations(keys, 2):
        vals_a = groups[a]["median"] is not None
        vals_b = groups[b]["median"] is not None
        if not (vals_a and vals_b):
            # If either group has no performance values, we cannot judge
            verdict = "same"
        else:
            # Use ezr.same which checks Cliff's Delta and KS
            xs = groups[a]["median"] is not None
            # We need the actual values, not the stats
            # So we must keep the raw values in a separate structure
            # Instead of using median/mean, we recompute the values list
            # Actually, easier: we keep the values in the group dict temporarily
            pass
    # This approach is flawed; instead we will compute values separately

    return comps


def main(csv_path: str) -> None:
    tbl = ezr.Tbl(ezr.csv(csv_path))
    rows = tbl.rows

    # Find goal column (ends with '-')
    goal_col = next(c for c in tbl.cols.values() if c.txt.endswith("-"))
    goal_at = goal_col.at

    # Prepare groups for keepAlive
    keep_col = next(c for c in tbl.cols.values() if c.txt == "keepAlive")
    access_col = next(c for c in tbl.cols.values() if c.txt == "accessLog")

    # Build raw value lists for each group to use in comparison
    def build_group_raw(rows, col):
        raw = {}
        for row in rows:
            key = row[col.at]
            if key == "?":
                continue
            raw.setdefault(key, {"n": 0, "values": []})
            raw[key]["n"] += 1
            g = row[goal_at]
            if g != "?":
                raw[key]["values"].append(float(g))
        # finalize stats
        for k, d in raw.items():
            d["median"] = _median(d["values"])
            d["mean"] = _mean(d["values"])
        return raw

    keep_groups = build_group_raw(rows, keep_col)
    access_groups = build_group_raw(rows, access_col)

    # Helper to compare two groups using ezr.same
    def compare(group_a, group_b):
        vals_a = keep_groups[group_a]["values"]
        vals_b = keep_groups[group_b]["values"]
        return "different" if not ezr.same(vals_a, vals_b) else "same"

    def compare_access(a, b):
        vals_a = access_groups[a]["values"]
        vals_b = access_groups[b]["values"]
        return "different" if not ezr.same(vals_a, vals_b) else "same"

    # APACHE-05 output
    print("== APACHE-05 ==")
    # Print group details
    for k in sorted(keep_groups.keys(), key=lambda x: str(x)):
        g = keep_groups[k]
        print(f"KeepAlive={k}: n={g['n']}, mean={g['mean']:.2f}, median={g['median']:.2f}")
    # Comparisons
    print("Comparisons:")
    for a, b in combinations(sorted(keep_groups.keys(), key=lambda x: str(x)), 2):
        verdict = compare(a, b)
        print(f"{a} vs {b}: {verdict}")

    # Machine-checked answer
    answer = {
        "groups": {
            str(k): {"n": v["n"], "median": v["median"], "mean": v["mean"]}
            for k, v in keep_groups.items()
        },
        "comparisons": [
            {"a": str(a), "b": str(b), "verdict": compare(a, b)}
            for a, b in combinations(sorted(keep_groups.keys(), key=lambda x: str(x)), 2)
        ],
    }
    print(f"ANSWER APACHE-05 {json.dumps(answer)}")

    # APACHE-06 output
    print("\n== APACHE-06 ==")
    for k in sorted(access_groups.keys(), key=lambda x: str(x)):
        g = access_groups[k]
        print(f"AccessLog={k}: n={g['n']}, mean={g['mean']:.2f}, median={g['median']:.2f}")
    print("Comparisons:")
    for a, b in combinations(sorted(access_groups.keys(), key=lambda x: str(x)), 2):
        verdict = compare_access(a, b)
        print(f"{a} vs {b}: {verdict}")

    answer2 = {
        "groups": {
            str(k): {"n": v["n"], "median": v["median"], "mean": v["mean"]}
            for k, v in access_groups.items()
        },
        "comparisons": [
            {"a": str(a), "b": str(b), "verdict": compare_access(a, b)}
            for a, b in combinations(sorted(access_groups.keys(), key=lambda x: str(x)), 2)
        ],
    }
    print(f"ANSWER APACHE-06 {json.dumps(answer2)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
