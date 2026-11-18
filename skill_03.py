#!/usr/bin/env python3
import sys
import json
import itertools
import ezr
from ezr import same  # statistical similarity test


def median(nums):
    """Return median of a list of numbers.  None if empty."""
    s = sorted(nums)
    n = len(s)
    if n == 0:
        return None
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def group_stats(rows, setting_col, goal_col):
    """Return dict mapping setting values to stats:
       {'value': {'n': int, 'median': float, 'mean': float}}
    """
    groups = {}
    # collect goal values for each setting value
    for row in rows:
        val = row[setting_col.at]
        goal = row[goal_col.at]
        if val == "?" or goal == "?":
            # skip rows with missing setting or goal value
            continue
        groups.setdefault(val, []).append(goal)

    result = {}
    for val, values in groups.items():
        n = len(values)
        result[str(val)] = {
            "n": n,
            "median": median(values),
            "mean": sum(values) / n if n else None,
        }
    return result


def comparisons(groups):
    """Return list of comparison dicts for all pairs of group values."""
    vals = list(groups.keys())
    comps = []
    for a, b in itertools.combinations(vals, 2):
        xs = groups[a]["median"]  # we need the actual values lists, not medians
        # retrieve the original lists
        # Actually we need the raw lists; so pass in the raw lists
        # Instead, modify to accept raw lists
        pass
    return comps


def compute_comparisons(raw_groups):
    """raw_groups: dict value -> list of performance numbers."""
    comps = []
    values = list(raw_groups.keys())
    for a, b in itertools.combinations(values, 2):
        xs = raw_groups[a]
        ys = raw_groups[b]
        verdict = "different" if not same(xs, ys) else "same"
        comps.append({"a": str(a), "b": str(b), "verdict": verdict})
    return comps


def print_group_results(title, groups, comps):
    print(f"== {title} ==")
    # Print each group
    for val, stats in groups.items():
        print(
            f"Setting {val}: n={stats['n']}, "
            f"mean={stats['mean']:.2f}, median={stats['median']:.2f}"
        )
    # Print comparisons
    for comp in comps:
        print(
            f"{comp['a']} vs {comp['b']}: verdict={comp['verdict']}"
        )
    print()  # blank line after section


def main(csv_path):
    rows = list(ezr.csv(csv_path))
    tbl = ezr.Tbl(rows)

    # Identify goal column (ends with '-')
    goal_col = next(col for col in tbl.cols.values() if col.txt.endswith("-"))

    # Keep raw lists for comparisons
    def raw_group_values(setting_col):
        groups = {}
        for row in rows:
            val = row[setting_col.at]
            goal = row[goal_col.at]
            if val == "?" or goal == "?":
                continue
            groups.setdefault(val, []).append(goal)
        return groups

    # APACHE-05: keepAlive
    keep_col = next(col for col in tbl.cols.values() if col.txt == "keepAlive")
    keep_raw = raw_group_values(keep_col)
    keep_stats = group_stats(rows, keep_col, goal_col)
    keep_comps = compute_comparisons(keep_raw)
    print_group_results("APACHE-05", keep_stats, keep_comps)
    answer_05 = {"groups": keep_stats, "comparisons": keep_comps}
    print(
        f"ANSWER APACHE-05 {json.dumps(answer_05, separators=(',', ':'))}"
    )

    # APACHE-06: accessLog
    log_col = next(col for col in tbl.cols.values() if col.txt == "accessLog")
    log_raw = raw_group_values(log_col)
    log_stats = group_stats(rows, log_col, goal_col)
    log_comps = compute_comparisons(log_raw)
    print_group_results("APACHE-06", log_stats, log_comps)
    answer_06 = {"groups": log_stats, "comparisons": log_comps}
    print(
        f"ANSWER APACHE-06 {json.dumps(answer_06, separators=(',', ':'))}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
