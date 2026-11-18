#!/usr/bin/env python3
"""
skill_03.py – NASA-05 & NASA-06

The program reads a CSV file whose path is supplied on the command line and
produces two reports:

* NASA-05 – compare Effort- between groups of projects that differ in
  programmer capability (pcap).
* NASA-06 – compare Defects- between groups of projects that differ in
  complexity (cplx).

For each comparison, the program decides whether the difference between the
median goal values is statistically reliable using a very simple rule:
the absolute difference between medians must exceed twice the standard error
of that difference.  The rule uses the population standard deviation and the
sample size of each group.

The program prints at least three lines of human‑readable text for each
report and ends each report with a machine‑checked JSON line starting with
`ANSWER <ID>`.
"""

import csv
import json
import math
import statistics
import sys

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def read_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(val):
    """Convert a string to float, handling missing values."""
    if val is None:
        return None
    val = val.strip()
    if val == "" or val == "?":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def median(lst):
    if not lst:
        return None
    return statistics.median(lst)


def mean(lst):
    if not lst:
        return None
    return statistics.mean(lst)


def std_dev(lst):
    if not lst:
        return 0.0
    # population standard deviation
    return statistics.pstdev(lst)


def standard_error(std, n):
    if n == 0:
        return 0.0
    return std / math.sqrt(n)


def group_stats(values):
    """Return n, median, mean, std dev, standard error for a list of floats."""
    n = len(values)
    m = median(values)
    avg = mean(values)
    std = std_dev(values)
    se = standard_error(std, n)
    return {"n": n, "median": m, "mean": avg, "std": std, "se": se}


def compare_groups(stats_a, stats_b, diff_key):
    """Return verdict 'different' or 'same' based on median difference."""
    med_a = stats_a[diff_key]
    med_b = stats_b[diff_key]
    if med_a is None or med_b is None:
        return "same"

    diff = abs(med_a - med_b)
    se_a = stats_a["se"]
    se_b = stats_b["se"]
    threshold = 2 * math.sqrt(se_a**2 + se_b**2)
    return "different" if diff > threshold else "same"


# ----------------------------------------------------------------------
# Main report generation
# ----------------------------------------------------------------------
def nasa05(rows):
    """Generate NASA‑05 report."""
    # Group by pcap, collect Effort- values
    groups = {}
    for row in rows:
        pcap = row.get("pcap", None)
        effort = to_float(row.get("Effort-", None))
        if pcap is None or pcap.strip() == "":
            continue
        if effort is None:
            continue
        groups.setdefault(pcap, []).append(effort)

    # Compute stats per group
    group_stats_dict = {}
    for k, vals in groups.items():
        stats = group_stats(vals)
        group_stats_dict[k] = {
            "n": stats["n"],
            "median": stats["median"],
            "mean": stats["mean"],
        }

    # Compare every pair of groups
    comparisons = []
    keys = list(groups.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = keys[i]
            b = keys[j]
            stats_a = group_stats(groups[a])
            stats_b = group_stats(groups[b])
            verdict = compare_groups(stats_a, stats_b, "median")
            comparisons.append({"a": a, "b": b, "verdict": verdict})

    # Print report
    print("== NASA-05 ==")
    # At least three lines of results
    for k in list(group_stats_dict.keys())[:3]:
        g = group_stats_dict[k]
        print(f"Group {k}: n={g['n']}, median={g['median']:.2f}, mean={g['mean']:.2f}")
    for comp in comparisons[:3]:
        print(f"Compare {comp['a']} vs {comp['b']}: verdict={comp['verdict']}")
    # Machine‑checked answer
    answer = {"groups": group_stats_dict, "comparisons": comparisons}
    print("ANSWER NASA-05", json.dumps(answer, separators=(",", ":")))


def nasa06(rows):
    """Generate NASA‑06 report."""
    # Group by cplx, collect Defects- values
    groups = {}
    for row in rows:
        cplx = row.get("cplx", None)
        defects = to_float(row.get("Defects-", None))
        if cplx is None or cplx.strip() == "":
            continue
        if defects is None:
            continue
        groups.setdefault(cplx, []).append(defects)

    # Compute stats per group
    group_stats_dict = {}
    for k, vals in groups.items():
        stats = group_stats(vals)
        group_stats_dict[k] = {
            "n": stats["n"],
            "median": stats["median"],
            "mean": stats["mean"],
        }

    # Compare every pair of groups
    comparisons = []
    keys = list(groups.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = keys[i]
            b = keys[j]
            stats_a = group_stats(groups[a])
            stats_b = group_stats(groups[b])
            verdict = compare_groups(stats_a, stats_b, "median")
            comparisons.append({"a": a, "b": b, "verdict": verdict})

    # Print report
    print("== NASA-06 ==")
    # At least three lines of results
    for k in list(group_stats_dict.keys())[:3]:
        g = group_stats_dict[k]
        print(f"Group {k}: n={g['n']}, median={g['median']:.2f}, mean={g['mean']:.2f}")
    for comp in comparisons[:3]:
        print(f"Compare {comp['a']} vs {comp['b']}: verdict={comp['verdict']}")
    # Machine‑checked answer
    answer = {"groups": group_stats_dict, "comparisons": comparisons}
    print("ANSWER NASA-06", json.dumps(answer, separators=(",", ":")))


def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = read_csv(csv_path)

    nasa05(rows)
    print()
    nasa06(rows)


if __name__ == "__main__":
    main()
