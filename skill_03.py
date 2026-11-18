#!/usr/bin/env python3
"""
Skill 03: NASA-05 & NASA-06 analysis

Usage:
    python skill_03.py <path-to-nasa93dem.csv>
"""

import sys
import json
import math

# Import helper functions from skill_01
try:
    from skill_01 import read_csv, compute_stats, format_number
except ImportError:
    # Minimal reimplementation if skill_01 is not available
    import csv
    def read_csv(path):
        rows = []
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = {}
                for k, v in row.items():
                    if v.strip() == '?':
                        parsed[k] = None
                    else:
                        try:
                            parsed[k] = float(v)
                        except ValueError:
                            parsed[k] = v
                rows.append(parsed)
        return rows

    def compute_stats(values):
        n = len(values)
        if n == 0:
            return {"mean": None, "median": None, "min": None, "max": None}
        sorted_vals = sorted(values)
        m = sum(values) / n
        mid = n // 2
        if n % 2 == 1:
            med = sorted_vals[mid]
        else:
            med = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
        return {"mean": m, "median": med, "min": sorted_vals[0], "max": sorted_vals[-1]}

    def format_number(x):
        if x is None:
            return "null"
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        return f"{x:.6f}".rstrip('0').rstrip('.')

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def compute_std(values):
    """Sample standard deviation with Bessel's correction."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))

def group_stats(rows, group_col, goal_col):
    """Return a dict of group value -> stats dict (n, mean, median, std)."""
    groups = {}
    for r in rows:
        key = r[group_col]
        if key is None:
            continue  # skip rows missing the group value
        val = r[goal_col]
        if val is None:
            continue  # skip rows missing the goal value
        groups.setdefault(key, []).append(val)
    stats = {}
    for k, vals in groups.items():
        stats[k] = {
            "n": len(vals),
            "mean": compute_stats(vals)["mean"],
            "median": compute_stats(vals)["median"],
            "std": compute_std(vals)
        }
    return stats

def compare_groups(stats):
    """Return list of comparison dicts for every unordered pair."""
    keys = sorted(stats.keys(), key=lambda x: str(x))
    comparisons = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = keys[i]
            b = keys[j]
            sa = stats[a]
            sb = stats[b]
            # Compute standard error of difference of means
            if sa["n"] == 0 or sb["n"] == 0:
                verdict = "same"
            else:
                se = math.sqrt(sa["std"] ** 2 / sa["n"] + sb["std"] ** 2 / sb["n"])
                diff = abs(sa["mean"] - sb["mean"])
                verdict = "different" if diff > 2 * se else "same"
            comparisons.append({"a": a, "b": b, "verdict": verdict})
    return comparisons

def json_groups(stats):
    """Convert stats dict to the required JSON structure with string keys."""
    out = {}
    for k, v in stats.items():
        out[str(k)] = {
            "n": v["n"],
            "median": v["median"],
            "mean": v["mean"]
        }
    return out

# ----------------------------------------------------------------------
# NASA-05
# ----------------------------------------------------------------------
def nasa_05(rows):
    stats = group_stats(rows, "pcap", "Effort-")
    comparisons = compare_groups(stats)

    print("== NASA-05 ==")
    # Print group summaries
    for k in sorted(stats.keys(), key=lambda x: str(x)):
        v = stats[k]
        print(f"pcap={k}: n={v['n']}, mean={format_number(v['mean'])}, median={format_number(v['median'])}")

    # Print comparisons
    for comp in comparisons:
        print(f"{comp['a']} vs {comp['b']}: {comp['verdict']}")

    json_obj = {"groups": json_groups(stats), "comparisons": comparisons}
    json_str = json.dumps(json_obj, separators=(',', ':'))
    print(f"ANSWER NASA-05 {json_str}")

# ----------------------------------------------------------------------
# NASA-06
# ----------------------------------------------------------------------
def nasa_06(rows):
    stats = group_stats(rows, "cplx", "Defects-")
    comparisons = compare_groups(stats)

    print("== NASA-06 ==")
    # Print group summaries
    for k in sorted(stats.keys(), key=lambda x: str(x)):
        v = stats[k]
        print(f"cplx={k}: n={v['n']}, mean={format_number(v['mean'])}, median={format_number(v['median'])}")

    # Print comparisons
    for comp in comparisons:
        print(f"{comp['a']} vs {comp['b']}: {comp['verdict']}")

    json_obj = {"groups": json_groups(stats), "comparisons": comparisons}
    json_str = json.dumps(json_obj, separators=(',', ':'))
    print(f"ANSWER NASA-06 {json_str}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)
    path = sys.argv[1]
    rows = read_csv(path)
    nasa_05(rows)
    nasa_06(rows)

if __name__ == "__main__":
    main()
