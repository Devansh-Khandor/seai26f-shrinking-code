#!/usr/bin/env python3
"""
skill_03.py: NASA-05 and NASA-06 analyses.
Usage:  python skill_03.py <csv path>
"""

import csv
import json
import math
import sys

# ----------------------------------------------------------------------
# Helper functions – copied from skill_01
# ----------------------------------------------------------------------
def parse_float(v):
    """Parse a string to float, ignoring '?'."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def median(lst):
    """Return median of a list of numbers."""
    n = len(lst)
    if n == 0:
        return None
    s = sorted(lst)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0

def std_dev(values, mean):
    """Population standard deviation."""
    if len(values) == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)

# ----------------------------------------------------------------------
# Core logic for NASA-05 and NASA-06
# ----------------------------------------------------------------------
def process_group(groups, target_col):
    """Return stats dict for each group and comparisons."""
    # stats per group
    stats = {}
    for g_val, vals in groups.items():
        n = len(vals)
        if n == 0:
            # skip groups with no valid data
            continue
        m = median(vals)
        mean_val = sum(vals) / n
        sd_val = std_dev(vals, mean_val)
        stats[g_val] = {
            "n": n,
            "median": round(m, 4) if m is not None else None,
            "mean": round(mean_val, 4),
            "sd": sd_val,  # keep for comparison
        }

    # pairwise comparisons
    comparisons = []
    sorted_keys = sorted(stats.keys())
    for i in range(len(sorted_keys)):
        a = sorted_keys[i]
        for j in range(i + 1, len(sorted_keys)):
            b = sorted_keys[j]
            stat_a = stats[a]
            stat_b = stats[b]
            mean_a = stat_a["mean"]
            mean_b = stat_b["mean"]
            n_a = stat_a["n"]
            n_b = stat_b["n"]
            sd_a = stat_a["sd"]
            sd_b = stat_b["sd"]
            se_diff = math.sqrt((sd_a ** 2) / n_a + (sd_b ** 2) / n_b)
            diff = abs(mean_a - mean_b)
            # If SE diff is zero, treat difference as reliable if diff > 0
            if se_diff == 0.0:
                verdict = "different" if diff > 0.0 else "same"
            else:
                verdict = "different" if diff > 1.96 * se_diff else "same"
            comparisons.append({
                "a": a,
                "b": b,
                "verdict": verdict
            })

    # Remove sd from stats before output
    for g in stats.values():
        g.pop("sd", None)

    return stats, comparisons

def main(csv_path):
    # Read CSV
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for idx, r in enumerate(reader, start=1):
            rows.append((idx, r))

    # NASA-05: Programmer Capability (pcap) vs Effort-
    pcap_groups = {}
    for idx, r in rows:
        pcap = r.get("pcap", "").strip()
        effort = parse_float(r.get("Effort-"))
        if effort is None:
            continue
        pcap_groups.setdefault(pcap, []).append(effort)

    pcap_stats, pcap_comparisons = process_group(pcap_groups, "Effort-")

    print("== NASA-05 ==")
    for g, stats in sorted(pcap_stats.items()):
        print(f"Group {g}: n={stats['n']} median={stats['median']} mean={stats['mean']}")
    for comp in pcap_comparisons:
        print(f"Compare {comp['a']} vs {comp['b']}: {comp['verdict']}")
    nasa05_json = {"groups": pcap_stats, "comparisons": pcap_comparisons}
    print(f"ANSWER NASA-05 {json.dumps(nasa05_json)}")

    # NASA-06: Complexity (cplx) vs Defects-
    cplx_groups = {}
    for idx, r in rows:
        cplx = r.get("cplx", "").strip()
        defect = parse_float(r.get("Defects-"))
        if defect is None:
            continue
        cplx_groups.setdefault(cplx, []).append(defect)

    cplx_stats, cplx_comparisons = process_group(cplx_groups, "Defects-")

    print("\n== NASA-06 ==")
    for g, stats in sorted(cplx_stats.items()):
        print(f"Group {g}: n={stats['n']} median={stats['median']} mean={stats['mean']}")
    for comp in cplx_comparisons:
        print(f"Compare {comp['a']} vs {comp['b']}: {comp['verdict']}")
    nasa06_json = {"groups": cplx_stats, "comparisons": cplx_comparisons}
    print(f"ANSWER NASA-06 {json.dumps(nasa06_json)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv path>")
        sys.exit(1)
    main(sys.argv[1])
