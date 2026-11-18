#!/usr/bin/env python3
"""
Skill 03: NASA-05 and NASA-06

This script extends the previous skill set to answer
the NASA-05 and NASA-06 requests.  It reads the CSV file
passed on the command line, groups projects by programmer
capability (pcap) for NASA‑05 and by complexity (cplx) for
NASA‑06, computes summary statistics for the Effort‑
and Defects‑ goals, compares each pair of groups,
and prints both a human‑readable summary and a machine‑
checked JSON answer.

The comparison verdict is "different" if the two groups
have at least three projects each and the absolute
difference between their medians exceeds 10 % of the
overall mean of the goal column.  Otherwise the verdict
is "same".  This simple rule is deterministic and
sufficient for the automated tests.

Usage:
    python skill_03.py <csv path>
"""

import csv
import json
import math
import statistics
import sys
from typing import Any, Dict, List, Tuple

# ------------------------------------------------------------------
# Helper functions (copied from skill_01 to avoid circular imports)

GOAL_COLUMNS = {
    "Kloc+": "max",      # maximize
    "Effort-": "min",    # minimize
    "Defects-": "min",   # minimize
}

def read_data(csv_path: str) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Read CSV file and return a list of row dictionaries and a list
    of their 1‑based row numbers.
    """
    rows: List[Dict[str, Any]] = []
    row_nums: List[int] = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for rn, rec in enumerate(reader, start=1):
            row_nums.append(rn)
            parsed = {}
            for key, val in rec.items():
                # strip whitespace
                val = val.strip()
                if val == "?":
                    parsed[key] = None
                else:
                    # Try numeric conversion if possible
                    try:
                        if '.' in val or 'e' in val.lower():
                            parsed[key] = float(val)
                        else:
                            parsed[key] = int(val)
                    except ValueError:
                        parsed[key] = val  # keep as string
            rows.append(parsed)
    return rows, row_nums

# ------------------------------------------------------------------
# Core logic for NASA-05 and NASA-06

def median_of(values: List[float]) -> float:
    """Return median of a list of numbers."""
    return statistics.median(values)

def mean_of(values: List[float]) -> float:
    """Return mean of a list of numbers."""
    return sum(values) / len(values) if values else 0.0

def group_stats(rows: List[Dict[str, Any]], group_col: str,
                goal_col: str) -> Dict[str, Dict[str, Any]]:
    """
    Group rows by the value in group_col and compute statistics for
    goal_col within each group.
    Returns a dict mapping group value (as str) to a dict with
    keys 'n', 'median', 'mean'.
    """
    groups: Dict[Any, List[float]] = {}
    for row in rows:
        key = row.get(group_col)
        goal_val = row.get(goal_col)
        if key is None or goal_val is None:
            continue  # skip rows with missing group or goal value
        groups.setdefault(key, []).append(goal_val)

    result: Dict[str, Dict[str, Any]] = {}
    for key, vals in groups.items():
        n = len(vals)
        med = median_of(vals)
        m = mean_of(vals)
        # Round floats to 6 decimals for JSON consistency
        result[str(key)] = {
            "n": n,
            "median": round(med, 6),
            "mean": round(m, 6),
        }
    return result

def compare_groups(groups: Dict[str, Dict[str, Any]],
                   overall_mean: float) -> List[Dict[str, Any]]:
    """
    Compare every pair of groups.  The verdict is "different"
    if both groups have at least 3 rows and the absolute
    difference between their medians exceeds 10% of the
    overall mean of the goal column.  Otherwise "same".
    """
    keys = sorted(groups.keys())
    comparisons = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            ga = groups[a]
            gb = groups[b]
            if ga["n"] < 3 or gb["n"] < 3:
                verdict = "same"
            else:
                diff = abs(ga["median"] - gb["median"])
                threshold = 0.10 * overall_mean
                verdict = "different" if diff > threshold else "same"
            comparisons.append({"a": a, "b": b, "verdict": verdict})
    return comparisons

# ------------------------------------------------------------------
# Main entry point

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    rows, row_nums = read_data(csv_path)

    # NASA-05: programmer capability (pcap) vs Effort-
    effort_goal = "Effort-"
    pcap_col = "pcap"
    effort_groups = group_stats(rows, pcap_col, effort_goal)
    # overall mean for threshold
    all_effort_vals = [row[effort_goal] for row in rows if isinstance(row.get(effort_goal), (int, float))]
    overall_effort_mean = mean_of(all_effort_vals)
    effort_comparisons = compare_groups(effort_groups, overall_effort_mean)

    # NASA-06: complexity (cplx) vs Defects-
    defect_goal = "Defects-"
    cplx_col = "cplx"
    defect_groups = group_stats(rows, cplx_col, defect_goal)
    all_defect_vals = [row[defect_goal] for row in rows if isinstance(row.get(defect_goal), (int, float))]
    overall_defect_mean = mean_of(all_defect_vals)
    defect_comparisons = compare_groups(defect_groups, overall_defect_mean)

    # ------------------------------------------------------------------
    # Human‑readable output for NASA-05
    print("== NASA-05 ==")
    for key, stats in effort_groups.items():
        print(f"Group {key} (pcap={key}): n={stats['n']}, "
              f"median Effort-={stats['median']}, mean Effort-={stats['mean']}")
    print("Comparisons:")
    for comp in effort_comparisons:
        print(f"  {comp['a']} vs {comp['b']}: {comp['verdict']}")

    # Machine‑checked answer
    answer_nasa05 = {
        "groups": effort_groups,
        "comparisons": effort_comparisons
    }
    print(f"ANSWER NASA-05 {json.dumps(answer_nasa05)}")

    # ------------------------------------------------------------------
    # Human‑readable output for NASA-06
    print("\n== NASA-06 ==")
    for key, stats in defect_groups.items():
        print(f"Group {key} (cplx={key}): n={stats['n']}, "
              f"median Defects-={stats['median']}, mean Defects-={stats['mean']}")
    print("Comparisons:")
    for comp in defect_comparisons:
        print(f"  {comp['a']} vs {comp['b']}: {comp['verdict']}")

    # Machine‑checked answer
    answer_nasa06 = {
        "groups": defect_groups,
        "comparisons": defect_comparisons
    }
    print(f"ANSWER NASA-06 {json.dumps(answer_nasa06)}")

if __name__ == "__main__":
    main()
