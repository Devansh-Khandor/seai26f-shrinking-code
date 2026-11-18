#!/usr/bin/env python3
"""
Skill 03 – NASA-05 & NASA-06
"""
import csv
import json
import sys
import statistics

# ----------------------------------------------------------------------
# Helper functions (copied from skill_01)
# ----------------------------------------------------------------------
GOALS = ["Kloc+", "Effort-", "Defects-"]  # order matters for output

def parse_value(v):
    """Parse a CSV cell to float, return None for missing."""
    if v == "" or v == "?":
        return None
    try:
        return float(v)
    except ValueError:
        return None

# ----------------------------------------------------------------------
# Group statistics helpers
# ----------------------------------------------------------------------
def group_stats(rows, group_col, target_col):
    """
    Returns a dict of {group_value: {"n": count, "median": m, "mean": mean}}
    """
    groups = {}
    for row in rows:
        g_val = row.get(group_col)
        # Skip rows with missing group value
        if g_val is None:
            continue
        if g_val not in groups:
            groups[g_val] = {"rows": [], "effort_vals": [], "defect_vals": []}
        groups[g_val]["rows"].append(row)
        # Collect target values only if present
        if target_col in GOALS:
            val = row.get(target_col)
            if val is not None:
                if target_col == "Effort-":
                    groups[g_val]["effort_vals"].append(val)
                elif target_col == "Defects-":
                    groups[g_val]["defect_vals"].append(val)
    # Build summary
    summary = {}
    for g, data in groups.items():
        n = len(data["rows"])
        effort_vals = data["effort_vals"]
        defect_vals = data["defect_vals"]
        median = statistics.median(effort_vals) if effort_vals else None
        mean = sum(effort_vals) / len(effort_vals) if effort_vals else None
        if target_col == "Defects-":
            median = statistics.median(defect_vals) if defect_vals else None
            mean = sum(defect_vals) / len(defect_vals) if defect_vals else None
        summary[g] = {"n": n, "median": median, "mean": mean}
    return summary

# ----------------------------------------------------------------------
# Comparisons
# ----------------------------------------------------------------------
def make_comparisons(group_summary):
    """
    For each pair of groups, determine if the difference is reliable.
    Here we deem the difference 'different' if the medians are both
    defined and not equal. Otherwise 'same'.
    """
    keys = sorted(group_summary.keys(), key=lambda x: (str(x)))
    comps = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = keys[i]
            b = keys[j]
            median_a = group_summary[a]["median"]
            median_b = group_summary[b]["median"]
            if median_a is not None and median_b is not None:
                verdict = "different" if median_a != median_b else "same"
            else:
                verdict = "same"
            comps.append({"a": a, "b": b, "verdict": verdict})
    return comps

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for col in reader.fieldnames:
                if col in GOALS:
                    parsed[col] = parse_value(row[col])
                else:
                    val = row[col]
                    if val == "" or val == "?":
                        parsed[col] = None
                    else:
                        parsed[col] = val
            rows.append(parsed)

    # ------------------------------------------------------------------
    # NASA-05: Programmer capability (pcap) groups, Effort-
    # ------------------------------------------------------------------
    groups_pc = group_stats(rows, "pcap", "Effort-")
    comps_pc = make_comparisons(groups_pc)

    print("== NASA-05 ==")
    # Print some results
    print(f"Number of pcap groups: {len(groups_pc)}")
    # Show first three groups
    for idx, (g, stats) in enumerate(groups_pc.items(), start=1):
        if idx > 3:
            break
        print(f"Group {g}: n={stats['n']}, median Effort-={stats['median']}, mean Effort-={stats['mean']}")
    # Show first three comparisons
    print("\nComparisons:")
    for idx, comp in enumerate(comps_pc[:3], start=1):
        print(f"{comp['a']} vs {comp['b']}: {comp['verdict']}")
    answer_05 = {
        "groups": groups_pc,
        "comparisons": comps_pc
    }
    print("ANSWER NASA-05 " + json.dumps(answer_05, separators=(',',':')))

    # ------------------------------------------------------------------
    # NASA-06: Complexity (cplx) groups, Defects-
    # ------------------------------------------------------------------
    groups_cplx = group_stats(rows, "cplx", "Defects-")
    comps_cplx = make_comparisons(groups_cplx)

    print("\n== NASA-06 ==")
    print(f"Number of cplx groups: {len(groups_cplx)}")
    for idx, (g, stats) in enumerate(groups_cplx.items(), start=1):
        if idx > 3:
            break
        print(f"Group {g}: n={stats['n']}, median Defects-={stats['median']}, mean Defects-={stats['mean']}")
    print("\nComparisons:")
    for idx, comp in enumerate(comps_cplx[:3], start=1):
        print(f"{comp['a']} vs {comp['b']}: {comp['verdict']}")
    answer_06 = {
        "groups": groups_cplx,
        "comparisons": comps_cplx
    }
    print("ANSWER NASA-06 " + json.dumps(answer_06, separators=(',',':')))

if __name__ == "__main__":
    main()
