#!/usr/bin/env python3
"""
Entry point for CAR-05 and CAR-06 skills.
"""

import json
import sys
from collections import defaultdict

# The ezr library is available in the same directory.
import ezr
from ezr import csv, same, ydist

# ----------------------------------------------------------------------
#  Utility helpers
# ----------------------------------------------------------------------
def mean(values):
    return sum(values) / len(values) if values else float("nan")

def median(values):
    n = len(values)
    if n == 0:
        return float("nan")
    s = sorted(values)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0

# ----------------------------------------------------------------------
#  CAR-05: cylinder comparison
# ----------------------------------------------------------------------
def car_05(rows, header):
    """
    Compare fuel efficiency of vehicles with different cylinder counts.
    """
    # Find column indexes
    clndrs_idx = header.index("Clndrs")
    mpg_idx    = header.index("Mpg+")

    # Group mpg values by cylinder count
    mpg_by_cyl = defaultdict(list)
    for row in rows:
        mpg = row[mpg_idx]
        if mpg == "?":
            continue
        cyl = row[clndrs_idx]
        mpg_by_cyl[cyl].append(float(mpg))

    # Build groups dict
    groups = {}
    for cyl, mpg_vals in mpg_by_cyl.items():
        groups[str(cyl)] = {
            "n": len(mpg_vals),
            "mean": round(mean(mpg_vals), 6),
            "median": round(median(mpg_vals), 6),
        }

    # Build comparisons list
    comparisons = []
    cyl_keys = sorted(mpg_by_cyl.keys(), key=lambda x: float(str(x)))
    for i in range(len(cyl_keys)):
        for j in range(i + 1, len(cyl_keys)):
            a = cyl_keys[i]
            b = cyl_keys[j]
            verdict = "different" if not same(mpg_by_cyl[a], mpg_by_cyl[b]) else "same"
            comparisons.append({
                "a": str(a),
                "b": str(b),
                "verdict": verdict,
            })

    # Print results
    print("== CAR-05 ==")
    for cyl, stats in groups.items():
        print(f"Cylinder {cyl}: n={stats['n']}, mean={stats['mean']}, median={stats['median']}")
    for comp in comparisons:
        print(f"Compare {comp['a']} vs {comp['b']}: {comp['verdict']}")
    # ANSWER line
    answer_json = {"groups": groups, "comparisons": comparisons}
    print("ANSWER CAR-05", json.dumps(answer_json, separators=(',', ':')))

# ----------------------------------------------------------------------
#  CAR-06: model year range comparison
# ----------------------------------------------------------------------
def car_06(rows, header):
    """
    Compare overall performance between two model-year ranges.
    """
    model_idx = header.index("Model")

    # Extract model values (int)
    models = [int(row[model_idx]) for row in rows if row[model_idx] != "?"]
    unique_models = sorted(set(models))
    # Determine threshold (median unique model)
    if not unique_models:
        threshold = None
    else:
        median_pos = len(unique_models) // 2
        threshold = unique_models[median_pos]

    # Build table for ydist calculation
    tbl = ezr.Tbl([header] + rows)

    # Prepare groups
    groups = []
    ydist_lists = []

    for group_idx, (lower, upper, label) in enumerate([
        (None, threshold, "A"),
        (threshold, None, "B"),
    ]):
        # Determine rows in this group
        group_rows = []
        group_ydists = []
        for row in rows:
            model_val = row[model_idx]
            if model_val == "?":
                continue
            m = int(model_val)
            in_group = False
            if lower is None:
                in_group = m <= upper
            else:
                in_group = m > lower
            if in_group:
                # Only consider rows with complete goal data
                if any(row[idx] == "?" for idx in range(len(header)) if header[idx][-1] in "+-"):
                    continue
                group_rows.append(row)
                group_ydists.append(ydist(tbl, row))

        # Compute group stats
        if group_rows:
            min_model = min(int(r[model_idx]) for r in group_rows)
            max_model = max(int(r[model_idx]) for r in group_rows)
        else:
            min_model = max_model = None

        groups.append({
            "from": min_model,
            "to": max_model,
            "n": len(group_rows),
        })
        ydist_lists.append(group_ydists)

    # Determine verdict
    if len(ydist_lists) == 2 and all(ydist_lists):
        verdict = "different" if not same(ydist_lists[0], ydist_lists[1]) else "same"
    else:
        verdict = "same"

    # Print results
    print("== CAR-06 ==")
    for idx, g in enumerate(groups):
        print(f"Group {idx+1} ({g['from']}-{g['to']}): n={g['n']}")
    print(f"Verdict: {verdict}")

    # ANSWER line
    answer_json = {"groups": groups, "verdict": verdict}
    print("ANSWER CAR-06", json.dumps(answer_json, separators=(',', ':')))

# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv path>")
        sys.exit(1)

    path = sys.argv[1]
    rows = [row for row in csv(path)]
    if not rows:
        print("No data found.")
        sys.exit(1)

    header = rows[0]
    data_rows = rows[1:]

    car_05(data_rows, header)
    print()
    car_06(data_rows, header)

if __name__ == "__main__":
    main()
