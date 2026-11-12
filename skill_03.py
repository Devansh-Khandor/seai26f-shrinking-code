#!/usr/bin/env python3
"""
Skill 03: Statistical comparison of project characteristics.

Usage:
    python skill_03.py <path_to_nasa93dem.csv>
"""

import csv
import sys
import math

Z_SCORE = 1.96  # approximate for 95% confidence

def read_data(csv_path):
    """
    Read CSV and return list of project dictionaries with relevant fields.
    Non-positive Kloc+ entries are skipped.
    """
    projects = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                k = float(row["Kloc+"].strip())
                if k <= 0:
                    continue
                eff = float(row["Effort-"].strip())
                defc = float(row["Defects-"].strip())
                projects.append({
                    "id": row["idX"].strip(),
                    "flex": row["flex"].strip(),
                    "cplx": row["cplx"].strip(),
                    "Effort": eff,
                    "Defects": defc,
                })
            except (ValueError, KeyError):
                continue
    return projects

def compute_stats(values):
    n = len(values)
    if n == 0:
        return None, None, 0
    mean = sum(values) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / n)
    return mean, std, n

def confidence_interval(std, n):
    if n <= 1:
        return None
    return Z_SCORE * std / math.sqrt(n)

def format_num(val, decimals=2):
    if val is None or math.isnan(val) or math.isinf(val):
        return "N/A"
    return f"{val:.{decimals}f}".rstrip('0').rstrip('.')

def compare_groups(groups, metric):
    """
    Compare each pair of groups using confidence interval overlap.
    Returns a list of tuples (group1, group2, mean1, mean2, ci1, ci2, significant)
    """
    comparisons = []
    keys = list(groups.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            g1, g2 = keys[i], keys[j]
            mean1, std1, n1 = groups[g1]["stats"]
            mean2, std2, n2 = groups[g2]["stats"]
            ci1 = groups[g1]["ci"]
            ci2 = groups[g2]["ci"]

            # if either CI is unavailable, skip significance test
            if ci1 is None or ci2 is None:
                significant = None
            else:
                # overlap test: intervals [mean±ci]
                low1, high1 = mean1 - ci1, mean1 + ci1
                low2, high2 = mean2 - ci2, mean2 + ci2
                significant = not (high1 < low2 or high2 < low1)

            comparisons.append((g1, g2, mean1, mean2, ci1, ci2, significant))
    return comparisons

def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    projects = read_data(csv_path)

    # Group by programmer capability (flex) for effort comparison
    flex_groups = {}
    for p in projects:
        flex = p["flex"]
        flex_groups.setdefault(flex, []).append(p["Effort"])

    for rating, efforts in flex_groups.items():
        mean, std, n = compute_stats(efforts)
        ci = confidence_interval(std, n)
        flex_groups[rating]["stats"] = (mean, std, n)
        flex_groups[rating]["ci"] = ci

    # Group by complexity (cplx) for defect comparison
    cplx_groups = {}
    for p in projects:
        cplx = p["cplx"]
        cplx_groups.setdefault(cplx, []).append(p["Defects"])

    for rating, defects in cplx_groups.items():
        mean, std, n = compute_stats(defects)
        ci = confidence_interval(std, n)
        cplx_groups[rating]["stats"] = (mean, std, n)
        cplx_groups[rating]["ci"] = ci

    # NASA-05: Programmer capability vs effort
    print("== NASA-05 ==")
    print("Programmer capability ratings and effort statistics:")
    for rating, info in sorted(flex_groups.items()):
        mean, std, n = info["stats"]
        ci = info["ci"]
        print(f"  {rating}: mean effort {format_num(mean)} "
              f"(± {format_num(ci)}) on {n} project(s)")

    # Pairwise comparison
    comparisons = compare_groups(flex_groups, metric="Effort")
    print("\nPairwise effort difference significance:")
    for g1, g2, mean1, mean2, ci1, ci2, signif in comparisons:
        if signif is None:
            note = "insufficient data for CI"
        else:
            note = "significant difference" if signif else "not significant"
        print(f"  {g1} vs {g2}: {note}")

    # NASA-06: Complexity vs defects
    print("\n== NASA-06 ==")
    print("Complexity ratings and defect statistics:")
    for rating, info in sorted(cplx_groups.items()):
        mean, std, n = info["stats"]
        ci = info["ci"]
        print(f"  {rating}: mean defects {format_num(mean)} "
              f"(± {format_num(ci)}) on {n} project(s)")

    comparisons = compare_groups(cplx_groups, metric="Defects")
    print("\nPairwise defect difference significance:")
    for g1, g2, mean1, mean2, ci1, ci2, signif in comparisons:
        if signif is None:
            note = "insufficient data for CI"
        else:
            note = "significant difference" if signif else "not significant"
        print(f"  {g1} vs {g2}: {note}")

if __name__ == "__main__":
    main()
