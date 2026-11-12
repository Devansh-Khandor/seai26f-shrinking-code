#!/usr/bin/env python3
"""
Skill 03: Comparative analysis of fuel efficiency by cylinder count (CAR-05)
      and performance by model year range (CAR-06).

Usage:
    python skill_03.py <path-to-auto93.csv>
"""

import csv
import sys
import math
import statistics
from pathlib import Path

# Reuse functionality from the previous skills
from skill_01 import load_data, compute_balanced_scores, format_value
from skill_02 import _impute_mpg, _predict_weight  # not directly needed, but import for completeness


# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------
def _group_by_cylinders(rows):
    """
    Return a dictionary mapping cylinder count to a list of mpg values.
    Missing or non-numeric MPG values are ignored.
    """
    groups = {}
    for r in rows:
        try:
            cyl = int(r["Clndrs"])
            mpg = float(r["Mpg+"])
            if math.isnan(mpg):
                continue
            groups.setdefault(cyl, []).append(mpg)
        except (KeyError, ValueError, TypeError):
            continue
    return groups


def _group_by_year(rows, min_year, max_year):
    """
    Return a list of scores for vehicles whose model year falls within
    the inclusive range [min_year, max_year].
    """
    scores = []
    for r in rows:
        try:
            year = int(r["Model"])
            if min_year <= year <= max_year:
                # Balanced score already computed in main
                if "score" in r:
                    scores.append(r["score"])
        except (KeyError, ValueError, TypeError):
            continue
    return scores


def _mean_ci(values, z=1.96):
    """
    Return (mean, lower_bound, upper_bound) for a 95% confidence interval.
    For small sample sizes (n <= 1) returns (mean, None, None).
    """
    n = len(values)
    if n == 0:
        return (None, None, None)
    mean = statistics.mean(values)
    if n == 1:
        return (mean, None, None)
    stdev = statistics.stdev(values)
    se = stdev / math.sqrt(n)
    margin = z * se
    return (mean, mean - margin, mean + margin)


# ------------------------------------------------------------------
# CAR-05: Cylinder‑based fuel‑efficiency comparison
# ------------------------------------------------------------------
def _car_05(rows):
    groups = _group_by_cylinders(rows)
    # Gather statistics
    stats = {}
    for cyl, mpg_list in sorted(groups.items()):
        mean, low, high = _mean_ci(mpg_list)
        stats[cyl] = {"count": len(mpg_list), "mean": mean, "ci_low": low, "ci_high": high}

    # Determine if any pair of groups has overlapping CIs
    reliable = True
    cyls = sorted(stats.keys())
    for i in range(len(cyls)):
        for j in range(i + 1, len(cyls)):
            a = stats[cyls[i]]
            b = stats[cyls[j]]
            if a["ci_low"] is None or b["ci_low"] is None:
                continue
            if not (a["ci_high"] < b["ci_low"] or b["ci_high"] < a["ci_low"]):
                reliable = False
                break
        if not reliable:
            break

    # Output
    print("== CAR-05 ==")
    print("Fuel efficiency by cylinder count:")
    for cyl in cyls:
        s = stats[cyl]
        ci_str = (
            f"[{format_value(s['ci_low'])}, {format_value(s['ci_high'])}]"
            if s["ci_low"] is not None
            else "(no CI)"
        )
        print(
            f"{cyl} cyl: {s['count']} vehicles, "
            f"Avg MPG: {format_value(s['mean'])} {ci_str}"
        )
    if reliable:
        print(
            "\nThe differences in mean MPG between cylinder groups are statistically "
            "significant and reliable for inventory decisions."
        )
    else:
        print(
            "\nThe confidence intervals overlap; the observed differences in MPG "
            "may not be statistically reliable enough to drive inventory decisions."
        )


# ------------------------------------------------------------------
# CAR-06: Performance comparison across model year ranges
# ------------------------------------------------------------------
def _car_06(rows):
    # First compute balanced scores for all rows
    scored_rows = compute_balanced_scores(rows)

    # Define two year ranges (example: 1970–1974 vs 1975–1979)
    range1 = (1970, 1974)
    range2 = (1975, 1979)

    scores1 = _group_by_year(scored_rows, *range1)
    scores2 = _group_by_year(scored_rows, *range2)

    # Compute statistics
    def stats_for(scores):
        return _mean_ci(scores)

    mean1, low1, high1 = stats_for(scores1)
    mean2, low2, high2 = stats_for(scores2)

    # Determine if CIs overlap
    reliable = True
    if low1 is not None and low2 is not None:
        if not (high1 < low2 or high2 < low1):
            reliable = False

    # Output
    print("\n== CAR-06 ==")
    print("Overall performance (balanced score) by model year range:")
    print(
        f"{range1[0]}–{range1[1]}: {len(scores1)} vehicles, "
        f"Avg Score: {format_value(mean1)} "
        f"[{format_value(low1)}, {format_value(high1)}]"
    )
    print(
        f"{range2[0]}–{range2[1]}: {len(scores2)} vehicles, "
        f"Avg Score: {format_value(mean2)} "
        f"[{format_value(low2)}, {format_value(high2)}]"
    )
    if reliable:
        print(
            "\nThe confidence intervals for the two year ranges do not overlap; "
            "the evidence supports a meaningful difference in overall performance."
        )
    else:
        print(
            "\nThe confidence intervals overlap; the evidence is too weak to "
            "conclude a meaningful difference in overall performance."
        )


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <path-to-auto93.csv>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    if not Path(csv_path).is_file():
        print(f"Error: file '{csv_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    rows = load_data(csv_path)
    if not rows:
        print("Error: no data loaded.", file=sys.stderr)
        sys.exit(1)

    _car_05(rows)
    _car_06(rows)


if __name__ == "__main__":
    main()
