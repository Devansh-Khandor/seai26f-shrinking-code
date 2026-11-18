#!/usr/bin/env python3
"""
Skill 03: NASA-05 and NASA-06

This script analyses a CSV of software projects and produces two
analyses:

  * NASA-05 – For each distinct programmer capability rating (pcap)
    compute the number of projects, the median and mean of development
    effort (Effort-), and pairwise comparisons that indicate whether
    effort differences between groups are statistically different
    enough to be considered reliable.

  * NASA-06 – For each distinct complexity rating (cplx)
    compute the number of projects, the median and mean of defect
    outcomes (Defects-), and pairwise comparisons that indicate
    whether defect differences between groups are statistically
    different enough to be considered reliable.

The comparisons use the `same` function from the bundled `ezr` module,
which checks for indistinguishability using Cliff's delta and the
Kolmogorov‑Smirnov test.  If the two groups are indistinguishable,
the verdict is "same"; otherwise it is "different".

The script prints a human‑readable summary for each NASA request and
a machine‑checked answer line with the exact JSON object required
by the assessment system.
"""

import csv
import json
import sys
from typing import Any, Dict, List, Tuple

import ezr

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def atom(s: str) -> Any:
    """Coerce a CSV cell to a Python value."""
    s = s.strip()
    if s == "?":
        return s
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return {"True": True, "False": False}.get(s, s)


def median(xs: List[float]) -> float | None:
    """Return median of a list of numbers or None if the list is empty."""
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    mid = n // 2
    if n % 2:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2.0


def mean(xs: List[float]) -> float | None:
    """Return mean of a list of numbers or None if the list is empty."""
    if not xs:
        return None
    return sum(xs) / len(xs)


# ----------------------------------------------------------------------
# Group statistics
# ----------------------------------------------------------------------
def group_stats(
    rows: List[List[Any]],
    header: List[str],
    group_col: str,
    value_col: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[float]]]:
    """Return per‑group statistics and raw numeric values."""
    group_index = header.index(group_col)
    value_index = header.index(value_col)

    raw_values: Dict[str, List[float]] = {}
    counts: Dict[str, int] = {}

    for row in rows:
        g = row[group_index]
        counts[str(g)] = counts.get(str(g), 0) + 1

        v = row[value_index]
        if isinstance(v, (int, float)):
            raw_values.setdefault(str(g), []).append(v)

    groups: Dict[str, Dict[str, Any]] = {}
    for g, n in counts.items():
        vals = raw_values.get(g, [])
        groups[g] = {
            "n": n,
            "median": median(vals),
            "mean": mean(vals),
        }

    return groups, raw_values


# ----------------------------------------------------------------------
# Pairwise comparisons
# ----------------------------------------------------------------------
def pairwise_comparisons(
    groups: Dict[str, Dict[str, Any]],
    raw_values: Dict[str, List[float]],
) -> List[Dict[str, str]]:
    """Return a list of comparison verdicts for every unordered pair of groups."""
    keys = sorted(groups.keys())
    comps: List[Dict[str, str]] = []

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            vals_a = raw_values.get(a, [])
            vals_b = raw_values.get(b, [])
            verdict = "different" if not ezr.same(vals_a, vals_b) else "same"
            comps.append({"a": a, "b": b, "verdict": verdict})

    return comps


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]

    # Read data
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows_raw = [row for row in reader]

    rows = [[atom(cell) for cell in row] for row in rows_raw]

    # NASA‑05: pcap vs Effort-
    groups_05, raw_05 = group_stats(rows, header, "pcap", "Effort-")
    comps_05 = pairwise_comparisons(groups_05, raw_05)

    # NASA‑06: cplx vs Defects-
    groups_06, raw_06 = group_stats(rows, header, "cplx", "Defects-")
    comps_06 = pairwise_comparisons(groups_06, raw_06)

    # ------------------------------------------------------------------
    # Output human‑readable summary
    # ------------------------------------------------------------------
    print("== NASA-05 ==")
    print("Groups (pcap):")
    for g, stats in sorted(groups_05.items()):
        print(
            f"  {g}: n={stats['n']}, median={stats['median']}, mean={stats['mean']}"
        )
    print("Comparisons:")
    for comp in comps_05:
        print(f"  {comp['a']} vs {comp['b']}: {comp['verdict']}")

    print("\n== NASA-06 ==")
    print("Groups (cplx):")
    for g, stats in sorted(groups_06.items()):
        print(
            f"  {g}: n={stats['n']}, median={stats['median']}, mean={stats['mean']}"
        )
    print("Comparisons:")
    for comp in comps_06:
        print(f"  {comp['a']} vs {comp['b']}: {comp['verdict']}")

    # ------------------------------------------------------------------
    # Machine‑checked answer lines
    # ------------------------------------------------------------------
    answer_05 = {"groups": groups_05, "comparisons": comps_05}
    answer_06 = {"groups": groups_06, "comparisons": comps_06}

    print(f"\nANSWER NASA-05 {json.dumps(answer_05)}")
    print(f"ANSWER NASA-06 {json.dumps(answer_06)}")


if __name__ == "__main__":
    main()
