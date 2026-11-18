#!/usr/bin/env python3
"""
skill_03.py: Compare effort and defect differences by rating.
Usage: python skill_03.py <csv-file>
"""

import json
import sys
from itertools import combinations
from typing import List, Dict, Tuple

import ezr


def _median(vals: List[float]) -> float:
    """Return the median of a non‑empty list of numbers."""
    vals = sorted(vals)
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return float(vals[mid])
    return (vals[mid - 1] + vals[mid]) / 2.0


def _mean(vals: List[float]) -> float:
    """Return the mean of a non‑empty list of numbers."""
    return sum(vals) / len(vals)


def _group_by(
    tbl: ezr.TBL, key_col: ezr.COL, val_col: ezr.COL
) -> Dict[ATOM, List[float]]:
    """
    Build a dict mapping key values to a list of numeric values.
    Skip rows where the value or the key is missing.
    """
    groups: Dict[ATOM, List[float]] = {}
    for row in tbl.rows:
        key = row[key_col.at]
        val = row[val_col.at]
        if key == "?" or val == "?":
            continue
        groups.setdefault(key, []).append(float(val))
    return groups


def _build_summary(groups: Dict[ATOM, List[float]]) -> Dict[ATOM, Dict[str, float]]:
    """
    For each group, compute n, median and mean of the collected values.
    """
    summary: Dict[ATOM, Dict[str, float]] = {}
    for key, vals in groups.items():
        if vals:
            summary[key] = {
                "n": len(vals),
                "median": _median(vals),
                "mean": _mean(vals),
            }
    return summary


def _compare_cohen(groups: Dict[ATOM, List[float]]) -> List[Dict[str, str]]:
    """Compare every pair of groups with Cohen's d (threshold 0.35)."""
    comps = []
    for a, b in combinations(groups.keys(), 2):
        vals_a = groups[a]
        vals_b = groups[b]
        verdict = "different" if not ezr.cohen(vals_a, vals_b) else "same"
        comps.append({"a": a, "b": b, "verdict": verdict})
    return comps


def _compare_same(groups: Dict[ATOM, List[float]]) -> List[Dict[str, str]]:
    """Compare every pair of groups with the 'same' test (cliffs + ks)."""
    comps = []
    for a, b in combinations(groups.keys(), 2):
        vals_a = groups[a]
        vals_b = groups[b]
        verdict = "same" if ezr.same(vals_a, vals_b) else "different"
        comps.append({"a": a, "b": b, "verdict": verdict})
    return comps


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv-file>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    tbl = ezr.Tbl(list(ezr.csv(path)))

    # NASA‑05: Programmer capability rating (pcap) vs Effort-
    pcap_col = next(c for c in tbl.cols.values() if c.txt == "pcap")
    effort_col = next(c for c in tbl.cols.values() if c.txt == "Effort-")

    eff_groups = _group_by(tbl, pcap_col, effort_col)
    eff_summary = _build_summary(eff_groups)
    eff_comparisons = _compare_cohen(eff_groups)

    print("== NASA-05 ==")
    for key, stats in eff_summary.items():
        print(
            f"Group pcap={key}: n={stats['n']}, "
            f"mean={stats['mean']:.3g}, median={stats['median']:.3g}"
        )
    for comp in eff_comparisons:
        print(
            f"Comparison pcap={comp['a']} vs pcap={comp['b']}: verdict={comp['verdict']}"
        )
    print(
        "ANSWER NASA-05 "
        + json.dumps(
            {"groups": eff_summary, "comparisons": eff_comparisons},
            separators=(",", ":"),
        )
    )

    # NASA‑06: Complexity rating (cplx) vs Defects-
    cplx_col = next(c for c in tbl.cols.values() if c.txt == "cplx")
    defects_col = next(c for c in tbl.cols.values() if c.txt == "Defects-")

    defect_groups = _group_by(tbl, cplx_col, defects_col)
    defect_summary = _build_summary(defect_groups)
    defect_comparisons = _compare_same(defect_groups)

    print("\n== NASA-06 ==")
    for key, stats in defect_summary.items():
        print(
            f"Group cplx={key}: n={stats['n']}, "
            f"mean={stats['mean']:.3g}, median={stats['median']:.3g}"
        )
    for comp in defect_comparisons:
        print(
            f"Comparison cplx={comp['a']} vs cplx={comp['b']}: verdict={comp['verdict']}"
        )
    print(
        "ANSWER NASA-06 "
        + json.dumps(
            {"groups": defect_summary, "comparisons": defect_comparisons},
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
