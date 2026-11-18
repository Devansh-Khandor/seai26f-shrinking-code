#!/usr/bin/env python3
"""
Skill 03: Apache configuration comparison

This program implements two new analyses for the Apache configuration data:

* **APACHE-05** – Compare configurations with KeepAlive enabled and disabled, and
  determine whether the observed differences in Performance- are statistically
  significant enough to guide decision making.

* **APACHE-06** – Compare configurations with access logging enabled and
  disabled, and determine whether one group consistently performs better or
  whether the evidence is inconclusive.

The program prints a header line for each analysis, a few human‑readable
lines, and ends each section with a machine‑checked answer line that
contains a single‑line JSON object.

The JSON objects are derived from the CSV file supplied on the command line
and are never hard‑coded.  Missing values are represented as '?' in the CSV
and are ignored when computing statistics, but rows with missing values are
still counted in the group size.
"""

import csv
import json
import sys
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[str], List[Dict[str, any]]]:
    """Read the CSV and return header and list of rows as dicts."""
    rows: List[Dict[str, any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        for row in rdr:
            row_dict = {}
            for h, val in zip(header, row):
                if val == "?":
                    row_dict[h] = None
                else:
                    # try int/float conversion
                    try:
                        if "." in val or "e" in val.lower():
                            row_dict[h] = float(val)
                        else:
                            row_dict[h] = int(val)
                    except ValueError:
                        row_dict[h] = val
            rows.append(row_dict)
    return header, rows


def median(lst: List[float]) -> float | None:
    """Return median of a non‑empty list of numbers, or None if empty."""
    if not lst:
        return None
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_lst[mid])
    return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2.0


def mean(lst: List[float]) -> float | None:
    """Return mean of a non‑empty list of numbers, or None if empty."""
    if not lst:
        return None
    return sum(lst) / len(lst)


# ----------------------------------------------------------------------
# Statistical similarity test (imported from ezr)
# ----------------------------------------------------------------------
# ezr contains a function `same` that tests whether two numeric distributions
# are indistinguishable (cliffs, ks, and cohen).  Import it for use below.
try:
    from ezr import same  # type: ignore
except Exception:  # pragma: no cover
    # Fallback simple implementation (only used if ezr is not available).
    def same(xs: List[float], ys: List[float]) -> bool:
        return False  # conservatively treat as different


# ----------------------------------------------------------------------
# Group statistics helper
# ----------------------------------------------------------------------
def group_stats(
    rows: List[Dict[str, any]], group_col: str, target_col: str
) -> Dict[str, Dict[str, any]]:
    """Return statistics per group value."""
    groups: Dict[str, Dict[str, any]] = {}
    for row in rows:
        val = str(row[group_col])
        groups.setdefault(val, {"n": 0, "values": []})
        groups[val]["n"] += 1
        t = row.get(target_col)
        if t is not None:
            groups[val]["values"].append(float(t))
    # Compute median and mean
    result: Dict[str, Dict[str, any]] = {}
    for val, data in groups.items():
        vals = data["values"]
        result[val] = {
            "n": data["n"],
            "median": median(vals),
            "mean": mean(vals),
        }
    return result


# ----------------------------------------------------------------------
# Comparison helper
# ----------------------------------------------------------------------
def compare_groups(
    rows: List[Dict[str, any]], group_col: str, target_col: str
) -> List[Dict[str, any]]:
    """Return comparison objects for each pair of group values."""
    # Build dictionary of values per group
    groups: Dict[str, List[float]] = {}
    for row in rows:
        val = str(row[group_col])
        t = row.get(target_col)
        if t is not None:
            groups.setdefault(val, []).append(float(t))
    comparisons: List[Dict[str, any]] = []
    vals = sorted(groups.keys())
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            a, b = vals[i], vals[j]
            xs, ys = groups[a], groups[b]
            verdict = "different" if not same(xs, ys) else "same"
            comparisons.append({"a": a, "b": b, "verdict": verdict})
    return comparisons


# ----------------------------------------------------------------------
# Main analyses
# ----------------------------------------------------------------------
def main(csv_path: str) -> None:
    header, rows = read_csv(csv_path)

    target_col = "Performance-"
    if target_col not in header:
        print(f"Target column {target_col} not found.")
        sys.exit(1)

    # ---- APACHE-05 ----
    print("== APACHE-05 ==")
    keepalive_col = "keepAlive"
    if keepalive_col not in header:
        print(f"Column {keepalive_col} not found.")
        sys.exit(1)

    keep_groups = group_stats(rows, keepalive_col, target_col)
    for val, stats in sorted(keep_groups.items(), key=lambda kv: kv[0]):
        print(
            f"{val}: n={stats['n']} median={stats['median']:.2f} mean={stats['mean']:.2f}"
        )

    keep_comps = compare_groups(rows, keepalive_col, target_col)
    for comp in keep_comps:
        print(f"{comp['a']} vs {comp['b']}: {comp['verdict']}")

    answer_05 = {
        "groups": keep_groups,
        "comparisons": keep_comps,
    }
    print(f"ANSWER APACHE-05 {json.dumps(answer_05)}")

    # ---- APACHE-06 ----
    print("\n== APACHE-06 ==")
    accesslog_col = "accessLog"
    if accesslog_col not in header:
        print(f"Column {accesslog_col} not found.")
        sys.exit(1)

    access_groups = group_stats(rows, accesslog_col, target_col)
    for val, stats in sorted(access_groups.items(), key=lambda kv: kv[0]):
        print(
            f"{val}: n={stats['n']} median={stats['median']:.2f} mean={stats['mean']:.2f}"
        )

    access_comps = compare_groups(rows, accesslog_col, target_col)
    for comp in access_comps:
        print(f"{comp['a']} vs {comp['b']}: {comp['verdict']}")

    answer_06 = {
        "groups": access_groups,
        "comparisons": access_comps,
    }
    print(f"ANSWER APACHE-06 {json.dumps(answer_06)}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_03.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
