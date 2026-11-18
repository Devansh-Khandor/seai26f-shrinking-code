#!/usr/bin/env python3
"""
skill_05.py

Usage: python skill_05.py <csv path>
"""

import json
import math
import sys
from collections import Counter
from typing import Any, Dict, List

import ezr

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def numeric(v: Any) -> float | None:
    """Return a numeric value, or None if missing."""
    return None if v == "?" else v


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def mode(values: List[Any]) -> Any:
    """Most common value. If tie, return first encountered."""
    if not values:
        return None
    counts = Counter(values)
    max_count = max(counts.values())
    for v in values:
        if counts[v] == max_count:
            return v
    return values[0]


def compute_goal_minmax(header: List[str], rows: List[List[Any]]) -> Dict[str, tuple[float, float]]:
    """Return min/max for each goal column."""
    mins_maxs: Dict[str, tuple[float, float]] = {}
    for i, name in enumerate(header):
        if name[-1] not in "+-":
            continue
        vals = [numeric(r[i]) for r in rows if numeric(r[i]) is not None]
        if not vals:
            mins_maxs[name] = (0.0, 1.0)
        else:
            mn, mx = min(vals), max(vals)
            if mn == mx:
                mx = mn + 1.0
            mins_maxs[name] = (mn, mx)
    return mins_maxs


def normalize_goal(name: str, val: float, mins_maxs: Dict[str, tuple[float, float]]) -> float:
    mn, mx = mins_maxs[name]
    if name[-1] == "+":  # maximize
        return (val - mn) / (mx - mn)
    else:  # minimize
        return (mx - val) / (mx - mn)


def composite_score(row: List[Any], header: List[str],
                    mins_maxs: Dict[str, tuple[float, float]]) -> float:
    """Mean of normalized goal values available in row."""
    vals = []
    for i, name in enumerate(header):
        if name[-1] not in "+-":
            continue
        v = numeric(row[i])
        if v is not None:
            vals.append(normalize_goal(name, v, mins_maxs))
    return mean(vals) if vals else 0.0


# ----------------------------------------------------------------------
# CAR-09: Poor fuel efficiency
# ----------------------------------------------------------------------
def car_09(rows: List[List[Any]], header: List[str]) -> None:
    print("== CAR-09 ==")

    # Find Mpg+ column index
    mpg_idx = header.index("Mpg+")
    mpg_vals = [numeric(r[mpg_idx]) for r in rows if numeric(r[mpg_idx]) is not None]
    mpg_mean = mean(mpg_vals)
    mpg_std = (sum((x - mpg_mean) ** 2 for x in mpg_vals) / len(mpg_vals)) ** 0.5 if mpg_vals else 0.0

    # Poor rows: below mean - std
    poor_rows = [i + 1 for i, r in enumerate(rows) if numeric(r[mpg_idx]) is not None and
                 numeric(r[mpg_idx]) < mpg_mean - mpg_std]
    # If fewer than 5, take bottom 5 rows by mpg
    if len(poor_rows) < 5:
        sorted_by_mpg = sorted([(numeric(r[mpg_idx]), i + 1) for i, r in enumerate(rows)
                               if numeric(r[mpg_idx]) is not None], key=lambda x: x[0])
        poor_rows = [idx for _, idx in sorted_by_mpg[:5]]

    # Association ranking
    assoc: List[tuple[str, float]] = []
    for i, name in enumerate(header):
        if name[-1] in "+-X":  # skip goals and X columns
            continue
        values = [r[i] for r in rows if r[i] != "?"]
        if not values:
            continue
        if isinstance(values[0], (int, float)):
            # numeric: compare mean between poor and all
            all_vals = [numeric(r[i]) for r in rows if numeric(r[i]) is not None]
            bad_vals = [numeric(r[i]) for r in rows if numeric(r[i]) is not None and
                        r[mpg_idx] is not None and numeric(r[mpg_idx]) < mpg_mean - mpg_std]
            if not bad_vals:
                diff = 0.0
            else:
                diff = abs(mean(bad_vals) - mean(all_vals))
        else:
            # symbolic: compare mode proportion
            all_counts = Counter(values)
            all_mode, all_mode_cnt = all_counts.most_common(1)[0]
            all_mode_prop = all_mode_cnt / len(values)
            bad_values = [r[i] for r in rows if r[i] != "?"
                          and numeric(r[mpg_idx]) is not None
                          and numeric(r[mpg_idx]) < mpg_mean - mpg_std]
            bad_counts = Counter(bad_values)
            if bad_counts:
                bad_mode, bad_mode_cnt = bad_counts.most_common(1)[0]
                bad_mode_prop = bad_mode_cnt / len(bad_values)
                diff = abs(bad_mode_prop - all_mode_prop)
            else:
                diff = 0.0
        assoc.append((name, diff))

    assoc.sort(key=lambda x: -x[1])
    ranking = [name for name, _ in assoc[:3] if _ > 0]
    if len(ranking) < 3:
        ranking = [name for name, _ in assoc[:3]]

    # Print results
    print(f"Poor Mpg+ rows: {poor_rows}")
    print(f"Strongest associations (top 3): {ranking}")

    # Machine-checked answer
    answer = {"poor": poor_rows, "ranking": ranking}
    print(f"ANSWER CAR-09 {json.dumps(answer)}")


# ----------------------------------------------------------------------
# CAR-10: Overall performance comparison
# ----------------------------------------------------------------------
def car_10(rows: List[List[Any]], header: List[str]) -> None:
    print("== CAR-10 ==")

    mins_maxs = compute_goal_minmax(header, rows)

    # Compute composite scores
    scores = [composite_score(r, header, mins_maxs) for r in rows]

    n = len(rows)
    n_group = max(5, int(math.ceil(n * 0.1)))  # at least 5

    bad_indices = sorted(range(n), key=lambda i: scores[i])[:n_group]
    good_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)[:n_group]

    bad_rows = [i + 1 for i in bad_indices]
    good_rows = [i + 1 for i in good_indices]

    # Typical values per column
    def typical(group: List[int]) -> Dict[str, Any]:
        typ: Dict[str, Any] = {}
        for i, name in enumerate(header):
            if name[-1] in "+-X":
                continue
            vals = [r[i] for r in rows if r[i] != "?" and r in [rows[j] for j in group]]
            if not vals:
                continue
            if isinstance(vals[0], (int, float)):
                typ[name] = mean(vals)
            else:
                typ[name] = mode(vals)
        return typ

    bad_typ = typical(bad_indices)
    good_typ = typical(good_indices)

    # Differences
    diffs: List[tuple[str, float, Any, Any]] = []
    for name in bad_typ:
        if name not in good_typ:
            continue
        bad_val = bad_typ[name]
        good_val = good_typ[name]
        if isinstance(bad_val, (int, float)):
            diff = abs(bad_val - good_val)
        else:
            # symbol: difference in proportion of mode
            # compute proportions
            def prop(group: List[int], val: Any) -> float:
                count = sum(1 for j in group if rows[j][header.index(name)] == val)
                return count / len(group)
            diff = abs(prop(bad_indices, bad_val) - prop(good_indices, good_val))
        diffs.append((name, diff, bad_val, good_val))

    diffs.sort(key=lambda x: -x[1])
    top_diffs = diffs[:3]
    differences = [{"feature": f, "bad": b, "good": g} for f, _, b, g in top_diffs]

    # Print results
    print(f"Bad rows: {bad_rows}")
    print(f"Good rows: {good_rows}")
    print(f"Top differences: {differences}")

    # Machine-checked answer
    answer = {"bad": bad_rows, "good": good_rows, "differences": differences}
    print(f"ANSWER CAR-10 {json.dumps(answer)}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows_iter = ezr.csv(csv_path)
    header = next(rows_iter)
    data_rows: List[List[Any]] = list(rows_iter)

    car_09(data_rows, header)
    car_10(data_rows, header)


if __name__ == "__main__":
    main()
