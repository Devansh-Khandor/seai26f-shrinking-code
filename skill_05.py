#!/usr/bin/env python3
"""
skill_05.py: Implements CAR-09 and CAR-10 for used‑car dealership data.

Usage:
    python skill_05.py <path_to_csv>
"""

import csv
import json
import sys
import math
from typing import List, Tuple, Dict, Any, Optional


# ------------------------------------------------------------------
# CSV reading utilities
# ------------------------------------------------------------------
def read_csv(path: str) -> Tuple[List[str], List[List[Any]]]:
    """Read CSV, convert numeric cells, keep '?' as None."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows: List[List[Any]] = []
        for row in reader:
            conv_row: List[Any] = []
            for v in row:
                if v == "?":
                    conv_row.append(None)
                else:
                    try:
                        if "." in v or "e" in v.lower():
                            conv_row.append(float(v))
                        else:
                            conv_row.append(int(v))
                    except ValueError:
                        conv_row.append(v)
            rows.append(conv_row)
    return header, rows


# ------------------------------------------------------------------
# Helper functions for goal statistics
# ------------------------------------------------------------------
def goal_indices(header: List[str]) -> List[int]:
    """Return indices of goal columns (ending with '+' or '-')."""
    return [i for i, h in enumerate(header) if h[-1] in ("+", "-")]


def non_goal_indices(header: List[str]) -> List[int]:
    """Indices of columns that are not goals and not 'X' columns."""
    return [
        i
        for i, h in enumerate(header)
        if not h.endswith(("X", "+", "-")) and not h.endswith("X")
    ]


def is_numeric(val: Any) -> bool:
    return isinstance(val, (int, float))


# ------------------------------------------------------------------
# CAR-09 helpers
# ------------------------------------------------------------------
def pearson_corr(xs: List[float], ys: List[float]) -> float:
    """Compute Pearson correlation coefficient, return 0 if undefined."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / math.sqrt(den_x * den_y)


def most_associated_features(
    header: List[str], rows: List[List[Any]], mpg_idx: int
) -> List[str]:
    """Return non‑goal columns sorted by association with poor Mpg+ (descending)."""
    ncols = len(header)
    non_goals = [i for i in range(ncols) if i not in goal_indices(header) and not header[i].endswith("X")]
    feature_scores: List[Tuple[float, str]] = []

    for idx in non_goals:
        col_name = header[idx]
        xs = []
        ys = []
        for r in rows:
            val = r[idx]
            mpg_val = r[mpg_idx]
            if val is None or mpg_val is None:
                continue
            if is_numeric(val):
                xs.append(val)
                ys.append(mpg_val)
            else:
                # categorical: compute mean mpg for each category
                # find category with lowest mean mpg
                categories = {}
                for rr in rows:
                    cv = rr[idx]
                    mv = rr[mpg_idx]
                    if cv is None or mv is None:
                        continue
                    categories.setdefault(cv, []).append(mv)
                if not categories:
                    continue
                # overall mean mpg
                all_vals = [mv for l in categories.values() for mv in l]
                overall_mean = sum(all_vals) / len(all_vals)
                # find category with lowest mean mpg
                best_mean = min(
                    sum(vals) / len(vals) for vals in categories.values()
                )
                diff = abs(overall_mean - best_mean)
                feature_scores.append((diff, col_name))
                break  # already accounted, skip numeric branch

        else:
            # numeric branch
            if xs:
                corr = pearson_corr(xs, ys)
                feature_scores.append((abs(corr), col_name))

    # Sort by score desc, then by name asc for tie‑breaking
    feature_scores.sort(key=lambda x: (-x[0], x[1]))
    ranking = [name for _, name in feature_scores]
    return ranking


# ------------------------------------------------------------------
# CAR-10 helpers
# ------------------------------------------------------------------
def compute_normalized_score(
    row: List[Any], gidx: List[int], mins: Dict[int, float], maxs: Dict[int, float], ismax: Dict[int, bool]
) -> Optional[float]:
    """Return overall normalized score for a single row, or None if any goal missing."""
    total = 0.0
    for idx in gidx:
        v = row[idx]
        if v is None:
            return None
        mn = mins[idx]
        mx = maxs[idx]
        if mn == mx:
            norm = 0.5
        else:
            if ismax[idx]:
                norm = (v - mn) / (mx - mn)
            else:
                norm = (mx - v) / (mx - mn)
        total += norm
    return total


def overall_scores(
    rows: List[List[Any]], gidx: List[int]
) -> Tuple[List[float], List[int]]:
    """Return list of scores and corresponding row numbers (1‑based)."""
    ismax = {i: True if header[i][-1] == "+" else False for i in gidx}
    # Compute min and max per goal
    mins = {}
    maxs = {}
    for idx in gidx:
        vals = [r[idx] for r in rows if r[idx] is not None]
        if vals:
            mins[idx] = min(vals)
            maxs[idx] = max(vals)
        else:
            mins[idx] = maxs[idx] = None

    scores: List[float] = []
    row_nums: List[int] = []

    for i, r in enumerate(rows, start=1):
        s = compute_normalized_score(r, gidx, mins, maxs, ismax)
        if s is not None:
            scores.append(s)
            row_nums.append(i)
    return scores, row_nums


def typical_value_numeric(values: List[float]) -> float:
    return sum(values) / len(values)


def mode_value(values: List[Any]) -> Any:
    freq: Dict[Any, int] = {}
    for v in values:
        freq[v] = freq.get(v, 0) + 1
    # return value with highest frequency, tie broken by first seen
    max_count = -1
    mode = None
    for v in values:
        cnt = freq[v]
        if cnt > max_count:
            max_count = cnt
            mode = v
    return mode


def differences_between_sets(
    header: List[str],
    rows_bad: List[List[Any]],
    rows_good: List[List[Any]],
    bad_indices: List[int],
    good_indices: List[int],
) -> List[Dict[str, Any]]:
    """Return list of difference dicts sorted by magnitude (desc)."""
    diffs: List[Tuple[float, str, Any, Any]] = []
    for idx in bad_indices:
        col_name = header[idx]
        # collect values, skip None
        bad_vals = [r[idx] for r in rows_bad if r[idx] is not None]
        good_vals = [r[idx] for r in rows_good if r[idx] is not None]
        if not bad_vals or not good_vals:
            continue
        if is_numeric(bad_vals[0]):
            bad_mean = typical_value_numeric(bad_vals)
            good_mean = typical_value_numeric(good_vals)
            diff_val = abs(bad_mean - good_mean)
            diffs.append((diff_val, col_name, round(bad_mean, 4), round(good_mean, 4)))
        else:
            bad_mode = mode_value(bad_vals)
            good_mode = mode_value(good_vals)
            diff_val = 1.0 if bad_mode != good_mode else 0.0
            diffs.append((diff_val, col_name, bad_mode, good_mode))

    # Sort by diff desc, then name asc
    diffs.sort(key=lambda x: (-x[0], x[1]))
    # Keep at least 3
    result = [
        {"feature": name, "bad": bad_val, "good": good_val}
        for _, name, bad_val, good_val in diffs[:max(3, len(diffs))]
    ]
    return result


# ------------------------------------------------------------------
# Main driver
# ------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    header, rows = read_csv(csv_path)

    # Identify indices
    gidx = goal_indices(header)
    non_goals = non_goal_indices(header)

    # Find Mpg+ index if present
    mpg_idx: Optional[int] = None
    for i, h in enumerate(header):
        if h == "Mpg+":
            mpg_idx = i
            break
    if mpg_idx is None:
        print("Error: no Mpg+ column found.")
        sys.exit(1)

    # --------------------------------------------------
    # CAR-09
    # --------------------------------------------------
    print("== CAR-09 ==")

    # Get rows with known Mpg+
    mpg_rows = [(i + 1, r[mpg_idx]) for i, r in enumerate(rows) if r[mpg_idx] is not None]
    mpg_rows.sort(key=lambda x: x[1])  # ascending
    # Determine number to pick: at least 5, at least 10% of rows (rounded down)
    pick_n = max(5, int(len(rows) * 0.10))
    pick_n = min(pick_n, len(mpg_rows))
    poor_rows = [rn for rn, _ in mpg_rows[:pick_n]]
    # Ranking of features
    ranking = most_associated_features(header, rows, mpg_idx)

    # Output some lines
    print(f"Poor rows: {poor_rows}")
    print(f"Ranking: {ranking}")

    # ANSWER line
    print(f"ANSWER CAR-09 {json.dumps({'poor': poor_rows, 'ranking': ranking}, separators=(',', ':'))}")

    # --------------------------------------------------
    # CAR-10
    # --------------------------------------------------
    print("\n== CAR-10 ==")

    # Compute overall scores
    scores, score_rows = overall_scores(rows, gidx)
    # mean score
    mean_score = sum(scores) / len(scores) if scores else 0.0
    bad_mask = [s < mean_score for s in scores]
    good_mask = [s >= mean_score for s in scores]

    # Select at least 5 each
    bad_indices = [score_rows[i] for i, ok in enumerate(bad_mask) if ok]
    good_indices = [score_rows[i] for i, ok in enumerate(good_mask) if ok]
    if len(bad_indices) < 5:
        # take top 5 lowest
        sorted_low = sorted(zip(scores, score_rows), key=lambda x: x[0])[:5]
        bad_indices = [i for _, i in sorted_low]
    if len(good_indices) < 5:
        sorted_high = sorted(zip(scores, score_rows), key=lambda x: -x[0])[:5]
        good_indices = [i for _, i in sorted_high]

    # Rows data
    rows_bad = [rows[i - 1] for i in bad_indices]
    rows_good = [rows[i - 1] for i in good_indices]

    # Differences
    diffs = differences_between_sets(header, rows_bad, rows_good, non_goals, non_goals)

    # Output
    print(f"Bad rows: {bad_indices}")
    print(f"Good rows: {good_indices}")
    print(f"Differences: {diffs}")

    # ANSWER line
    print(
        f"ANSWER CAR-10 {json.dumps({'bad': bad_indices, 'good': good_indices, 'differences': diffs}, separators=(',', ':'))}"
    )


if __name__ == "__main__":
    main()
