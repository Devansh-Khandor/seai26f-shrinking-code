#!/usr/bin/env python3
"""
skill_05.py

Implements CAR-09 and CAR-10 for the auto93 dataset.

CAR-09:
    Identify rows with unusually poor fuel efficiency (low Mpg+) and
    rank the non‑goal features most strongly associated with that outcome.

CAR-10:
    Find the common characteristics of vehicles with unfavorable overall
    performance versus those with favorable performance and
    quantify the differences.

The script is self‑contained and uses only the Python standard library.
"""

import csv
import json
import sys
import math
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Set

# ------------------------------------------------------------------
# Data utilities
# ------------------------------------------------------------------
def read_data(path: str) -> List[Dict[str, Any]]:
    """
    Read CSV into list of dicts.
    '?' or '' -> None.
    numeric strings -> float.
    others keep as str.
    """
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if v == '?' or v == '':
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows


def goal_columns(rows: List[Dict[str, Any]]) -> Set[str]:
    """
    Return set of column names that are goals.
    Goals end with '-' (minimize) or '+' (maximize).
    """
    return {k for k in rows[0] if k.endswith('-') or k.endswith('+')}


def non_goal_columns(rows: List[Dict[str, Any]]) -> Set[str]:
    """
    Return set of column names that are not goals and not X columns.
    """
    goals = goal_columns(rows)
    return {k for k in rows[0] if k not in goals and not k.endswith('X')}


# ------------------------------------------------------------------
# Statistics helpers
# ------------------------------------------------------------------
def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: List[float]) -> float:
    if not values:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def mode(values: List[Any]) -> Any:
    if not values:
        return None
    counter = Counter(values)
    # choose the smallest value in case of ties (for deterministic output)
    max_count = max(counter.values())
    modes = [k for k, v in counter.items() if v == max_count]
    return sorted(modes)[0]


# ------------------------------------------------------------------
# CAR-09 helpers
# ------------------------------------------------------------------
def identify_poor_rows(rows: List[Dict[str, Any]], target: str, n: int = 5) -> List[int]:
    """
    Return a list of row numbers (1‑based) of the worst (lowest)
    target values. Rows with missing target are ignored.
    """
    available = [(idx + 1, r[target]) for idx, r in enumerate(rows) if r[target] is not None]
    available.sort(key=lambda x: x[1])  # ascending
    return [idx for idx, _ in available[:n]]


def compute_association(rows: List[Dict[str, Any]],
                       poor_set: Set[int],
                       non_goal_cols: Set[str],
                       goal: str) -> List[str]:
    """
    Compute an association score for each non‑goal column.
    For numeric columns, use absolute difference of means between
    poor rows and the rest.
    For categorical columns, use absolute difference of mode
    frequencies between the two groups.
    Return the column names sorted by descending score.
    """
    # Separate poor and non‑poor rows
    poor_rows = [rows[idx - 1] for idx in poor_set]
    non_poor_rows = [r for i, r in enumerate(rows, start=1) if i not in poor_set]

    scores = {}
    for col in non_goal_cols:
        # Check if numeric
        is_numeric = True
        for r in poor_rows:
            v = r.get(col)
            if v is not None and not isinstance(v, float):
                is_numeric = False
                break
        if is_numeric:
            # compute means (skip None)
            poor_vals = [r[col] for r in poor_rows if r[col] is not None]
            rest_vals = [r[col] for r in non_poor_rows if r[col] is not None]
            if not poor_vals or not rest_vals:
                score = 0.0
            else:
                score = abs(mean(poor_vals) - mean(rest_vals))
        else:
            # categorical
            poor_vals = [r[col] for r in poor_rows if r[col] is not None]
            rest_vals = [r[col] for r in non_poor_rows if r[col] is not None]
            if not poor_vals or not rest_vals:
                score = 0.0
            else:
                mode_poor = mode(poor_vals)
                mode_rest = mode(rest_vals)
                # compute frequency difference
                freq_poor = poor_vals.count(mode_poor) / len(poor_vals)
                freq_rest = rest_vals.count(mode_rest) / len(rest_vals)
                score = abs(freq_poor - freq_rest)
        scores[col] = score

    # Sort by score descending
    sorted_cols = sorted(scores.items(), key=lambda x: -x[1])
    return [col for col, _ in sorted_cols]


# ------------------------------------------------------------------
# CAR-10 helpers
# ------------------------------------------------------------------
def performance_score(row: Dict[str, Any],
                      goals: Set[str],
                      mins: Dict[str, float],
                      maxs: Dict[str, float]) -> float:
    """
    Compute overall performance score for a row.
    For goals ending with '-': minimize, so higher is better.
    For goals ending with '+': maximize, so higher is better.
    Normalise each goal to [0,1] and sum.
    """
    score = 0.0
    for col in goals:
        v = row.get(col)
        if v is None:
            return None
        mn = mins[col]
        mx = maxs[col]
        if mx == mn:
            norm = 0.0
        else:
            if col.endswith('-'):
                norm = (mx - v) / (mx - mn)
            else:
                norm = (v - mn) / (mx - mn)
        score += norm
    return score


def compute_perf_scores(rows: List[Dict[str, Any]], goals: Set[str]) -> List[Tuple[int, float]]:
    """
    Return list of (row_number, score). Skip rows that cannot be scored.
    """
    # Compute min and max per goal over all rows that have the goal
    mins, maxs = {}, {}
    for col in goals:
        vals = [r[col] for r in rows if r[col] is not None]
        mins[col] = min(vals) if vals else 0.0
        maxs[col] = max(vals) if vals else 0.0

    scores = []
    for idx, row in enumerate(rows, start=1):
        score = performance_score(row, goals, mins, maxs)
        if score is not None:
            scores.append((idx, score))
    return scores


def identify_perf_groups(scores: List[Tuple[int, float]], n: int = 5) -> Tuple[List[int], List[int]]:
    """
    From scored rows, return bad (bottom n) and good (top n) row numbers.
    """
    sorted_by_score = sorted(scores, key=lambda x: x[1])  # ascending
    bad = [idx for idx, _ in sorted_by_score[:n]]
    sorted_desc = sorted(scores, key=lambda x: -x[1])
    good = [idx for idx, _ in sorted_desc[:n]]
    return bad, good


def typical_value(values: List[Any]) -> Any:
    """
    Return mean for numeric, mode for others.
    """
    if not values:
        return None
    if isinstance(values[0], float):
        return round(mean(values), 2)
    return mode(values)


def differences_between_groups(rows: List[Dict[str, Any]],
                               bad_set: Set[int],
                               good_set: Set[int],
                               non_goal_cols: Set[str]) -> List[Dict[str, Any]]:
    """
    For each non‑goal column, compute typical value in bad and good groups
    and the difference. Return a list sorted by difference descending.
    """
    bad_rows = [rows[i - 1] for i in bad_set]
    good_rows = [rows[i - 1] for i in good_set]

    diff_list = []
    for col in non_goal_cols:
        bad_vals = [r[col] for r in bad_rows if r[col] is not None]
        good_vals = [r[col] for r in good_rows if r[col] is not None]

        if not bad_vals or not good_vals:
            continue

        bad_typ = typical_value(bad_vals)
        good_typ = typical_value(good_vals)

        # compute numeric difference
        if isinstance(bad_typ, float) and isinstance(good_typ, float):
            diff = abs(bad_typ - good_typ)
        else:
            # categorical: 0 if same else 1
            diff = 0.0 if bad_typ == good_typ else 1.0

        diff_list.append({
            "feature": col,
            "bad": bad_typ,
            "good": good_typ,
            "diff": diff
        })

    # sort by diff descending, then by feature name
    diff_list.sort(key=lambda x: (-x["diff"], x["feature"]))
    # drop helper diff field
    for d in diff_list:
        del d["diff"]
    return diff_list


# ------------------------------------------------------------------
# Main routine
# ------------------------------------------------------------------
def main(csv_path: str):
    rows = read_data(csv_path)
    goals = goal_columns(rows)
    non_goals = non_goal_columns(rows)

    # CAR-09
    print("== CAR-09 ==")
    # Identify poorly fuel efficient rows (lowest Mpg+)
    mpg_col = "Mpg+"
    poor_rows = identify_poor_rows(rows, mpg_col, n=5)
    print(f"Poor rows (lowest {mpg_col}): {poor_rows}")

    # Ranking of associated features
    ranking = compute_association(rows, set(poor_rows), non_goals, mpg_col)
    print("Feature ranking (most associated first):")
    for idx, col in enumerate(ranking[:5], 1):
        print(f"{idx}. {col}")

    answer09 = {
        "poor": poor_rows,
        "ranking": ranking[:5]  # at least 3
    }
    print(f"ANSWER CAR-09 {json.dumps(answer09, ensure_ascii=False)}")

    # CAR-10
    print("\n== CAR-10 ==")
    # Compute performance scores
    perf_scores = compute_perf_scores(rows, goals)
    bad, good = identify_perf_groups(perf_scores, n=5)
    print(f"Bad rows (lowest performance): {bad}")
    print(f"Good rows (highest performance): {good}")

    # Differences in typical feature values
    diffs = differences_between_groups(rows, set(bad), set(good), non_goals)
    print("Top differences between bad and good groups:")
    for d in diffs[:5]:
        print(f"{d['feature']}: bad={d['bad']} good={d['good']}")

    answer10 = {
        "bad": bad,
        "good": good,
        "differences": diffs[:5]  # at least 3
    }
    print(f"ANSWER CAR-10 {json.dumps(answer10, ensure_ascii=False)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
