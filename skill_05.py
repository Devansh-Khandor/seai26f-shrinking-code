#!/usr/bin/env python3
"""
skill_05.py

Usage:
    python skill_05.py <csv_path>
"""

import csv
import json
import sys
import math
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple

# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
def parse_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read CSV and return rows as list of dicts and header list."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [row for row in reader]
    return rows, header


def to_float(val: str) -> float:
    """Convert string to float; return NaN if missing or not convertible."""
    if val.strip() == "?":
        return float("nan")
    try:
        return float(val)
    except ValueError:
        return float("nan")


def is_numeric(val: str) -> bool:
    """True if val can be parsed as float and is not '?'."""
    if val.strip() == "?":
        return False
    try:
        float(val)
        return True
    except ValueError:
        return False


def mean_std(values: List[float]) -> Tuple[float, float]:
    """Return mean and sample standard deviation (nan-safe)."""
    vals = [v for v in values if not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return (0.0, 0.0)
    mean = sum(vals) / n
    if n == 1:
        return (mean, 0.0)
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    return (mean, math.sqrt(var))


def pearson_corr(x_vals: List[float], y_vals: List[float]) -> float:
    """Compute Pearson correlation coefficient (nan-safe)."""
    n = len(x_vals)
    if n < 2:
        return 0.0
    mean_x = sum(x_vals) / n
    mean_y = sum(y_vals) / n
    num = sum((x_vals[i] - mean_x) * (y_vals[i] - mean_y) for i in range(n))
    den_x = sum((x_vals[i] - mean_x) ** 2 for i in range(n))
    den_y = sum((y_vals[i] - mean_y) ** 2 for i in range(n))
    denom = math.sqrt(den_x * den_y)
    if denom == 0:
        return 0.0
    return num / denom


# ------------------------------------------------------------------
# NASA-09 functions
# ------------------------------------------------------------------
def identify_poor_effort(rows: List[Dict[str, str]],
                         header: List[str]) -> List[int]:
    """Return list of row numbers with unusually high Effort-."""
    effort_vals = [to_float(r["Effort-"]) for r in rows]
    mean, std = mean_std(effort_vals)
    threshold = mean + 2 * std
    poor = [idx + 1 for idx, v in enumerate(effort_vals) if v > threshold]
    # Ensure at least 5
    if len(poor) < 5:
        threshold = mean + 1.5 * std
        poor = [idx + 1 for idx, v in enumerate(effort_vals) if v > threshold]
    if len(poor) < 5:
        # Take top 5
        sorted_indices = sorted(range(len(effort_vals)),
                               key=lambda i: effort_vals[i],
                               reverse=True)
        poor = [sorted_indices[i] + 1 for i in range(min(5, len(sorted_indices)))]
    return poor


def compute_ranking(rows: List[Dict[str, str]],
                    header: List[str],
                    non_goal_cols: List[str],
                    poor_rows: List[int]) -> List[str]:
    """Return list of feature names most associated with poor Effort-."""
    # Gather data for all rows
    # Build list of (row idx, effort, feature values)
    data = []
    for idx, row in enumerate(rows):
        eff = to_float(row["Effort-"])
        if math.isnan(eff):
            continue
        feat = {}
        for col in non_goal_cols:
            val = row[col]
            if val.strip() == "?":
                continue
            if is_numeric(val):
                feat[col] = float(val)
            else:
                feat[col] = val
        data.append((idx + 1, eff, feat))

    # Compute association per column
    assoc: List[Tuple[float, str]] = []  # (strength, col)
    for col in non_goal_cols:
        # collect pairs
        pairs_num = []
        pairs_cat = defaultdict(list)
        for _, eff, feat in data:
            if col not in feat:
                continue
            val = feat[col]
            if isinstance(val, float):
                pairs_num.append((val, eff))
            else:
                pairs_cat[val].append(eff)
        if pairs_num:
            xs, ys = zip(*pairs_num)
            r = pearson_corr(xs, ys)
            assoc.append((abs(r), col))
        elif pairs_cat:
            # categorical: compute max-min diff of mean Effort- per category
            means = []
            for cat, eff_list in pairs_cat.items():
                if eff_list:
                    means.append(sum(eff_list) / len(eff_list))
            if len(means) >= 2:
                diff = max(means) - min(means)
                assoc.append((abs(diff), col))
        # else: no data, skip
    # sort by association descending
    assoc.sort(key=lambda x: x[0], reverse=True)
    ranked = [col for _, col in assoc]
    return ranked[:max(3, len(ranked))]  # at least 3 if possible


# ------------------------------------------------------------------
# NASA-10 functions
# ------------------------------------------------------------------
def identify_bad_good_defects(rows: List[Dict[str, str]]) -> Tuple[List[int], List[int]]:
    """Return bad (high defects) and good (low defects) row numbers."""
    def_vals = [to_float(r["Defects-"]) for r in rows]
    mean, std = mean_std(def_vals)
    threshold = mean + 2 * std
    bad = [idx + 1 for idx, v in enumerate(def_vals) if v > threshold]
    if len(bad) < 5:
        threshold = mean + 1.5 * std
        bad = [idx + 1 for idx, v in enumerate(def_vals) if v > threshold]
    if len(bad) < 5:
        sorted_idx = sorted(range(len(def_vals)), key=lambda i: def_vals[i], reverse=True)
        bad = [sorted_idx[i] + 1 for i in range(min(5, len(sorted_idx)))]

    # Good: lowest 5 defects
    good_candidates = [(i, v) for i, v in enumerate(def_vals) if not math.isnan(v)]
    good_sorted = sorted(good_candidates, key=lambda x: x[1])
    good = [idx + 1 for idx, _ in good_sorted[:5]]
    return bad, good


def typical_value(rows: List[Dict[str, str]],
                  header: List[str],
                  feature: str,
                  rows_idx: List[int]) -> Any:
    """Compute typical value (mean for numeric, mode for categorical)."""
    vals = []
    for idx in rows_idx:
        val = rows[idx - 1][feature]
        if val.strip() == "?":
            continue
        if is_numeric(val):
            vals.append(float(val))
        else:
            vals.append(val)
    if not vals:
        return None
    if isinstance(vals[0], float):
        mean_val = sum(vals) / len(vals)
        return round(mean_val, 4)
    # categorical: mode
    cnt = Counter(vals)
    max_count = max(cnt.values())
    modes = [v for v, c in cnt.items() if c == max_count]
    return sorted(modes)[0]  # deterministic


def compute_differences(rows: List[Dict[str, str]],
                        header: List[str],
                        non_goal_cols: List[str],
                        bad_idx: List[int],
                        good_idx: List[int]) -> List[Dict[str, Any]]:
    """Return list of differences between bad and good rows."""
    diffs = []
    for col in non_goal_cols:
        bad_val = typical_value(rows, header, col, bad_idx)
        good_val = typical_value(rows, header, col, good_idx)
        if bad_val is None or good_val is None:
            continue
        if isinstance(bad_val, float):
            diff = abs(bad_val - good_val)
            diffs.append({"feature": col, "bad": bad_val, "good": good_val, "diff": diff})
        else:
            diff = 1 if bad_val != good_val else 0
            diffs.append({"feature": col, "bad": bad_val, "good": good_val, "diff": diff})
    # sort by diff descending
    diffs.sort(key=lambda x: x["diff"], reverse=True)
    # return up to needed (at least 3)
    result = []
    for d in diffs[:max(3, len(diffs))]:
        result.append({"feature": d["feature"],
                       "bad": d["bad"],
                       "good": d["good"]})
    return result


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(argv: List[str]) -> None:
    if len(argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)

    csv_path = argv[1]
    rows, header = parse_csv(csv_path)

    # Identify goal, X, and non-goal columns
    goal_columns = [c for c in header if c.endswith("+") or c.endswith("-")]
    x_columns = [c for c in header if c.endswith("X")]
    non_goal_cols = [c for c in header if c not in goal_columns and c not in x_columns]

    # ---------- NASA-09 ----------
    poor_rows = identify_poor_effort(rows, header)
    ranking = compute_ranking(rows, header, non_goal_cols, poor_rows)

    print("== NASA-09 ==")
    # Ensure at least 3 lines
    print(f"Poor Effort- rows: {poor_rows}")
    print(f"Top associated features: {ranking}")
    print(f"Number of poor rows: {len(poor_rows)}")
    # Answer JSON
    answer_09 = {"poor": poor_rows, "ranking": ranking}
    print("ANSWER NASA-09 " + json.dumps(answer_09))

    # ---------- NASA-10 ----------
    bad_idx, good_idx = identify_bad_good_defects(rows)

    differences = compute_differences(rows, header, non_goal_cols, bad_idx, good_idx)

    print("\n== NASA-10 ==")
    print(f"Bad defect rows: {bad_idx}")
    print(f"Good defect rows: {good_idx}")
    print(f"Top differences: {differences}")
    # Answer JSON
    answer_10 = {"bad": bad_idx, "good": good_idx, "differences": differences}
    print("ANSWER NASA-10 " + json.dumps(answer_10))

if __name__ == "__main__":
    main(sys.argv)
