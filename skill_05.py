#!/usr/bin/env python3
"""
Skill 05: Advanced analysis – poor fuel efficiency & overall performance characteristics.

Usage:
    python skill_05.py <csv_path>
"""

import sys
import csv
import math
import json
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

# --------------------------------------------------------------------------- #
# Reuse helpers from skill_01
# --------------------------------------------------------------------------- #

try:
    from skill_01 import parse_csv, is_missing, to_float, compute_stats, normalize
except Exception:
    # Minimal fallback implementations
    def parse_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            header = reader.fieldnames if reader.fieldnames else []
        return rows, header

    def is_missing(val: str) -> bool:
        return val == '?' or val is None

    def to_float(val: str) -> float:
        try:
            return float(val)
        except (ValueError, TypeError):
            return math.nan

    def compute_stats(values: List[float]) -> Dict[str, float]:
        n = len(values)
        if n == 0:
            return {"mean": math.nan, "median": math.nan}
        mean_val = sum(values) / n
        sorted_vals = sorted(values)
        mid = n // 2
        if n % 2 == 1:
            med_val = sorted_vals[mid]
        else:
            med_val = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
        return {"mean": mean_val, "median": med_val}

    def normalize(val: float, min_val: float, max_val: float, maximize: bool) -> float:
        if max_val == min_val:
            return 1.0
        return (val - min_val) / (max_val - min_val) if maximize else (max_val - val) / (max_val - min_val)


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def pearson_corr(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n == 0:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denx = sum((xi - mean_x) ** 2 for xi in x)
    deny = sum((yi - mean_y) ** 2 for yi in y)
    if denx == 0 or deny == 0:
        return 0.0
    return num / math.sqrt(denx * deny)


def typical_numeric(values: List[float]) -> float:
    if not values:
        return math.nan
    return sum(values) / len(values)


def typical_categorical(values: List[str]) -> str:
    if not values:
        return ""
    counter = Counter(values)
    most_common, _ = counter.most_common(1)[0]
    return most_common


# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    rows, header = parse_csv(csv_path)

    # Goal columns (ending +/-) and their direction
    goal_columns = {col: col[-1] for col in header if col.endswith('-') or col.endswith('+')}
    non_goal_columns = [col for col in header if col not in goal_columns and not col.endswith('X')]

    # Pre‑compute min/max for goals for scoring
    col_min, col_max = {}, {}
    for col, dirn in goal_columns.items():
        vals = [to_float(r[col]) for r in rows if not is_missing(r[col])]
        if vals:
            col_min[col] = min(vals)
            col_max[col] = max(vals)
        else:
            col_min[col] = col_max[col] = 0.0

    # Compute composite scores for all rows with complete goal data
    scores = []  # (row_index, score)
    for idx, row in enumerate(rows, start=1):
        has_all = all(not is_missing(row[col]) for col in goal_columns)
        if not has_all:
            continue
        score = 0.0
        for col, dirn in goal_columns.items():
            v = to_float(row[col])
            best = normalize(v, col_min[col], col_max[col], maximize=(dirn == '+'))
            score += best
        scores.append((idx, score))

    # Helper to get percentiles
    def percentile_index(n, p):
        idx = int(math.ceil(p * n)) - 1
        return max(0, min(idx, n - 1))

    # ----------------------------------------------------------------------- #
    # CAR-09: Poor fuel efficiency analysis
    # ----------------------------------------------------------------------- #
    mpg_col = 'Mpg+'
    mpg_vals = [to_float(r[mpg_col]) for r in rows if not is_missing(r[mpg_col])]
    if not mpg_vals:
        poor_rows = []
        ranking = []
    else:
        mpg_vals_sorted = sorted(mpg_vals)
        n_mpg = len(mpg_vals_sorted)
        threshold_idx = percentile_index(n_mpg, 0.2)
        threshold = mpg_vals_sorted[threshold_idx]
        poor_rows = [idx for idx, row in enumerate(rows, start=1)
                     if not is_missing(row[mpg_col]) and to_float(row[mpg_col]) <= threshold]
        if len(poor_rows) < 5:
            # fallback: take all rows with mpg
            poor_rows = [idx for idx, row in enumerate(rows, start=1) if not is_missing(row[mpg_col])]

        # Association ranking
        overall_mean_mpg = sum(mpg_vals) / n_mpg
        assoc_strength = {}
        for col in non_goal_columns:
            # Collect pairs where both mpg and feature are present
            x_vals = []
            y_vals = []
            for row in rows:
                if is_missing(row[col]) or is_missing(row[mpg_col]):
                    continue
                fval = to_float(row[col])
                if math.isnan(fval):
                    continue
                x_vals.append(fval)
                y_vals.append(to_float(row[mpg_col]))
            if not x_vals:
                assoc_strength[col] = 0.0
                continue
            # Check if column is numeric or categorical by inspecting values
            if all(isinstance(v, (int, float)) for v in x_vals):
                # Numeric: Pearson correlation
                corr = pearson_corr(x_vals, y_vals)
                assoc_strength[col] = abs(corr)
            else:
                # Categorical: weighted average absolute difference of category means
                cat_counts = defaultdict(int)
                cat_sum_mpg = defaultdict(float)
                for f, mpg in zip(x_vals, y_vals):
                    cat_counts[f] += 1
                    cat_sum_mpg[f] += mpg
                cat_means = {cat: cat_sum_mpg[cat] / cat_counts[cat] for cat in cat_counts}
                weighted_diff = sum(abs(cat_means[cat] - overall_mean_mpg) * (cat_counts[cat] / len(x_vals))
                                    for cat in cat_means)
                assoc_strength[col] = weighted_diff
        # Rank descending
        ranking = sorted(assoc_strength.keys(), key=lambda c: assoc_strength[c], reverse=True)
        ranking = ranking[:max(3, len(ranking))]

    # ----------------------------------------------------------------------- #
    # CAR-10: Overall performance characteristic comparison
    # ----------------------------------------------------------------------- #
    if not scores:
        bad_rows = []
        good_rows = []
        differences = []
    else:
        scores_sorted = sorted(scores, key=lambda kv: kv[1])
        n_scores = len(scores_sorted)
        # Bottom 20% bad, top 20% good
        bad_idx = percentile_index(n_scores, 0.2)
        good_idx = n_scores - percentile_index(n_scores, 0.2) - 1
        bad_rows = [idx for idx, _ in scores_sorted[:bad_idx + 1]]
        good_rows = [idx for idx, _ in scores_sorted[good_idx:]]
        if len(bad_rows) < 5:
            bad_rows = [idx for idx, _ in scores_sorted[:5]]
        if len(good_rows) < 5:
            good_rows = [idx for idx, _ in scores_sorted[-5:]]

        # Compute typical values per group
        def group_typical(col, group_rows):
            values = []
            for idx in group_rows:
                val = rows[idx - 1][col]
                if is_missing(val):
                    continue
                if col in goal_columns or col.endswith('X'):
                    continue  # skip goal columns and X
                if to_float(val) is not None and not math.isnan(to_float(val)):
                    # numeric
                    values.append(to_float(val))
                else:
                    values.append(val)
            if not values:
                return None
            # determine numeric or categorical
            if all(isinstance(v, float) for v in values):
                return round(typical_numeric(values), 4)
            else:
                return typical_categorical(values)

        differences = []
        for col in non_goal_columns:
            bad_val = group_typical(col, bad_rows)
            good_val = group_typical(col, good_rows)
            if bad_val is None or good_val is None:
                continue
            # Compute difference metric
            if isinstance(bad_val, float) and isinstance(good_val, float):
                diff_metric = abs(bad_val - good_val)
            else:
                diff_metric = 1.0 if bad_val != good_val else 0.0
            if diff_metric == 0.0:
                continue
            differences.append({
                "feature": col,
                "bad": bad_val,
                "good": good_val,
                "metric": diff_metric  # internal for sorting
            })
        # Sort by metric descending
        differences.sort(key=lambda d: d["metric"], reverse=True)
        # Keep top 3
        differences = differences[:3]
        # Remove the internal key
        for d in differences:
            d.pop("metric", None)

    # ----------------------------------------------------------------------- #
    # Output
    # ----------------------------------------------------------------------- #
    # CAR-09
    print("== CAR-09 ==")
    print(f"Unusually poor Mpg+ rows: {poor_rows}")
    print("Association ranking:")
    for col_name in ranking:
        print(f"  {col_name}")
    print()

    # CAR-10
    print("== CAR-10 ==")
    print(f"Bad performance rows: {bad_rows}")
    print(f"Good performance rows: {good_rows}")
    print("Differences (most pronounced first):")
    for diff in differences:
        print(f"  {diff['feature']}: bad={diff['bad']}, good={diff['good']}")
    print()

    # Machine-checked answers
    print(f"ANSWER CAR-09 {json.dumps({'poor': poor_rows, 'ranking': ranking}, separators=(',', ':'))}")
    print(f"ANSWER CAR-10 {json.dumps({'bad': bad_rows, 'good': good_rows, 'differences': differences}, separators=(',', ':'))}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skill_05.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
