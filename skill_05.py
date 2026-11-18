#!/usr/bin/env python3
"""
skill_05.py <csv_path>
Implements CAR-09 (fuel‑efficiency outliers) and CAR-10 (overall performance comparison).
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def parse_value(v: str) -> Optional[float]:
    if v == '?' or v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None

def read_csv(csv_path: str) -> Tuple[List[Tuple[int, Dict[str, Optional[float]]]], List[str]]:
    rows: List[Tuple[int, Dict[str, Optional[float]]]] = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for i, r in enumerate(reader, start=1):
            parsed: Dict[str, Optional[float]] = {k: parse_value(v) for k, v in r.items()}
            rows.append((i, parsed))
    return rows, headers

def goal_columns(headers: List[str]) -> List[str]:
    return [h for h in headers if h.endswith('+') or h.endswith('-')]

def min_max_goals(rows: List[Tuple[int, Dict[str, Optional[float]]]], goals: List[str]) -> Dict[str, Tuple[float, float]]:
    minmax: Dict[str, Tuple[float, float]] = {}
    for g in goals:
        vals = [row[1][g] for row in rows if row[1].get(g) is not None]
        if vals:
            minmax[g] = (min(vals), max(vals))
    return minmax

def composite_score(row: Dict[str, Optional[float]],
                    minmax: Dict[str, Tuple[float, float]],
                    goals: List[str]) -> Optional[float]:
    scores = []
    for g in goals:
        v = row.get(g)
        if v is None or g not in minmax:
            return None
        mn, mx = minmax[g]
        if mn == mx:
            norm = 0.0
        else:
            if g.endswith('+'):
                norm = (v - mn) / (mx - mn)
            else:
                norm = (mx - v) / (mx - mn)
        scores.append(norm)
    return sum(scores) / len(scores)

def numeric_columns(headers: List[str], goals: List[str]) -> List[str]:
    return [h for h in headers if h not in goals and is_numeric_column(h)]

def is_numeric_column(col: str) -> bool:
    # Assume all columns are numeric except those that look like strings? We'll treat as numeric
    # by trying to convert first non-? entry; if fails, treat as non-numeric.
    return True  # Simplification: all columns considered numeric for this dataset

def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def mode(values: List[float]) -> float:
    if not values:
        return None
    counts = Counter(values)
    return counts.most_common(1)[0][0]

def categorical_mode(values: List[str]) -> str:
    if not values:
        return None
    counts = Counter(values)
    return counts.most_common(1)[0][0]

def filter_rows_with_goal(rows: List[Tuple[int, Dict[str, Optional[float]]]],
                          goals: List[str]) -> List[Tuple[int, Dict[str, Optional[float]]]]:
    return [row for row in rows if all(row[1].get(g) is not None for g in goals)]

# --------------------------------------------------------------------------- #
# Analysis functions
# --------------------------------------------------------------------------- #

def car_09(rows: List[Tuple[int, Dict[str, Optional[float]]]],
          headers: List[str],
          minmax: Dict[str, Tuple[float, float]],
          goals: List[str]) -> Dict:
    # Determine poor Mpg+ threshold (bottom 10% or at least 5)
    mpg_col = next(g for g in goals if g.endswith('+') and g == 'Mpg+')
    mpg_vals = [(r[0], r[1][mpg_col]) for r in rows if r[1].get(mpg_col) is not None]
    mpg_vals_sorted = sorted(mpg_vals, key=lambda x: x[1])  # ascending
    n = len(mpg_vals_sorted)
    threshold_idx = max(4, int(n * 0.1) - 1)  # at least 5
    threshold = mpg_vals_sorted[threshold_idx][1]
    poor_rows = [idx for idx, val in mpg_vals_sorted if val <= threshold]

    # Prepare feature statistics
    non_goals = [h for h in headers if h not in goals]
    # Overall means for numeric columns
    overall_means: Dict[str, float] = {}
    for col in non_goals:
        vals = [r[1][col] for r in rows if r[1].get(col) is not None]
        if vals:
            overall_means[col] = mean(vals)
    # Means for poor rows
    poor_means: Dict[str, float] = {}
    for col in non_goals:
        vals = [r[1][col] for r in rows if r[0] in poor_rows and r[1].get(col) is not None]
        if vals:
            poor_means[col] = mean(vals)

    # Ranking of features by absolute difference
    diffs: List[Tuple[str, float]] = []
    for col in non_goals:
        if col in overall_means and col in poor_means:
            diff = abs(overall_means[col] - poor_means[col])
            diffs.append((col, diff))
        else:
            # For categorical, simple indicator if modes differ
            vals_all = [r[1][col] for r in rows if r[1].get(col) is not None]
            vals_poor = [r[1][col] for r in rows if r[0] in poor_rows and r[1].get(col) is not None]
            if vals_all and vals_poor:
                mode_all = categorical_mode([str(v) for v in vals_all])
                mode_poor = categorical_mode([str(v) for v in vals_poor])
                diff = 0.0 if mode_all == mode_poor else 1.0
                diffs.append((col, diff))

    diffs.sort(key=lambda x: (-x[1], x[0]))
    ranking = [col for col, _ in diffs[:3]]

    return {"poor": poor_rows, "ranking": ranking}

def car_10(rows: List[Tuple[int, Dict[str, Optional[float]]]],
          headers: List[str],
          minmax: Dict[str, Tuple[float, float]],
          goals: List[str]) -> Dict:
    # Compute composite scores for rows with all goal values
    scored_rows = [(idx, composite_score(data, minmax, goals)) for idx, data in rows if composite_score(data, minmax, goals) is not None]
    scored_rows.sort(key=lambda x: x[1])  # ascending
    n = len(scored_rows)
    # Bottom 25% bad
    bad_count = max(5, int(n * 0.25))
    bad_rows = [idx for idx, _ in scored_rows[:bad_count]]
    # Top 25% good
    good_count = max(5, int(n * 0.25))
    good_rows = [idx for idx, _ in scored_rows[-good_count:]]

    non_goals = [h for h in headers if h not in goals]

    # Compute typical values
    def typical(group: List[int]) -> Tuple[Dict[str, float], Dict[str, str]]:
        numeric_vals: Dict[str, List[float]] = defaultdict(list)
        categorical_vals: Dict[str, List[str]] = defaultdict(list)
        for idx in group:
            data = next(r for r in rows if r[0] == idx)[1]
            for col in non_goals:
                val = data.get(col)
                if val is None:
                    continue
                if isinstance(val, float) or isinstance(val, int):
                    numeric_vals[col].append(val)
                else:
                    categorical_vals[col].append(str(val))
        numeric_mean: Dict[str, float] = {col: mean(vals) for col, vals in numeric_vals.items()}
        cat_mode: Dict[str, str] = {col: categorical_mode(vals) for col, vals in categorical_vals.items()}
        return numeric_mean, cat_mode

    bad_num_mean, bad_cat_mode = typical(bad_rows)
    good_num_mean, good_cat_mode = typical(good_rows)

    diffs: List[Tuple[str, float, str, str]] = []
    for col in non_goals:
        if col in bad_num_mean and col in good_num_mean:
            diff = abs(bad_num_mean[col] - good_num_mean[col])
            diffs.append((col, diff, bad_num_mean[col], good_num_mean[col]))
        elif col in bad_cat_mode and col in good_cat_mode:
            diff = 0.0 if bad_cat_mode[col] == good_cat_mode[col] else 1.0
            diffs.append((col, diff, bad_cat_mode[col], good_cat_mode[col]))

    diffs.sort(key=lambda x: (-x[1], x[0]))
    diff_output = []
    for col, diff, bad_val, good_val in diffs[:3]:
        if isinstance(bad_val, float) or isinstance(bad_val, int):
            bad_val_json = bad_val
            good_val_json = good_val
        else:
            bad_val_json = bad_val
            good_val_json = good_val
        diff_output.append({
            "feature": col,
            "bad": bad_val_json,
            "good": good_val_json
        })

    return {"bad": bad_rows, "good": good_rows, "differences": diff_output}

# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #

def main(csv_path: str):
    rows, headers = read_csv(csv_path)
    goals = goal_columns(headers)
    minmax = min_max_goals(rows, goals)

    # CAR-09
    print("== CAR-09 ==")
    res09 = car_09(rows, headers, minmax, goals)
    print(f"Poor Mpg+ rows (bottom 10%): {res09['poor']}")
    print(f"Top 3 features most associated with poor Mpg+: {res09['ranking']}")
    print(f"ANSWER CAR-09 {json.dumps(res09)}")

    # CAR-10
    print("\n== CAR-10 ==")
    res10 = car_10(rows, headers, minmax, goals)
    print(f"Bad overall performance rows (bottom 25%): {res10['bad']}")
    print(f"Good overall performance rows (top 25%): {res10['good']}")
    print("Top 3 characteristic differences:")
    for d in res10['differences']:
        print(f"  {d['feature']}: bad={d['bad']}, good={d['good']}")
    print(f"ANSWER CAR-10 {json.dumps(res10)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python skill_05.py <csv_path>\n")
        sys.exit(1)
    main(sys.argv[1])
